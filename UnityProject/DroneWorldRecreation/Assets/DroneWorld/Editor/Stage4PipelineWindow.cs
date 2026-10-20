using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using DroneWorld.Stage2;
using UnityEditor;
using UnityEngine;
using Debug = UnityEngine.Debug;
using Process = System.Diagnostics.Process;

namespace DroneWorld.Stage4.Editor
{
    [InitializeOnLoad]
    internal static class Stage4PipelineRunner
    {
        private static readonly ConcurrentQueue<string> PendingLines = new();
        private static readonly List<string> OutputLines = new();
        private static readonly List<string> GateFailures = new();
        private static readonly List<string> Warnings = new();
        private static Process process;
        private static string outputDirectory;
        private static Stage2CalibrationAsset calibration;

        public static event Action Changed;
        public static bool IsRunning => process != null && !process.HasExited;
        public static string Status { get; private set; } = "Ready";
        public static IReadOnlyList<string> LogLines => OutputLines;
        public static IReadOnlyList<string> FailureReasons => GateFailures;
        public static IReadOnlyList<string> ReportWarnings => Warnings;

        static Stage4PipelineRunner()
        {
            EditorApplication.update += Update;
        }

        public static void Start(
            Stage2CalibrationAsset sourceCalibration, string outputPath,
            string externalLabels)
        {
            if (IsRunning)
                throw new InvalidOperationException("Stage 4A is already running.");
            if (sourceCalibration == null || !sourceCalibration.IsUsable)
                throw new InvalidDataException(
                    "Open or import a usable Stage 2 calibration first.");
            if (string.IsNullOrWhiteSpace(outputPath))
                throw new ArgumentException(
                    "Select a Stage 4A output folder.", nameof(outputPath));
            if (!string.IsNullOrWhiteSpace(externalLabels) &&
                !File.Exists(externalLabels))
                throw new FileNotFoundException(
                    "The corrected canonical label PNG does not exist.",
                    externalLabels);

            string stage1Bundle =
                sourceCalibration.SourceManifest.SourceBundlePath;
            string stage2Bundle = sourceCalibration.SourceCalibrationPath;
            if (!File.Exists(Path.Combine(stage1Bundle, "report.json")))
                throw new FileNotFoundException(
                    "The source Stage 1 report is missing.", stage1Bundle);
            if (!File.Exists(Path.Combine(stage2Bundle, "world_calibration.json")))
                throw new FileNotFoundException(
                    "The source Stage 2 calibration is missing.", stage2Bundle);

            string repository = RepositoryRoot;
            string python = Path.Combine(
                repository, ".venv", "Scripts", "python.exe");
            string script = Path.Combine(repository, "stage4.py");
            if (!File.Exists(python))
                throw new FileNotFoundException(
                    "Python environment is missing. Create .venv and install requirements.",
                    python);
            if (!File.Exists(script))
                throw new FileNotFoundException("stage4.py is missing.", script);
            if (string.IsNullOrWhiteSpace(externalLabels))
            {
                string model = Path.Combine(
                    repository, "models", "openearthmap-mask2former",
                    "model.safetensors");
                if (!File.Exists(model))
                    throw new FileNotFoundException(
                    "Stage 4 aerial model is missing. Run setup_stage4_model.py first.",
                        model);
                string roadModel = Path.Combine(
                    repository, "models", "yolo26x-sem.pt");
                if (!File.Exists(roadModel))
                    throw new FileNotFoundException(
                        "Stage 4 road-specialist model is missing. Run " +
                        "setup_stage4_model.py first.", roadModel);
            }

            calibration = sourceCalibration;
            outputDirectory = Path.GetFullPath(outputPath);
            OutputLines.Clear();
            GateFailures.Clear();
            Warnings.Clear();
            while (PendingLines.TryDequeue(out _)) { }
            Append($"Stage 1: {stage1Bundle}");
            Append($"Stage 2: {stage2Bundle}");
            Append($"Artifacts: {outputDirectory}");
            Append(string.IsNullOrWhiteSpace(externalLabels)
                ? "Mode: OpenEarthMap aerial semantics + conservative road specialist."
                : $"Mode: operator-corrected canonical labels: {externalLabels}");
            Append(
                "A pass creates only a review scene. Geometry remains blocked until acceptance.");

            string arguments =
                $"-u -B {Quote(script)} {Quote(stage1Bundle)} " +
                $"{Quote(stage2Bundle)} --output {Quote(outputDirectory)}";
            if (!string.IsNullOrWhiteSpace(externalLabels))
                arguments += $" --semantic-labels {Quote(externalLabels)}";
            var startInfo = new ProcessStartInfo
            {
                FileName = python,
                Arguments = arguments,
                WorkingDirectory = repository,
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
            };
            startInfo.EnvironmentVariables["PYTHONUNBUFFERED"] = "1";
            process = new Process { StartInfo = startInfo };
            process.OutputDataReceived += CaptureLine;
            process.ErrorDataReceived += CaptureLine;
            if (!process.Start())
                throw new InvalidOperationException(
                    "Failed to start the Stage 4A Python process.");
            process.BeginOutputReadLine();
            process.BeginErrorReadLine();
            Status = "Creating a semantic proposal...";
            Changed?.Invoke();
        }

        private static void CaptureLine(
            object sender, DataReceivedEventArgs eventArguments)
        {
            if (!string.IsNullOrWhiteSpace(eventArguments.Data))
                PendingLines.Enqueue(eventArguments.Data);
        }

        private static void Update()
        {
            bool appended = false;
            while (PendingLines.TryDequeue(out string line))
            {
                Append(line);
                appended = true;
            }
            if (appended)
                Changed?.Invoke();
            if (process == null || !process.HasExited)
                return;

            process.WaitForExit();
            int exitCode = process.ExitCode;
            process.Dispose();
            process = null;
            while (PendingLines.TryDequeue(out string line))
                Append(line);
            if (exitCode == 0)
            {
                try
                {
                    ReadReportMessages();
                    Status = "Proposal passed structurally. Importing Unity review scene...";
                    Changed?.Invoke();
                    Stage4BundleImporter.ImportBundle(outputDirectory, calibration);
                    Status =
                        "PASS - Stage4SemanticReview.unity opened; geometry is still blocked.";
                }
                catch (Exception exception)
                {
                    Status = "Unity import failed; the previous scene was preserved.";
                    Append(exception.ToString());
                    Debug.LogException(exception);
                }
            }
            else if (exitCode == 2)
            {
                Status = "FAILED STRUCTURAL GATE - Unity scene was not changed.";
                ReadReportMessages();
            }
            else
            {
                Status = $"Stage 4A process failed with exit code {exitCode}.";
            }
            Append(Status);
            Changed?.Invoke();
        }

        private static void ReadReportMessages()
        {
            try
            {
                Stage4ReportData report = JsonUtility.FromJson<Stage4ReportData>(
                    File.ReadAllText(Path.Combine(outputDirectory, "report.json")));
                GateFailures.Clear();
                Warnings.Clear();
                foreach (string reason in
                         report?.failure_reasons ?? Array.Empty<string>())
                {
                    GateFailures.Add(reason);
                    Append($"Gate: {reason}");
                }
                foreach (string warning in
                         report?.warnings ?? Array.Empty<string>())
                {
                    Warnings.Add(warning);
                    Append($"Warning: {warning}");
                }
            }
            catch (Exception exception)
            {
                Append($"Could not read Stage 4A report: {exception.Message}");
            }
        }

        private static void Append(string line)
        {
            OutputLines.Add(line);
            if (OutputLines.Count > 1500)
                OutputLines.RemoveRange(0, OutputLines.Count - 1500);
        }

        private static string Quote(string value) =>
            $"\"{value.Replace("\"", "\\\"")}\"";

        public static string RepositoryRoot
        {
            get
            {
                string project = Directory.GetParent(Application.dataPath)!.FullName;
                string unityProjects = Directory.GetParent(project)!.FullName;
                return Directory.GetParent(unityProjects)!.FullName;
            }
        }
    }

    public sealed class Stage4PipelineWindow : EditorWindow
    {
        private const string OutputPreference = "DroneWorld.Stage4.Output";
        private const string LabelsPreference = "DroneWorld.Stage4.ExternalLabels";
        private Stage2CalibrationAsset calibration;
        private string outputPath;
        private string externalLabels;
        private Vector2 logScroll;

        [MenuItem("Drone World/Stage 4/Run Stage 4A Semantic Mapping...")]
        public static void ShowWindow()
        {
            GetWindow<Stage4PipelineWindow>("Stage 4A Pipeline");
        }

        private void OnEnable()
        {
            try
            {
                calibration = Stage4BundleImporter.FindCurrentCalibration();
            }
            catch
            {
                calibration = null;
            }
            outputPath = EditorPrefs.GetString(
                OutputPreference,
                Path.Combine(Stage4PipelineRunner.RepositoryRoot, "stage4_out"));
            externalLabels = EditorPrefs.GetString(LabelsPreference, string.Empty);
            Stage4PipelineRunner.Changed += OnRunnerChanged;
        }

        private void OnDisable()
        {
            Stage4PipelineRunner.Changed -= OnRunnerChanged;
        }

        private void OnRunnerChanged()
        {
            logScroll.y = float.MaxValue;
            Repaint();
        }

        private void OnGUI()
        {
            EditorGUILayout.Space(8f);
            EditorGUILayout.LabelField(
                "Static ground -> semantic proposal -> operator review",
                EditorStyles.boldLabel);
            EditorGUILayout.HelpBox(
                "Stage 4A proposes road, pavement, building, vegetation, terrain, " +
                "water, and other-static labels. It does not estimate building height " +
                "and cannot generate geometry until you review and accept the map.",
                MessageType.Info);
            using (new EditorGUI.DisabledScope(Stage4PipelineRunner.IsRunning))
            {
                calibration = (Stage2CalibrationAsset)EditorGUILayout.ObjectField(
                    "Stage 2 calibration", calibration,
                    typeof(Stage2CalibrationAsset), false);
                DrawFolderField();
                DrawLabelsField();
            }
            if (string.IsNullOrWhiteSpace(externalLabels))
            {
                EditorGUILayout.HelpBox(
                    "Automatic mode uses OpenEarthMap aerial semantics. A second model " +
                    "may only promote road pixels; it cannot propose buildings. " +
                    "Resolution and rendering style can still differ, so review remains mandatory.",
                    MessageType.Info);
            }
            using (new EditorGUI.DisabledScope(
                       Stage4PipelineRunner.IsRunning || calibration == null))
            {
                if (GUILayout.Button("Run Stage 4A", GUILayout.Height(32f)))
                    RunPipeline();
            }

            EditorGUILayout.Space(8f);
            EditorGUILayout.LabelField("Status", EditorStyles.boldLabel);
            MessageType statusType = Stage4PipelineRunner.Status.StartsWith("PASS")
                ? MessageType.Info
                : Stage4PipelineRunner.Status.Contains("FAIL")
                    ? MessageType.Error : MessageType.None;
            EditorGUILayout.HelpBox(Stage4PipelineRunner.Status, statusType);
            foreach (string reason in Stage4PipelineRunner.FailureReasons)
                EditorGUILayout.HelpBox(reason, MessageType.Error);
            foreach (string warning in Stage4PipelineRunner.ReportWarnings)
                EditorGUILayout.HelpBox(warning, MessageType.Warning);

            EditorGUILayout.LabelField("Pipeline log", EditorStyles.boldLabel);
            logScroll = EditorGUILayout.BeginScrollView(
                logScroll, GUILayout.MinHeight(180f));
            string logText = string.Join("\n", Stage4PipelineRunner.LogLines);
            var style = new GUIStyle(EditorStyles.textArea) { wordWrap = true };
            float width = Mathf.Max(100f, position.width - 38f);
            float height = Mathf.Max(
                180f, style.CalcHeight(new GUIContent(logText), width));
            EditorGUILayout.SelectableLabel(
                logText, style, GUILayout.Height(height));
            EditorGUILayout.EndScrollView();
        }

        private void DrawFolderField()
        {
            EditorGUILayout.BeginHorizontal();
            outputPath = EditorGUILayout.TextField("Artifact folder", outputPath);
            if (GUILayout.Button("Browse", GUILayout.Width(72f)))
            {
                string selected = EditorUtility.OpenFolderPanel(
                    "Select Stage 4A artifact folder", outputPath, "");
                if (!string.IsNullOrEmpty(selected))
                    outputPath = selected;
            }
            EditorGUILayout.EndHorizontal();
        }

        private void DrawLabelsField()
        {
            EditorGUILayout.BeginHorizontal();
            externalLabels = EditorGUILayout.TextField(
                "Corrected labels (optional)", externalLabels);
            if (GUILayout.Button("Browse", GUILayout.Width(72f)))
            {
                string selected = EditorUtility.OpenFilePanel(
                    "Select canonical one-channel semantic PNG",
                    string.IsNullOrWhiteSpace(externalLabels)
                        ? Stage4PipelineRunner.RepositoryRoot
                        : Path.GetDirectoryName(externalLabels),
                    "png");
                if (!string.IsNullOrEmpty(selected))
                    externalLabels = selected;
            }
            EditorGUILayout.EndHorizontal();
            if (!string.IsNullOrWhiteSpace(externalLabels) &&
                GUILayout.Button("Clear corrected labels"))
                externalLabels = string.Empty;
        }

        private void RunPipeline()
        {
            try
            {
                EditorPrefs.SetString(OutputPreference, outputPath);
                EditorPrefs.SetString(LabelsPreference, externalLabels);
                Stage4PipelineRunner.Start(
                    calibration, outputPath, externalLabels);
            }
            catch (Exception exception)
            {
                Debug.LogException(exception);
                EditorUtility.DisplayDialog(
                    "Could not start Stage 4A", exception.Message, "OK");
            }
        }
    }
}
