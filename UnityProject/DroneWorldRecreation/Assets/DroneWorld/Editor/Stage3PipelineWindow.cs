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

namespace DroneWorld.Stage3.Editor
{
    [InitializeOnLoad]
    internal static class Stage3PipelineRunner
    {
        private static readonly ConcurrentQueue<string> PendingLines = new();
        private static readonly List<string> OutputLines = new();
        private static readonly List<string> GateFailures = new();
        private static Process process;
        private static string outputDirectory;
        private static Stage2CalibrationAsset calibration;

        public static event Action Changed;
        public static bool IsRunning => process != null && !process.HasExited;
        public static string Status { get; private set; } = "Ready";
        public static IReadOnlyList<string> LogLines => OutputLines;
        public static IReadOnlyList<string> FailureReasons => GateFailures;

        static Stage3PipelineRunner()
        {
            EditorApplication.update += Update;
        }

        public static void Start(
            Stage2CalibrationAsset sourceCalibration, string outputPath)
        {
            if (IsRunning)
                throw new InvalidOperationException("Stage 3 is already running.");
            if (sourceCalibration == null || !sourceCalibration.IsUsable)
                throw new InvalidDataException(
                    "Open or import a usable Stage 2 calibration first.");
            if (string.IsNullOrWhiteSpace(outputPath))
                throw new ArgumentException(
                    "Select a Stage 3 output folder.", nameof(outputPath));
            string stage1Bundle = sourceCalibration.SourceManifest.SourceBundlePath;
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
            string script = Path.Combine(repository, "stage3.py");
            if (!File.Exists(python))
                throw new FileNotFoundException(
                    "Python environment is missing. Create .venv and install requirements.",
                    python);
            if (!File.Exists(script))
                throw new FileNotFoundException("stage3.py is missing.", script);

            calibration = sourceCalibration;
            outputDirectory = Path.GetFullPath(outputPath);
            OutputLines.Clear();
            GateFailures.Clear();
            while (PendingLines.TryDequeue(out _)) { }
            Append($"Stage 1: {stage1Bundle}");
            Append($"Stage 2: {stage2Bundle}");
            Append($"Artifacts: {outputDirectory}");
            Append(
                "High-accuracy full-frame + tiled + TTA detection runs on every CFR frame.");
            Append("Unity will be modified only if the Stage 3 quality gate passes.");

            var startInfo = new ProcessStartInfo
            {
                FileName = python,
                Arguments =
                    $"-u -B {Quote(script)} {Quote(stage1Bundle)} " +
                    $"{Quote(stage2Bundle)} --output {Quote(outputDirectory)}",
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
                    "Failed to start the Stage 3 Python process.");
            process.BeginOutputReadLine();
            process.BeginErrorReadLine();
            Status = "Detecting and tracking cars/people on the exact CFR timeline...";
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
                    Status = "Tracks passed. Importing and building Unity playback...";
                    Changed?.Invoke();
                    Stage3BundleImporter.ImportBundle(outputDirectory, calibration);
                    Status = "PASS - Stage3ActorPlayback.unity was created and opened.";
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
                Status = "FAILED QUALITY GATE - Unity scene was not changed.";
                AppendFailureReasons();
            }
            else
            {
                Status = $"Stage 3 process failed with exit code {exitCode}.";
            }
            Append(Status);
            Changed?.Invoke();
        }

        private static void AppendFailureReasons()
        {
            try
            {
                Stage3ReportData report = JsonUtility.FromJson<Stage3ReportData>(
                    File.ReadAllText(Path.Combine(outputDirectory, "report.json")));
                foreach (string reason in
                         report?.failure_reasons ?? Array.Empty<string>())
                {
                    GateFailures.Add(reason);
                    Append($"Gate: {reason}");
                }
            }
            catch (Exception exception)
            {
                Append($"Could not read failure report: {exception.Message}");
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

    public sealed class Stage3PipelineWindow : EditorWindow
    {
        private const string OutputPreference = "DroneWorld.Stage3.Output";
        private Stage2CalibrationAsset calibration;
        private string outputPath;
        private Vector2 logScroll;

        [MenuItem("Drone World/Stage 3/Run Stage 3 Tracking Pipeline...")]
        public static void ShowWindow()
        {
            GetWindow<Stage3PipelineWindow>("Stage 3 Pipeline");
        }

        private void OnEnable()
        {
            try
            {
                calibration = Stage3BundleImporter.FindCurrentCalibration();
            }
            catch
            {
                calibration = null;
            }
            outputPath = EditorPrefs.GetString(
                OutputPreference,
                Path.Combine(Stage3PipelineRunner.RepositoryRoot, "stage3_out"));
            Stage3PipelineRunner.Changed += OnRunnerChanged;
        }

        private void OnDisable()
        {
            Stage3PipelineRunner.Changed -= OnRunnerChanged;
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
                "Validated video -> real car/person tracks -> Unity playback",
                EditorStyles.boldLabel);
            EditorGUILayout.HelpBox(
                "Stage 3 uses the exact Stage 1 CFR video and Stage 2 mapping. " +
                "The accuracy-first detector runs full-frame and overlapping tiles " +
                "with TTA on every frame, so this can take substantially longer than Stage 1.",
                MessageType.Info);
            using (new EditorGUI.DisabledScope(Stage3PipelineRunner.IsRunning))
            {
                calibration = (Stage2CalibrationAsset)EditorGUILayout.ObjectField(
                    "Stage 2 calibration", calibration,
                    typeof(Stage2CalibrationAsset), false);
                DrawOutputField();
            }
            if (calibration != null && calibration.IsApproximateScale)
            {
                EditorGUILayout.HelpBox(
                    "Scale is approximate: placement follows the selected Unity map, " +
                    "but speed is not a real-world measurement.", MessageType.Warning);
            }
            using (new EditorGUI.DisabledScope(
                       Stage3PipelineRunner.IsRunning || calibration == null))
            {
                if (GUILayout.Button("Run Stage 3", GUILayout.Height(32f)))
                    RunPipeline();
            }
            EditorGUILayout.Space(8f);
            EditorGUILayout.LabelField("Status", EditorStyles.boldLabel);
            MessageType statusType = Stage3PipelineRunner.Status.StartsWith("PASS")
                ? MessageType.Info
                : Stage3PipelineRunner.Status.Contains("FAIL")
                    ? MessageType.Error : MessageType.None;
            EditorGUILayout.HelpBox(Stage3PipelineRunner.Status, statusType);
            foreach (string reason in Stage3PipelineRunner.FailureReasons)
                EditorGUILayout.HelpBox(reason, MessageType.Error);

            EditorGUILayout.LabelField("Pipeline log", EditorStyles.boldLabel);
            logScroll = EditorGUILayout.BeginScrollView(
                logScroll, GUILayout.MinHeight(180f));
            string logText = string.Join("\n", Stage3PipelineRunner.LogLines);
            var style = new GUIStyle(EditorStyles.textArea) { wordWrap = true };
            float width = Mathf.Max(100f, position.width - 38f);
            float height = Mathf.Max(
                180f, style.CalcHeight(new GUIContent(logText), width));
            EditorGUILayout.SelectableLabel(
                logText, style, GUILayout.Height(height));
            EditorGUILayout.EndScrollView();
        }

        private void DrawOutputField()
        {
            EditorGUILayout.BeginHorizontal();
            outputPath = EditorGUILayout.TextField("Artifact folder", outputPath);
            if (GUILayout.Button("Browse", GUILayout.Width(72f)))
            {
                string selected = EditorUtility.OpenFolderPanel(
                    "Select Stage 3 artifact folder", outputPath, "");
                if (!string.IsNullOrEmpty(selected))
                    outputPath = selected;
            }
            EditorGUILayout.EndHorizontal();
        }

        private void RunPipeline()
        {
            try
            {
                EditorPrefs.SetString(OutputPreference, outputPath);
                Stage3PipelineRunner.Start(calibration, outputPath);
            }
            catch (Exception exception)
            {
                Debug.LogException(exception);
                EditorUtility.DisplayDialog(
                    "Could not start Stage 3", exception.Message, "OK");
            }
        }
    }
}
