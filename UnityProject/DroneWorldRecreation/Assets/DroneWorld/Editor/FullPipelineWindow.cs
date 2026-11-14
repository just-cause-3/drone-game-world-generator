using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using DroneWorld.Stage1;
using DroneWorld.Stage1.Editor;
using DroneWorld.Stage2;
using DroneWorld.Stage2.Editor;
using DroneWorld.Stage3.Editor;
using DroneWorld.Stage4.Editor;
using DroneWorld.Stage4Geometry.Editor;
using UnityEditor;
using UnityEngine;
using Debug = UnityEngine.Debug;
using Process = System.Diagnostics.Process;

namespace DroneWorld.FullPipeline.Editor
{
    [Serializable]
    internal sealed class FullPipelineReportData
    {
        public string schema_version;
        public bool pass;
        public string status;
        public int failed_stage;
        public string[] failure_reasons;
        public FullPipelineStageSummaries stages;
    }

    [Serializable] internal sealed class FullPipelineStageSummaries
    {
        public FullPipelineStageSummary stage1, stage2, stage3, stage4;
    }
    [Serializable] internal sealed class FullPipelineStageSummary
    {
        public bool pass;
        public string report_sha256;
    }

    [InitializeOnLoad]
    internal static class FullPipelineRunner
    {
        private static readonly ConcurrentQueue<string> PendingLines = new();
        private static readonly List<string> OutputLines = new();
        private static readonly List<string> GateFailures = new();
        private static Process process;
        private static string runRoot;
        private static bool importing;
        private static bool reloadLocked;

        public static event Action Changed;
        // An exited child remains busy until Update consumes its result/imports it.
        public static bool IsRunning => process != null || importing;
        public static string Status { get; private set; } = "Ready";
        public static IReadOnlyList<string> LogLines => OutputLines;
        public static IReadOnlyList<string> FailureReasons => GateFailures;

        static FullPipelineRunner()
        {
            EditorApplication.update += Update;
        }

        public static void Start(string videoPath, string runDirectory, float worldWidth)
        {
            StartInternal(videoPath, runDirectory, worldWidth, false);
        }

        public static void StartFast(string videoPath, string runDirectory, float worldWidth)
        {
            StartInternal(videoPath, runDirectory, worldWidth, true);
        }

        public static void ImportPassingRun(string runDirectory)
        {
            RequireEditMode();
            if (IsRunning)
                throw new InvalidOperationException(
                    "Wait for the running pipeline process to finish before importing.");
            if (string.IsNullOrWhiteSpace(runDirectory))
                throw new ArgumentException("Select a passing run folder.", nameof(runDirectory));
            runRoot = Path.GetFullPath(runDirectory);
            OutputLines.Clear();
            GateFailures.Clear();
            Append($"Importing existing passing run: {runRoot}");
            importing = true;
            try { ImportValidatedBundles(); }
            catch
            {
                Status = "IMPORT FAILED - Python artifacts preserved; Unity import may be partial.";
                throw;
            }
            finally
            {
                importing = false;
                Append(Status);
                Changed?.Invoke();
            }
        }

        private static void StartInternal(
            string videoPath, string runDirectory, float worldWidth, bool fastMode)
        {
            RequireEditMode();
            if (IsRunning)
                throw new InvalidOperationException("The full pipeline is already running.");
            if (string.IsNullOrWhiteSpace(videoPath))
                throw new ArgumentException("Select an input video.", nameof(videoPath));
            if (string.IsNullOrWhiteSpace(runDirectory))
                throw new ArgumentException("Select a run folder.", nameof(runDirectory));
            if (!float.IsFinite(worldWidth) || worldWidth <= 0f)
                throw new ArgumentOutOfRangeException(
                    nameof(worldWidth), "Approximate world width must be positive.");

            string video = Path.GetFullPath(videoPath);
            if (!File.Exists(video))
                throw new FileNotFoundException("Input video does not exist.", video);
            string folderProblem = PipelineRunFolder.Problem(video, runDirectory, fastMode);
            if (folderProblem != null)
                throw new ArgumentException(folderProblem + " Use a new folder under the repository's runs directory.");
            string repository = RepositoryRoot;
            string python = Path.Combine(repository, ".venv", "Scripts", "python.exe");
            string scriptName = fastMode ? "fast_pipeline.py" : "full_pipeline.py";
            string script = Path.Combine(repository, scriptName);
            if (!File.Exists(python))
                throw new FileNotFoundException(
                    "Python environment is missing. Complete the initial setup first.", python);
            if (!File.Exists(script))
                throw new FileNotFoundException($"{scriptName} is missing.", script);
            RequireModel(
                Path.Combine(repository, "models", "visdrone-yolo26x.pt"),
                "Stage 1/3 aerial detector is missing. Run setup_drone_model.py.");
            RequireModel(
                Path.Combine(repository, "models", "inpainting_lama_2025jan.onnx"),
                "Stage 1 LaMa inpainting model is missing. Run " +
                "setup_inpainting_model.py.");
            RequireModel(
                Path.Combine(repository, "models", "openearthmap-mask2former",
                    "model.safetensors"),
                "Stage 4 aerial model is missing. Run setup_stage4_model.py.");
            RequireModel(
                Path.Combine(repository, "models", "yolo26x-sem.pt"),
                "Stage 4 road model is missing. Run setup_stage4_model.py.");

            runRoot = Path.GetFullPath(runDirectory);
            OutputLines.Clear();
            GateFailures.Clear();
            while (PendingLines.TryDequeue(out _)) { }
            Append($"Video: {video}");
            Append($"Isolated run: {runRoot}");
            Append($"Approximate image width: {worldWidth:F2} m");
            Append(fastMode
                ? "Mode: balanced fast (alternate-frame Stage 3 inference; full-rate playback)."
                : "Mode: maximum accuracy (Stage 3 inference on every frame).");
            Append("Unity will only import artifacts after all four Python stages pass.");

            string arguments =
                $"-u -B {Quote(script)} {Quote(video)} --run-root {Quote(runRoot)} " +
                $"--assumed-world-width {worldWidth.ToString(System.Globalization.CultureInfo.InvariantCulture)}";
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
            // Prevent script reloads from losing the process handle/run folder.
            EditorApplication.LockReloadAssemblies();
            reloadLocked = true;
            try
            {
                if (!process.Start())
                    throw new InvalidOperationException("Failed to start the full pipeline.");
                process.BeginOutputReadLine();
                process.BeginErrorReadLine();
            }
            catch
            {
                if (process != null)
                {
                    try { if (!process.HasExited) process.Kill(); }
                    catch (InvalidOperationException) { }
                    process.Dispose();
                    process = null;
                }
                ReleaseReloadLock();
                throw;
            }
            Status = "Running Stage 1 of 4...";
            Changed?.Invoke();
        }

        private static void RequireModel(string path, string message)
        {
            if (!File.Exists(path))
                throw new FileNotFoundException(message, path);
        }

        private static void CaptureLine(object sender, DataReceivedEventArgs eventArguments)
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
                StatusFromLine(line);
                appended = true;
            }
            if (appended)
                Changed?.Invoke();
            if (process == null || !process.HasExited)
                return;
            if (EditorApplication.isPlayingOrWillChangePlaymode)
            {
                Status = "Processing finished. Exit Play Mode to finish the pipeline and import results.";
                Changed?.Invoke();
                return;
            }

            importing = true;
            try
            {
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
                        ImportValidatedBundles();
                    }
                    catch (Exception exception)
                    {
                        Status = "IMPORT FAILED - Python artifacts preserved; Unity import may be partial.";
                        Append(exception.ToString());
                        Debug.LogException(exception);
                    }
                }
                else if (exitCode == 2)
                {
                    ReadGateFailure();
                    Status = "QUALITY GATE FAILED - no bundles were imported into Unity.";
                }
                else
                {
                    ReadGateFailure();
                    Status = $"Pipeline process failed with exit code {exitCode}; Unity was not changed.";
                }
                Append(Status);
                Changed?.Invoke();
            }
            finally
            {
                importing = false;
                ReleaseReloadLock();
            }
        }

        private static void RequireEditMode()
        {
            if (EditorApplication.isPlayingOrWillChangePlaymode)
                throw new InvalidOperationException("Exit Play Mode before running or importing the pipeline.");
        }

        private static void ReleaseReloadLock()
        {
            if (!reloadLocked) return;
            reloadLocked = false;
            EditorApplication.UnlockReloadAssemblies();
        }

        private static void ImportValidatedBundles()
        {
            FullPipelineValidation.Validate(runRoot);
            Status = "Processing passed. Importing all bundles into Unity...";
            Changed?.Invoke();
            Stage1ArtifactManifest manifest = Stage1BundleImporter.ImportBundle(
                Path.Combine(runRoot, "stage1"));
            Stage2CalibrationAsset calibration =
                Stage2CalibrationImporter.ImportCalibration(
                    Path.Combine(runRoot, "stage2"), manifest);
            string calibrationAssetPath = AssetDatabase.GetAssetPath(calibration);
            if (string.IsNullOrWhiteSpace(calibrationAssetPath))
                throw new InvalidDataException(
                    "Unity imported Stage 2 but could not resolve its asset path.");
            Stage3BundleImporter.ImportBundle(
                Path.Combine(runRoot, "stage3"), calibration);
            calibration = AssetDatabase.LoadAssetAtPath<Stage2CalibrationAsset>(
                calibrationAssetPath);
            if (calibration == null)
                throw new InvalidDataException(
                    "Unity could not reload Stage 2 after building the Stage 3 scene.");
            Stage4BundleImporter.ImportBundle(
                Path.Combine(runRoot, "stage4"), calibration);
            Status =
                "STAGES 1-4A COMPLETE - actor playback and semantic review imported. " +
                "NOT the final 3D world: review/acceptance and Stage 4B-C geometry are next.";
        }

        private static void StatusFromLine(string line)
        {
            if (line.Contains("Stage 2/4:")) Status = "Running Stage 2 of 4...";
            else if (line.Contains("Stage 3/4:")) Status = "Running Stage 3 of 4...";
            else if (line.Contains("Stage 4/4:")) Status = "Running Stage 4 of 4...";
        }

        private static FullPipelineReportData ReadReport()
        {
            string path = Path.Combine(runRoot, "pipeline_report.json");
            if (!File.Exists(path))
                throw new FileNotFoundException("pipeline_report.json is missing.", path);
            return JsonUtility.FromJson<FullPipelineReportData>(File.ReadAllText(path));
        }

        private static void ReadGateFailure()
        {
            GateFailures.Clear();
            try
            {
                FullPipelineReportData report = ReadReport();
                foreach (string reason in report?.failure_reasons ?? Array.Empty<string>())
                {
                    GateFailures.Add(reason);
                    Append($"Gate: {reason}");
                }
                if (report != null && report.failed_stage > 0)
                    Append($"Stopped at Stage {report.failed_stage}.");
            }
            catch (Exception exception)
            {
                Append($"Could not read pipeline report: {exception.Message}");
            }
        }

        private static void Append(string line)
        {
            OutputLines.Add(line);
            if (OutputLines.Count > 2500)
                OutputLines.RemoveRange(0, OutputLines.Count - 2500);
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

    public sealed class FullPipelineWindow : EditorWindow
    {
        private const string VideoPreference = "DroneWorld.FullPipeline.Video";
        private const string RunPreference = "DroneWorld.FullPipeline.RunRoot";
        private const string WidthPreference = "DroneWorld.FullPipeline.WorldWidth";
        private string videoPath;
        private string runPath;
        private float worldWidth;
        private Vector2 logScroll;
        private bool importScheduled;

        [MenuItem("Drone World/Run Full Pipeline (Stages 1-4A)...", priority = 1)]
        public static void ShowWindow()
        {
            GetWindow<FullPipelineWindow>("Full Drone World Pipeline");
        }

        private void OnEnable()
        {
            string repository = FullPipelineRunner.RepositoryRoot;
            videoPath = EditorPrefs.GetString(
                VideoPreference,
                Path.Combine(repository, "sample_vids", "remove_the_trees_and_people.mp4"));
            runPath = EditorPrefs.GetString(
                RunPreference,
                Path.Combine(repository, "runs", "remove_the_trees_and_people"));
            worldWidth = EditorPrefs.GetFloat(WidthPreference, 40f);
            FullPipelineRunner.Changed += OnRunnerChanged;
        }

        private void OnDisable()
        {
            FullPipelineRunner.Changed -= OnRunnerChanged;
            EditorApplication.delayCall -= ExecuteScheduledImport;
            importScheduled = false;
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
                "Video -> static ground -> scale -> actors -> semantic review",
                EditorStyles.boldLabel);
            EditorGUILayout.HelpBox(
                "One click runs Stages 1, 2, 3, and 4A in an isolated folder. " +
                "Stage 2 uses an approximate operator-chosen image width. Stage 4A " +
                "still requires human review. This command stops BEFORE Stage 4B-C " +
                "3D geometry; it never automatically approves semantic labels.",
                MessageType.Info);

            using (new EditorGUI.DisabledScope(FullPipelineRunner.IsRunning || EditorApplication.isPlayingOrWillChangePlaymode))
            {
                DrawVideoField();
                DrawRunField();
                worldWidth = EditorGUILayout.FloatField(
                    "Approx. image width (m)", worldWidth);
                EditorGUILayout.HelpBox("Output folder, NOT the folder containing videos. A new path is created automatically.", MessageType.None);
                if (GUILayout.Button("Use New Run Folder"))
                    runPath = PipelineRunFolder.Suggested(videoPath, false);
            }
            using (new EditorGUI.DisabledScope(FullPipelineRunner.IsRunning || EditorApplication.isPlayingOrWillChangePlaymode || importScheduled))
            {
                if (GUILayout.Button("Run Stages 1-4A", GUILayout.Height(36f)))
                    RunPipeline();
                if (GUILayout.Button("Import Existing Passing Run (No Processing)", GUILayout.Height(28f)))
                {
                    EditorPrefs.SetString(RunPreference, runPath);
                    importScheduled = true;
                    EditorApplication.delayCall += ExecuteScheduledImport;
                }
            }

            EditorGUILayout.Space(8f);
            EditorGUILayout.LabelField("Status", EditorStyles.boldLabel);
            MessageType type = FullPipelineRunner.Status.StartsWith("PASS")
                ? MessageType.Info
                : FullPipelineRunner.Status.Contains("FAIL")
                    ? MessageType.Error : MessageType.None;
            EditorGUILayout.HelpBox(FullPipelineRunner.Status, type);
            DrawRemainingSteps();
            foreach (string reason in FullPipelineRunner.FailureReasons)
                EditorGUILayout.HelpBox(reason, MessageType.Error);

            EditorGUILayout.LabelField("Pipeline log", EditorStyles.boldLabel);
            logScroll = EditorGUILayout.BeginScrollView(
                logScroll, GUILayout.MinHeight(220f));
            string logText = string.Join("\n", FullPipelineRunner.LogLines);
            var style = new GUIStyle(EditorStyles.textArea) { wordWrap = true };
            float width = Mathf.Max(100f, position.width - 38f);
            float height = Mathf.Max(220f, style.CalcHeight(new GUIContent(logText), width));
            EditorGUILayout.SelectableLabel(logText, style, GUILayout.Height(height));
            EditorGUILayout.EndScrollView();
        }

        private void DrawVideoField()
        {
            EditorGUILayout.BeginHorizontal();
            videoPath = EditorGUILayout.TextField("Input video", videoPath);
            if (GUILayout.Button("Browse", GUILayout.Width(72f)))
            {
                string selected = EditorUtility.OpenFilePanel(
                    "Select stationary top-down video",
                    string.IsNullOrWhiteSpace(videoPath)
                        ? FullPipelineRunner.RepositoryRoot
                        : Path.GetDirectoryName(videoPath),
                    "mp4,mov,avi,mkv");
                if (!string.IsNullOrEmpty(selected))
                {
                    videoPath = selected;
                    string name = Path.GetFileNameWithoutExtension(selected);
                    runPath = Path.Combine(FullPipelineRunner.RepositoryRoot, "runs", name);
                }
            }
            EditorGUILayout.EndHorizontal();
        }

        private void DrawRunField()
        {
            EditorGUILayout.BeginHorizontal();
            runPath = EditorGUILayout.TextField("Isolated run folder", runPath);
            if (GUILayout.Button("Browse", GUILayout.Width(72f)))
            {
                string selected = EditorUtility.OpenFolderPanel(
                    "Select an isolated full-pipeline run folder", runPath, "");
                if (!string.IsNullOrEmpty(selected))
                    runPath = selected;
            }
            EditorGUILayout.EndHorizontal();
        }

        private void RunPipeline()
        {
            try
            {
                string confirmed = PipelineRunFolder.Confirm(videoPath, runPath, false);
                if (confirmed == null) return;
                runPath = confirmed;
                EditorPrefs.SetString(VideoPreference, videoPath);
                EditorPrefs.SetString(RunPreference, runPath);
                EditorPrefs.SetFloat(WidthPreference, worldWidth);
                FullPipelineRunner.Start(videoPath, runPath, worldWidth);
            }
            catch (Exception exception)
            {
                Debug.LogException(exception);
                EditorUtility.DisplayDialog(
                    "Could not start full pipeline", exception.Message, "OK");
            }
        }

        private void ExecuteScheduledImport()
        {
            importScheduled = false;
            try { FullPipelineRunner.ImportPassingRun(runPath); }
            catch (Exception exception)
            {
                Debug.LogException(exception);
                EditorUtility.DisplayDialog("Could not import passing run", exception.Message, "OK");
            }
        }

        internal static void DrawRemainingSteps()
        {
            EditorGUILayout.HelpBox(
                "To finish: select the Stage 4A review root in the Hierarchy, inspect/correct " +
                "the map and explicitly accept it. Then generate the Stage 4B-C world. " +
                "A passing processing report is not proof of reconstruction accuracy.", MessageType.Info);
            using (new EditorGUI.DisabledScope(FullPipelineRunner.IsRunning || EditorApplication.isPlayingOrWillChangePlaymode))
                if (GUILayout.Button("Next: Open Stage 4B-C Geometry (Acceptance Required)"))
                    EditorApplication.delayCall += Stage4GeometryPipelineWindow.OpenWindow;
        }
    }
}
