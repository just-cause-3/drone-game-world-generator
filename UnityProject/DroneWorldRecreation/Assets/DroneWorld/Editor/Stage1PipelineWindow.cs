using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using UnityEditor;
using UnityEngine;
using Debug = UnityEngine.Debug;
using Process = System.Diagnostics.Process;

namespace DroneWorld.Stage1.Editor
{
    [InitializeOnLoad]
    internal static class Stage1PipelineRunner
    {
        private static readonly ConcurrentQueue<string> PendingLines = new();
        private static readonly List<string> OutputLines = new();
        private static readonly List<string> GateFailures = new();
        private static Process process;
        private static string outputDirectory;

        public static event Action Changed;
        public static bool IsRunning => process != null && !process.HasExited;
        public static string Status { get; private set; } = "Ready";
        public static IReadOnlyList<string> LogLines => OutputLines;
        public static IReadOnlyList<string> FailureReasons => GateFailures;

        static Stage1PipelineRunner()
        {
            EditorApplication.update += Update;
        }

        public static void Start(string videoPath, string outputPath)
        {
            if (IsRunning)
                throw new InvalidOperationException("Stage 1 is already running.");
            if (string.IsNullOrWhiteSpace(videoPath))
                throw new ArgumentException("Select an input video before running Stage 1.",
                                            nameof(videoPath));
            if (string.IsNullOrWhiteSpace(outputPath))
                throw new ArgumentException("Select an artifact output folder before running Stage 1.",
                                            nameof(outputPath));
            string video = Path.GetFullPath(videoPath);
            if (!File.Exists(video))
                throw new FileNotFoundException("Input video does not exist.", video);
            string repository = RepositoryRoot;
            string python = Path.Combine(repository, ".venv", "Scripts", "python.exe");
            string stageScript = Path.Combine(repository, "stage1.py");
            if (!File.Exists(python))
                throw new FileNotFoundException(
                    "Python environment is missing. Create .venv and install requirements first.", python);
            if (!File.Exists(stageScript))
                throw new FileNotFoundException("stage1.py is missing.", stageScript);

            outputDirectory = Path.GetFullPath(outputPath);
            OutputLines.Clear();
            GateFailures.Clear();
            while (PendingLines.TryDequeue(out _)) { }
            Append($"Video: {video}");
            Append($"Artifacts: {outputDirectory}");
            Append("Unity will be modified only if the Stage 1 quality gate passes.");

            var startInfo = new ProcessStartInfo
            {
                FileName = python,
                Arguments = $"-u -B {Quote(stageScript)} {Quote(video)} --output {Quote(outputDirectory)}",
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
                throw new InvalidOperationException("Failed to start the Stage 1 Python process.");
            process.BeginOutputReadLine();
            process.BeginErrorReadLine();
            Status = "Processing video and validating static ground...";
            Changed?.Invoke();
        }

        private static void CaptureLine(object sender, DataReceivedEventArgs eventArgs)
        {
            if (!string.IsNullOrWhiteSpace(eventArgs.Data))
                PendingLines.Enqueue(eventArgs.Data);
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
                    Status = "Ground passed. Importing artifacts and building Unity scene...";
                    Changed?.Invoke();
                    Stage1BundleImporter.ImportBundle(outputDirectory);
                    Status = "PASS — Stage1Preview.unity was created and opened.";
                    Append(Status);
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
                Status = "FAILED QUALITY GATE — Unity scene was not changed.";
                AppendFailureReasons();
            }
            else
            {
                Status = $"Stage 1 process failed with exit code {exitCode}; Unity was not changed.";
            }
            Append(Status);
            Changed?.Invoke();
        }

        private static void AppendFailureReasons()
        {
            try
            {
                string reportPath = Path.Combine(outputDirectory, "report.json");
                Stage1ReportData report = JsonUtility.FromJson<Stage1ReportData>(
                    File.ReadAllText(reportPath));
                if (report?.failure_reasons != null)
                {
                    foreach (string reason in report.failure_reasons)
                    {
                        GateFailures.Add(reason);
                        Append($"Gate: {reason}");
                    }
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
            if (OutputLines.Count > 1000)
                OutputLines.RemoveRange(0, OutputLines.Count - 1000);
        }

        private static string Quote(string argument)
        {
            return $"\"{argument.Replace("\"", "\\\"")}\"";
        }

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

    public sealed class Stage1PipelineWindow : EditorWindow
    {
        private const string VideoPreference = "DroneWorld.Stage1.Video";
        private const string OutputPreference = "DroneWorld.Stage1.Output";
        private string videoPath;
        private string outputPath;
        private Vector2 logScroll;

        [MenuItem("Drone World/Run Stage 1 Pipeline...")]
        public static void ShowWindow()
        {
            GetWindow<Stage1PipelineWindow>("Stage 1 Pipeline");
        }

        private void OnEnable()
        {
            videoPath = EditorPrefs.GetString(VideoPreference, string.Empty);
            outputPath = EditorPrefs.GetString(
                OutputPreference, Path.Combine(Stage1PipelineRunner.RepositoryRoot, "stage1_out"));
            Stage1PipelineRunner.Changed += OnRunnerChanged;
        }

        private void OnDisable()
        {
            Stage1PipelineRunner.Changed -= OnRunnerChanged;
        }

        private void OnRunnerChanged()
        {
            logScroll.y = float.MaxValue;
            Repaint();
        }

        private void OnGUI()
        {
            EditorGUILayout.Space(8f);
            EditorGUILayout.LabelField("Video → validated static ground → Unity preview",
                                       EditorStyles.boldLabel);
            EditorGUILayout.HelpBox(
                "Stage 1 accepts one continuous stationary, top-down shot with bounded zoom. " +
                "A failed quality report never modifies the Unity preview scene.",
                MessageType.Info);

            using (new EditorGUI.DisabledScope(Stage1PipelineRunner.IsRunning))
            {
                DrawPathField("Input video", ref videoPath, true);
                DrawPathField("Artifact folder", ref outputPath, false);
            }
            if (string.IsNullOrWhiteSpace(videoPath))
            {
                EditorGUILayout.HelpBox(
                    "Choose the stationary top-down input video with the Browse button.",
                    MessageType.Warning);
            }
            EditorGUILayout.Space(6f);
            using (new EditorGUI.DisabledScope(
                       Stage1PipelineRunner.IsRunning || string.IsNullOrWhiteSpace(videoPath)))
            {
                if (GUILayout.Button("Run Stage 1", GUILayout.Height(32f)))
                    RunPipeline();
            }

            EditorGUILayout.Space(8f);
            EditorGUILayout.LabelField("Status", EditorStyles.boldLabel);
            MessageType statusType = Stage1PipelineRunner.Status.StartsWith("PASS")
                ? MessageType.Info
                : Stage1PipelineRunner.Status.Contains("FAIL")
                    ? MessageType.Error : MessageType.None;
            EditorGUILayout.HelpBox(Stage1PipelineRunner.Status, statusType);
            foreach (string reason in Stage1PipelineRunner.FailureReasons)
                EditorGUILayout.HelpBox(reason, MessageType.Error);

            EditorGUILayout.LabelField("Pipeline log", EditorStyles.boldLabel);
            logScroll = EditorGUILayout.BeginScrollView(logScroll, GUILayout.MinHeight(180f));
            string logText = string.Join("\n", Stage1PipelineRunner.LogLines);
            var logStyle = new GUIStyle(EditorStyles.textArea) { wordWrap = true };
            float logWidth = Mathf.Max(100f, position.width - 38f);
            float logHeight = Mathf.Max(
                180f, logStyle.CalcHeight(new GUIContent(logText), logWidth));
            EditorGUILayout.SelectableLabel(
                logText, logStyle, GUILayout.Height(logHeight));
            EditorGUILayout.EndScrollView();
        }

        private void DrawPathField(string label, ref string path, bool video)
        {
            EditorGUILayout.BeginHorizontal();
            path = EditorGUILayout.TextField(label, path);
            if (GUILayout.Button("Browse", GUILayout.Width(72f)))
            {
                string selected = video
                    ? EditorUtility.OpenFilePanelWithFilters(
                        "Select stationary top-down video", path,
                        new[] { "Video files", "mp4,mov,avi,mkv,m4v", "All files", "*" })
                    : EditorUtility.OpenFolderPanel("Select Stage 1 artifact folder", path, "");
                if (!string.IsNullOrEmpty(selected))
                    path = selected;
            }
            EditorGUILayout.EndHorizontal();
        }

        private void RunPipeline()
        {
            try
            {
                EditorPrefs.SetString(VideoPreference, videoPath);
                EditorPrefs.SetString(OutputPreference, outputPath);
                Stage1PipelineRunner.Start(videoPath, outputPath);
            }
            catch (Exception exception)
            {
                Debug.LogException(exception);
                EditorUtility.DisplayDialog("Could not start Stage 1", exception.Message, "OK");
            }
        }
    }
}
