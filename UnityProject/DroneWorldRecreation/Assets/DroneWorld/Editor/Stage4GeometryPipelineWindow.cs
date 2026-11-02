using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using DroneWorld.Stage4;
using DroneWorld.Stage4.Editor;
using DroneWorld.Stage4Geometry;
using UnityEditor;
using UnityEngine;
using Debug = UnityEngine.Debug;
using Process = System.Diagnostics.Process;

namespace DroneWorld.Stage4Geometry.Editor
{
    internal enum Stage4HeightPolicy
    {
        Fixed,
        FootprintPrior,
    }

    [InitializeOnLoad]
    internal static class Stage4GeometryPipelineRunner
    {
        private static readonly ConcurrentQueue<string> PendingLines = new();
        private static readonly List<string> OutputLines = new();
        private static Process process;
        private static string outputDirectory;
        private static Stage4SemanticAsset semantics;

        public static event Action Changed;
        public static bool IsRunning => process != null && !process.HasExited;
        public static string Status { get; private set; } = "Ready";
        public static IReadOnlyList<string> LogLines => OutputLines;
        public static string RepositoryRoot => Path.GetFullPath(Path.Combine(
            Application.dataPath, "..", "..", ".."));

        static Stage4GeometryPipelineRunner()
        {
            EditorApplication.update += Update;
        }

        public static void Start(
            Stage4SemanticAsset source, string output, Stage4HeightPolicy heightPolicy,
            float buildingHeight, float minimumBuildingArea,
            float simplifyTolerance, bool generateVegetation)
        {
            if (IsRunning)
                throw new InvalidOperationException(
                    "Stage 4B geometry generation is already running.");
            Stage4GeometryBundleImporter.ValidateAcceptedSemantics(source);
            if (string.IsNullOrWhiteSpace(output))
                throw new ArgumentException("Select an output folder.", nameof(output));
            if (!float.IsFinite(buildingHeight) || buildingHeight <= 0f ||
                !float.IsFinite(minimumBuildingArea) || minimumBuildingArea <= 0f ||
                !float.IsFinite(simplifyTolerance) || simplifyTolerance <= 0f)
                throw new ArgumentOutOfRangeException(
                    nameof(buildingHeight),
                    "Height, minimum area, and simplification tolerance must be positive.");

            string repository = RepositoryRoot;
            string python = Path.Combine(repository, ".venv", "Scripts", "python.exe");
            string script = Path.Combine(repository, "stage4b.py");
            if (!File.Exists(python))
                throw new FileNotFoundException(
                    "Python environment is missing. Complete initial setup first.", python);
            if (!File.Exists(script))
                throw new FileNotFoundException("stage4b.py is missing.", script);

            semantics = source;
            outputDirectory = Path.GetFullPath(output);
            OutputLines.Clear();
            while (PendingLines.TryDequeue(out _)) { }
            Append($"Accepted semantics: {source.SourceBundlePath}");
            Append($"Geometry output: {outputDirectory}");
            Append("Every generated height will be marked as assumed.");

            string policy = heightPolicy == Stage4HeightPolicy.Fixed
                ? "fixed" : "footprint_prior";
            string arguments =
                $"-u -B {Quote(script)} {Quote(source.SourceBundlePath)} " +
                $"--output {Quote(outputDirectory)} --height-policy {policy} " +
                $"--building-height {Invariant(buildingHeight)} " +
                $"--minimum-building-area {Invariant(minimumBuildingArea)} " +
                $"--simplify-tolerance {Invariant(simplifyTolerance)}";
            if (generateVegetation)
                arguments += " --generate-vegetation";
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
                    "Failed to start the Stage 4B geometry process.");
            process.BeginOutputReadLine();
            process.BeginErrorReadLine();
            Status = "Generating accepted geometry...";
            Changed?.Invoke();
        }

        public static void ImportExisting(
            Stage4SemanticAsset source, string output)
        {
            if (IsRunning)
                throw new InvalidOperationException(
                    "Wait for Stage 4B generation to finish first.");
            Stage4GeometryBundleImporter.ImportBundle(output, source);
            Status = "PASS - provisional 3D world scene created.";
            Changed?.Invoke();
        }

        private static void CaptureLine(object sender, DataReceivedEventArgs eventArgs)
        {
            if (!string.IsNullOrWhiteSpace(eventArgs.Data))
                PendingLines.Enqueue(eventArgs.Data);
        }

        private static void Update()
        {
            bool changed = false;
            while (PendingLines.TryDequeue(out string line))
            {
                Append(line);
                changed = true;
            }
            if (process == null || !process.HasExited)
            {
                if (changed)
                    Changed?.Invoke();
                return;
            }

            process.WaitForExit();
            while (PendingLines.TryDequeue(out string line))
                Append(line);
            int exitCode = process.ExitCode;
            process.Dispose();
            process = null;
            if (exitCode != 0)
            {
                Status = $"FAILED - geometry process exited with code {exitCode}.";
                Changed?.Invoke();
                return;
            }
            try
            {
                Stage4GeometryBundleImporter.ImportBundle(outputDirectory, semantics);
                Status = "PASS - Stage4GeneratedWorld.unity created. " +
                         "All heights remain provisional assumptions.";
            }
            catch (Exception exception)
            {
                Debug.LogException(exception);
                Status = $"FAILED IMPORT - {exception.Message}";
            }
            Changed?.Invoke();
        }

        private static void Append(string line)
        {
            OutputLines.Add(line);
            if (OutputLines.Count > 300)
                OutputLines.RemoveAt(0);
        }

        private static string Quote(string value)
        {
            return $"\"{value.Replace("\"", "\\\"")}\"";
        }

        private static string Invariant(float value)
        {
            return value.ToString("R", CultureInfo.InvariantCulture);
        }
    }

    public sealed class Stage4GeometryPipelineWindow : EditorWindow
    {
        private const string OutputPreference = "DroneWorld.Stage4Geometry.Output";
        private const string HeightPreference = "DroneWorld.Stage4Geometry.Height";
        private Stage4SemanticAsset semantics;
        private string outputPath;
        private Stage4HeightPolicy heightPolicy = Stage4HeightPolicy.Fixed;
        private float buildingHeight = 8f;
        private float minimumBuildingArea = 6f;
        private float simplifyTolerance = 0.25f;
        private bool generateVegetation;
        private Vector2 scroll;

        [MenuItem("Drone World/Stage 4/Generate Accepted 3D World (Stage 4B-C)...")]
        public static void OpenWindow()
        {
            GetWindow<Stage4GeometryPipelineWindow>("Stage 4B-C Geometry");
        }

        internal static void OpenForAsset(Stage4SemanticAsset asset)
        {
            Stage4GeometryPipelineWindow window =
                GetWindow<Stage4GeometryPipelineWindow>("Stage 4B-C Geometry");
            window.semantics = asset;
            window.SetDefaultOutput();
            window.Repaint();
        }

        private void OnEnable()
        {
            try
            {
                semantics = Stage4GeometryBundleImporter.FindCurrentSemantics();
            }
            catch
            {
                semantics = null;
            }
            outputPath = EditorPrefs.GetString(OutputPreference, string.Empty);
            buildingHeight = EditorPrefs.GetFloat(HeightPreference, 8f);
            if (string.IsNullOrWhiteSpace(outputPath))
                SetDefaultOutput();
            Stage4GeometryPipelineRunner.Changed += Repaint;
        }

        private void OnDisable()
        {
            Stage4GeometryPipelineRunner.Changed -= Repaint;
        }

        private void OnGUI()
        {
            EditorGUILayout.LabelField(
                "Accepted semantics -> provisional editable 3D world",
                EditorStyles.boldLabel);
            EditorGUILayout.HelpBox(
                "This phase extrudes accepted building roof regions and optionally " +
                "creates conservative vegetation placeholders. Stationary top-down " +
                "video cannot measure height, so every vertical dimension is explicitly " +
                "marked as assumed.", MessageType.Info);
            using (new EditorGUI.DisabledScope(Stage4GeometryPipelineRunner.IsRunning))
            {
                semantics = (Stage4SemanticAsset)EditorGUILayout.ObjectField(
                    "Accepted Stage 4A map", semantics,
                    typeof(Stage4SemanticAsset), false);
                DrawOutputField();
                heightPolicy = (Stage4HeightPolicy)EditorGUILayout.EnumPopup(
                    "Height policy", heightPolicy);
                buildingHeight = EditorGUILayout.FloatField(
                    "Fixed building height (m)", buildingHeight);
                minimumBuildingArea = EditorGUILayout.FloatField(
                    "Minimum building area (m²)", minimumBuildingArea);
                simplifyTolerance = EditorGUILayout.FloatField(
                    "Boundary tolerance (m)", simplifyTolerance);
                generateVegetation = EditorGUILayout.Toggle(
                    "Vegetation placeholders", generateVegetation);
            }

            if (semantics != null)
            {
                Stage4BundleImporter.RefreshAcceptanceState(semantics);
                EditorGUILayout.HelpBox(
                    semantics.AcceptedForGeometry
                        ? "Accepted map is hash-valid and ready for geometry."
                        : "BLOCKED: accept the exact semantic map in the Stage 4A " +
                          "review inspector first.",
                    semantics.AcceptedForGeometry
                        ? MessageType.Info : MessageType.Warning);
            }
            using (new EditorGUI.DisabledScope(
                       Stage4GeometryPipelineRunner.IsRunning || semantics == null ||
                       !semantics.AcceptedForGeometry ||
                       string.IsNullOrWhiteSpace(outputPath)))
            {
                if (GUILayout.Button(
                        "Generate and Import Provisional 3D World",
                        GUILayout.Height(34f)))
                    RunGeometry();
                if (GUILayout.Button("Import Existing Passing Geometry Bundle"))
                    ImportExisting();
            }

            EditorGUILayout.Space(8f);
            MessageType statusType =
                Stage4GeometryPipelineRunner.Status.StartsWith("PASS")
                    ? MessageType.Info
                    : Stage4GeometryPipelineRunner.Status.StartsWith("FAIL")
                        ? MessageType.Error : MessageType.None;
            EditorGUILayout.HelpBox(Stage4GeometryPipelineRunner.Status, statusType);
            EditorGUILayout.LabelField("Pipeline log", EditorStyles.boldLabel);
            scroll = EditorGUILayout.BeginScrollView(scroll, GUILayout.MinHeight(140f));
            EditorGUILayout.SelectableLabel(
                string.Join("\n", Stage4GeometryPipelineRunner.LogLines),
                EditorStyles.textArea, GUILayout.ExpandHeight(true));
            EditorGUILayout.EndScrollView();
        }

        private void DrawOutputField()
        {
            EditorGUILayout.BeginHorizontal();
            outputPath = EditorGUILayout.TextField("Geometry output", outputPath);
            if (GUILayout.Button("Browse", GUILayout.Width(72f)))
            {
                string selected = EditorUtility.OpenFolderPanel(
                    "Select Stage 4B geometry output", outputPath, "");
                if (!string.IsNullOrEmpty(selected))
                    outputPath = selected;
            }
            EditorGUILayout.EndHorizontal();
        }

        private void SetDefaultOutput()
        {
            if (semantics != null && !string.IsNullOrWhiteSpace(semantics.SourceBundlePath))
                outputPath = Path.Combine(semantics.SourceBundlePath, "geometry");
        }

        private void RunGeometry()
        {
            try
            {
                SavePreferences();
                Stage4GeometryPipelineRunner.Start(
                    semantics, outputPath, heightPolicy, buildingHeight,
                    minimumBuildingArea, simplifyTolerance, generateVegetation);
            }
            catch (Exception exception)
            {
                Debug.LogException(exception);
                EditorUtility.DisplayDialog(
                    "Could not start Stage 4B", exception.Message, "OK");
            }
        }

        private void ImportExisting()
        {
            try
            {
                SavePreferences();
                EditorApplication.delayCall += () =>
                {
                    try
                    {
                        Stage4GeometryPipelineRunner.ImportExisting(
                            semantics, outputPath);
                    }
                    catch (Exception exception)
                    {
                        Debug.LogException(exception);
                        EditorUtility.DisplayDialog(
                            "Could not import Stage 4B", exception.Message, "OK");
                    }
                };
            }
            catch (Exception exception)
            {
                Debug.LogException(exception);
            }
        }

        private void SavePreferences()
        {
            EditorPrefs.SetString(OutputPreference, outputPath);
            EditorPrefs.SetFloat(HeightPreference, buildingHeight);
        }
    }
}
