using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using DroneWorld.Stage1;
using DroneWorld.Stage1.Editor;
using UnityEditor;
using UnityEngine;
using Debug = UnityEngine.Debug;
using Process = System.Diagnostics.Process;

namespace DroneWorld.Stage2.Editor
{
    public sealed class Stage2CalibrationWindow : EditorWindow
    {
        [Serializable]
        private sealed class DraftPoint
        {
            public string Id;
            public float U;
            public float V;
            public float X;
            public float Z;
        }

        [SerializeField] private Stage1ArtifactManifest manifest;
        [SerializeField] private List<DraftPoint> points = new();
        [SerializeField] private string originDescription = "chosen surveyed ground point";
        [SerializeField] private string xAxisDescription = "east/right";
        [SerializeField] private string zAxisDescription = "north/forward";
        [SerializeField] private float groundY;
        [SerializeField] private bool approximateAutomaticScale = true;
        [SerializeField] private float approximateWorldWidth = 40f;
        [SerializeField] private string outputDirectory;
        [SerializeField] private Vector2 scroll;
        private Texture2D groundTexture;
        private string status = "Click fixed ground landmarks, then enter their measured X/Z.";
        private bool calibrationScheduled;

        [MenuItem("Drone World/Run Stage 2 Metric Calibration...")]
        public static void ShowWindow()
        {
            GetWindow<Stage2CalibrationWindow>("Stage 2 Calibration");
        }

        private void OnEnable()
        {
            minSize = new Vector2(620f, 640f);
            if (manifest == null)
            {
                try
                {
                    manifest = Stage2CalibrationImporter.FindCurrentStage1Manifest();
                }
                catch
                {
                    // The window explains the missing input instead of failing to open.
                }
            }
            if (string.IsNullOrWhiteSpace(outputDirectory))
            {
                outputDirectory = Path.Combine(
                    Stage1PipelineRunner.RepositoryRoot, "stage2_out");
            }
            LoadGroundTexture();
        }

        private void OnGUI()
        {
            EditorGUILayout.LabelField(
                "Stage 1 reference pixels → measured Unity X/Z metres",
                EditorStyles.boldLabel);
            EditorGUILayout.HelpBox(
                "Use fixed landmarks on the flat ground. Click a point in the image, " +
                "then enter that exact landmark's measured X and Z. Four points are " +
                "the minimum; six or more spread around the image are recommended.",
                MessageType.Info);

            EditorGUI.BeginChangeCheck();
            manifest = (Stage1ArtifactManifest)EditorGUILayout.ObjectField(
                "Stage 1 manifest", manifest,
                typeof(Stage1ArtifactManifest), false);
            if (EditorGUI.EndChangeCheck())
                LoadGroundTexture();

            if (manifest == null || groundTexture == null)
            {
                EditorGUILayout.HelpBox(
                    "Import a passing Stage 1 bundle, then select its generated " +
                    "Stage1Manifest asset.", MessageType.Warning);
                if (GUILayout.Button("Find Imported Stage 1 Manifest"))
                {
                    try
                    {
                        manifest =
                            Stage2CalibrationImporter.FindCurrentStage1Manifest();
                        LoadGroundTexture();
                    }
                    catch (Exception exception)
                    {
                        status = exception.Message;
                    }
                }
                EditorGUILayout.HelpBox(status, MessageType.None);
                return;
            }

            scroll = EditorGUILayout.BeginScrollView(scroll);
            approximateAutomaticScale = EditorGUILayout.ToggleLeft(
                "Use approximate automatic scale (no measurements required)",
                approximateAutomaticScale,
                EditorStyles.boldLabel);
            if (approximateAutomaticScale)
            {
                approximateWorldWidth = EditorGUILayout.FloatField(
                    "Assumed world width", approximateWorldWidth);
                float approximateDepth = approximateWorldWidth *
                                         manifest.PixelHeight /
                                         manifest.PixelWidth;
                EditorGUILayout.HelpBox(
                    $"The complete image will become approximately " +
                    $"{approximateWorldWidth:F1} × {approximateDepth:F1} Unity " +
                    "units. No clicked landmarks are used. Distances and vehicle " +
                    "speeds will be approximate.",
                    MessageType.Warning);
            }
            DrawReferenceImage();
            EditorGUILayout.Space(8f);
            if (!approximateAutomaticScale)
                DrawPointEditor();
            EditorGUILayout.Space(8f);
            DrawCoordinateSettings();
            EditorGUILayout.Space(8f);
            DrawRunControls();
            EditorGUILayout.EndScrollView();
        }

        private void DrawReferenceImage()
        {
            EditorGUILayout.LabelField(
                $"Reference ground: {manifest.PixelWidth} × {manifest.PixelHeight}",
                EditorStyles.boldLabel);
            float aspect = manifest.PixelWidth / (float)manifest.PixelHeight;
            Rect imageRect = GUILayoutUtility.GetAspectRect(
                aspect, GUILayout.ExpandWidth(true));
            GUI.DrawTexture(
                imageRect, groundTexture, ScaleMode.StretchToFill, false);

            Event current = Event.current;
            if (!approximateAutomaticScale &&
                current.type == EventType.MouseDown &&
                current.button == 0 &&
                imageRect.Contains(current.mousePosition))
            {
                float normalizedU =
                    Mathf.Clamp01((current.mousePosition.x - imageRect.x) /
                                  imageRect.width);
                float normalizedV =
                    Mathf.Clamp01((current.mousePosition.y - imageRect.y) /
                                  imageRect.height);
                points.Add(new DraftPoint
                {
                    Id = $"gcp_{points.Count + 1:00}",
                    U = normalizedU * (manifest.PixelWidth - 1),
                    V = normalizedV * (manifest.PixelHeight - 1),
                });
                status = $"Added {points[^1].Id}. Enter its measured X/Z below.";
                current.Use();
                Repaint();
            }

            if (!approximateAutomaticScale)
            {
                for (int index = 0; index < points.Count; index++)
                {
                    DraftPoint point = points[index];
                    float x = imageRect.x +
                              point.U / Mathf.Max(1f, manifest.PixelWidth - 1f) *
                              imageRect.width;
                    float y = imageRect.y +
                              point.V / Mathf.Max(1f, manifest.PixelHeight - 1f) *
                              imageRect.height;
                    Rect marker = new Rect(x - 5f, y - 5f, 10f, 10f);
                    EditorGUI.DrawRect(marker, new Color(1f, 0.82f, 0.05f, 1f));
                    GUI.Label(new Rect(x + 7f, y - 10f, 75f, 20f),
                              point.Id, EditorStyles.whiteMiniLabel);
                }
            }
        }

        private void DrawPointEditor()
        {
            EditorGUILayout.BeginHorizontal();
            EditorGUILayout.LabelField(
                $"Ground control points ({points.Count})",
                EditorStyles.boldLabel);
            GUILayout.FlexibleSpace();
            if (GUILayout.Button("Clear All", GUILayout.Width(80f)) &&
                EditorUtility.DisplayDialog(
                    "Clear calibration points?",
                    "This removes every unsaved point from the window.", "Clear", "Cancel"))
            {
                points.Clear();
            }
            EditorGUILayout.EndHorizontal();

            EditorGUILayout.HelpBox(
                "Pixel U/V comes from your click. World X/Z is not automatic: " +
                "enter the measured metre coordinate for every point. Only one " +
                "point may use the origin (0, 0).",
                MessageType.Info);

            if (points.Count == 0)
                EditorGUILayout.HelpBox("Click the reference image to add a point.",
                                        MessageType.None);
            for (int index = 0; index < points.Count; index++)
            {
                DraftPoint point = points[index];
                EditorGUILayout.BeginVertical(EditorStyles.helpBox);
                EditorGUILayout.BeginHorizontal();
                point.Id = EditorGUILayout.TextField("ID", point.Id);
                if (GUILayout.Button("Remove", GUILayout.Width(70f)))
                {
                    points.RemoveAt(index);
                    EditorGUILayout.EndHorizontal();
                    EditorGUILayout.EndVertical();
                    break;
                }
                EditorGUILayout.EndHorizontal();
                EditorGUILayout.BeginHorizontal();
                EditorGUI.BeginDisabledGroup(true);
                point.U = EditorGUILayout.FloatField("Pixel U", point.U);
                point.V = EditorGUILayout.FloatField("Pixel V", point.V);
                EditorGUI.EndDisabledGroup();
                EditorGUILayout.EndHorizontal();
                EditorGUILayout.BeginHorizontal();
                point.X = EditorGUILayout.FloatField("World X (m)", point.X);
                point.Z = EditorGUILayout.FloatField("World Z (m)", point.Z);
                EditorGUILayout.EndHorizontal();
                EditorGUILayout.EndVertical();
            }
        }

        private void DrawCoordinateSettings()
        {
            EditorGUILayout.LabelField("Coordinate convention", EditorStyles.boldLabel);
            originDescription = EditorGUILayout.TextField(
                "Origin", originDescription);
            xAxisDescription = EditorGUILayout.TextField(
                "+X direction", xAxisDescription);
            zAxisDescription = EditorGUILayout.TextField(
                "+Z direction", zAxisDescription);
            groundY = EditorGUILayout.FloatField("Ground Y (m)", groundY);
        }

        private void DrawRunControls()
        {
            EditorGUILayout.LabelField("Stage 2 output", EditorStyles.boldLabel);
            EditorGUILayout.BeginHorizontal();
            outputDirectory = EditorGUILayout.TextField(outputDirectory);
            if (GUILayout.Button("Browse", GUILayout.Width(80f)))
            {
                string selected = EditorUtility.OpenFolderPanel(
                    "Choose Stage 2 output folder",
                    Path.GetDirectoryName(outputDirectory), "");
                if (!string.IsNullOrEmpty(selected))
                    outputDirectory = selected;
            }
            EditorGUILayout.EndHorizontal();

            bool missingInput = approximateAutomaticScale
                ? approximateWorldWidth <= 0f
                : points.Count < 4;
            EditorGUI.BeginDisabledGroup(missingInput || calibrationScheduled);
            string buttonLabel = approximateAutomaticScale
                ? "Build Approximate-Scale Scene"
                : "Validate Calibration and Build Metric Scene";
            if (GUILayout.Button(buttonLabel,
                                 GUILayout.Height(36f)))
            {
                calibrationScheduled = true;
                status = "Stage 2 calibration scheduled...";
                // Scene creation and modal dialogs invalidate GUILayout state when
                // executed inside this scroll view's current OnGUI event.
                EditorApplication.delayCall += ExecuteScheduledCalibration;
            }
            EditorGUI.EndDisabledGroup();
            if (!approximateAutomaticScale && points.Count < 4)
                EditorGUILayout.HelpBox(
                    $"Add {4 - points.Count} more point(s) to enable calibration.",
                    MessageType.Warning);
            EditorGUILayout.HelpBox(status, MessageType.None);
        }

        private void ExecuteScheduledCalibration()
        {
            calibrationScheduled = false;
            if (this == null)
                return;
            try
            {
                RunCalibration();
            }
            catch (Exception exception)
            {
                status = exception.Message;
                Debug.LogException(exception);
                EditorUtility.DisplayDialog(
                    "Stage 2 failed", exception.Message, "OK");
            }
            finally
            {
                Repaint();
            }
        }

        private void RunCalibration()
        {
            if (approximateAutomaticScale)
            {
                if (!float.IsFinite(approximateWorldWidth) ||
                    approximateWorldWidth <= 0f)
                    throw new InvalidDataException(
                        "Approximate world width must be positive.");
            }
            else
            {
                ValidateDraftPoints();
            }
            string repository = Stage1PipelineRunner.RepositoryRoot;
            string python = Path.Combine(repository, ".venv", "Scripts", "python.exe");
            string stageScript = Path.Combine(repository, "stage2.py");
            if (!File.Exists(python))
                throw new FileNotFoundException(
                    "Repository Python environment is missing.", python);
            if (!File.Exists(stageScript))
                throw new FileNotFoundException("stage2.py is missing.", stageScript);
            if (string.IsNullOrWhiteSpace(manifest.SourceBundlePath) ||
                !Directory.Exists(manifest.SourceBundlePath))
                throw new DirectoryNotFoundException(
                    "The source Stage 1 bundle is no longer available.");

            string pointPath = Path.Combine(
                manifest.SourceBundlePath, "stage2_control_points.json");
            SaveControlPoints(pointPath);
            string output = Path.GetFullPath(outputDirectory);
            status = "Running Stage 2 calibration quality gate...";
            Repaint();

            var startInfo = new ProcessStartInfo
            {
                FileName = python,
                Arguments =
                    $"-B {Quote(stageScript)} {Quote(manifest.SourceBundlePath)} " +
                    $"{Quote(pointPath)} --output {Quote(output)}",
                WorkingDirectory = repository,
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
            };
            using var process = Process.Start(startInfo);
            if (process == null)
                throw new InvalidOperationException(
                    "Could not start the Stage 2 Python process.");
            string standardOutput = process.StandardOutput.ReadToEnd();
            string standardError = process.StandardError.ReadToEnd();
            process.WaitForExit();
            string processLog = (standardOutput + Environment.NewLine +
                                 standardError).Trim();
            if (process.ExitCode != 0)
            {
                throw new InvalidDataException(
                    $"Calibration process exited with code {process.ExitCode}." +
                    (string.IsNullOrWhiteSpace(processLog)
                        ? string.Empty
                        : Environment.NewLine + processLog));
            }

            Stage2CalibrationAsset asset =
                Stage2CalibrationImporter.ImportCalibration(output, manifest);
            status =
                $"PASS — metric scene created. RMSE " +
                $"{asset.ReprojectionRmseMetres:F4} m; " +
                $"confidence {asset.Confidence}.";
            EditorUtility.DisplayDialog(
                "Stage 2 passed",
                status + Environment.NewLine +
                "Inspect calibration_overlay.png before accepting the result.",
                "OK");
        }

        private void ValidateDraftPoints()
        {
            if (points.Count < 4)
                throw new InvalidDataException(
                    "At least four control points are required.");
            var identifiers = new HashSet<string>(StringComparer.Ordinal);
            foreach (DraftPoint point in points)
            {
                point.Id = point.Id?.Trim();
                if (string.IsNullOrEmpty(point.Id))
                    throw new InvalidDataException(
                        "Every control point needs a unique ID.");
                if (!identifiers.Add(point.Id))
                    throw new InvalidDataException(
                        $"Duplicate control-point ID: {point.Id}");
                if (!float.IsFinite(point.X) || !float.IsFinite(point.Z))
                    throw new InvalidDataException(
                        $"Control point {point.Id} has an invalid world coordinate.");
            }

            const float duplicateToleranceMetres = 0.001f;
            for (int first = 0; first < points.Count; first++)
            {
                for (int second = first + 1; second < points.Count; second++)
                {
                    DraftPoint a = points[first];
                    DraftPoint b = points[second];
                    if (Mathf.Abs(a.X - b.X) <= duplicateToleranceMetres &&
                        Mathf.Abs(a.Z - b.Z) <= duplicateToleranceMetres)
                    {
                        throw new InvalidDataException(
                            $"{a.Id} and {b.Id} both use world coordinate " +
                            $"({a.X:F3}, {a.Z:F3}) m. Enter the separately " +
                            "measured X/Z coordinate for every marker; only one " +
                            "point may be the origin (0, 0).");
                    }
                    if (Mathf.Abs(a.U - b.U) <= 0.01f &&
                        Mathf.Abs(a.V - b.V) <= 0.01f)
                    {
                        throw new InvalidDataException(
                            $"{a.Id} and {b.Id} identify the same image pixel. " +
                            "Move or remove one marker.");
                    }
                }
            }
        }

        private void SaveControlPoints(string path)
        {
            if (approximateAutomaticScale)
            {
                SaveApproximateControlPoints(path);
                return;
            }
            var data = new Stage2ControlPointFileData
            {
                calibration_mode = "measured_control_points",
                origin_description = originDescription,
                x_axis_description = xAxisDescription,
                z_axis_description = zAxisDescription,
                ground_y_m = groundY,
                points = new Stage2ControlPointData[points.Count],
            };
            for (int index = 0; index < points.Count; index++)
            {
                DraftPoint point = points[index];
                data.points[index] = new Stage2ControlPointData
                {
                    id = point.Id,
                    image = new Stage2ImagePointData { u = point.U, v = point.V },
                    world = new Stage2WorldPointData { x = point.X, z = point.Z },
                };
            }
            File.WriteAllText(path, JsonUtility.ToJson(data, true));
        }

        private void SaveApproximateControlPoints(string path)
        {
            float halfWidth = approximateWorldWidth * 0.5f;
            float depth = approximateWorldWidth * manifest.PixelHeight /
                          manifest.PixelWidth;
            float halfDepth = depth * 0.5f;
            float maxU = manifest.PixelWidth - 1f;
            float maxV = manifest.PixelHeight - 1f;
            var data = new Stage2ControlPointFileData
            {
                calibration_mode = "approximate_image_extent",
                assumed_world_width_m = approximateWorldWidth,
                origin_description = "approximate image centre",
                x_axis_description = "image right",
                z_axis_description = "image up",
                ground_y_m = groundY,
                points = new[]
                {
                    MakePoint("auto_tl", 0f, 0f, -halfWidth, halfDepth),
                    MakePoint("auto_tr", maxU, 0f, halfWidth, halfDepth),
                    MakePoint("auto_br", maxU, maxV, halfWidth, -halfDepth),
                    MakePoint("auto_bl", 0f, maxV, -halfWidth, -halfDepth),
                },
            };
            File.WriteAllText(path, JsonUtility.ToJson(data, true));
        }

        private static Stage2ControlPointData MakePoint(
            string identifier, float u, float v, float x, float z)
        {
            return new Stage2ControlPointData
            {
                id = identifier,
                image = new Stage2ImagePointData { u = u, v = v },
                world = new Stage2WorldPointData { x = x, z = z },
            };
        }

        private void LoadGroundTexture()
        {
            groundTexture = manifest == null
                ? null
                : AssetDatabase.LoadAssetAtPath<Texture2D>(
                    manifest.GroundTextureAssetPath);
            Repaint();
        }

        private static string Quote(string argument)
        {
            return $"\"{argument.Replace("\"", "\\\"")}\"";
        }
    }
}
