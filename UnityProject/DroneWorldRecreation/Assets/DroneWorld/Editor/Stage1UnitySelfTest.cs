using System;
using System.IO;
using DroneWorld.Stage1;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace DroneWorld.Stage1.Editor
{
    public static class Stage1UnitySelfTest
    {
        private const int FixtureWidth = 13;
        private const int FixtureHeight = 7;

        /// <summary>
        /// Batch-mode integration test for validation, external import, texture setup,
        /// manifest creation, coordinate orientation, and scene generation.
        /// </summary>
        public static void Run()
        {
            string temporaryBundle = Path.Combine(
                Path.GetTempPath(), $"drone-world-stage1-{Guid.NewGuid():N}");
            string generatedAssetFolder = null;
            Directory.CreateDirectory(temporaryBundle);
            try
            {
                CreateFixtureBundle(temporaryBundle);
                Stage1ArtifactManifest manifest = Stage1BundleImporter.ImportBundle(temporaryBundle);
                generatedAssetFolder = Path.GetDirectoryName(
                    manifest.GroundTextureAssetPath)!.Replace('\\', '/');

                Require(manifest.SchemaVersion == "stage1-ground-v1", "Manifest schema mismatch.");
                Require(manifest.PixelWidth == FixtureWidth &&
                        manifest.PixelHeight == FixtureHeight,
                        "Manifest dimensions mismatch.");
                Require(!manifest.HasMetricCalibration,
                        "Stage 1 preview must not claim metric calibration.");

                Texture2D importedGround = AssetDatabase.LoadAssetAtPath<Texture2D>(
                    manifest.GroundTextureAssetPath);
                Require(importedGround != null, "Imported ground texture could not be loaded.");
                Require(importedGround.width == FixtureWidth &&
                        importedGround.height == FixtureHeight,
                        $"Imported ground was unexpectedly resized to " +
                        $"{importedGround.width}x{importedGround.height}.");
                TextureImporter groundImporter = AssetImporter.GetAtPath(
                    manifest.GroundTextureAssetPath) as TextureImporter;
                Require(groundImporter != null,
                        "Imported ground TextureImporter could not be loaded.");
                Require(groundImporter.npotScale == TextureImporterNPOTScale.None,
                        "Imported ground does not preserve non-power-of-two dimensions.");

                Stage1GroundView view = UnityEngine.Object.FindAnyObjectByType<Stage1GroundView>();
                Require(view != null, "Generated preview has no Stage1GroundView.");
                Mesh mesh = view.GetComponent<MeshFilter>().sharedMesh;
                Require(mesh != null && mesh.vertexCount == 4, "Ground preview mesh is invalid.");
                Require(mesh.vertices[2].z > mesh.vertices[0].z,
                        "Image top must map toward positive preview Z.");
                Require(mesh.uv[2] == new Vector2(0f, 1f),
                        "Top-left image UV convention was not preserved.");
                Require(EditorBuildSettings.scenes.Length == 1 &&
                        EditorBuildSettings.scenes[0].path == "Assets/Scenes/Stage1Preview.unity",
                        "Generated preview is not the only enabled scene.");
                Debug.Log("STAGE1 UNITY SELFTEST PASS");
            }
            finally
            {
                // Restore the repository to its honest state: no real Stage 1 bundle has passed.
                EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
                if (!string.IsNullOrEmpty(generatedAssetFolder))
                    AssetDatabase.DeleteAsset(generatedAssetFolder);
                AssetDatabase.DeleteAsset("Assets/Scenes/Stage1Preview.unity");
                Stage1ProjectBootstrap.InitializeProject();
                if (Directory.Exists(temporaryBundle))
                    Directory.Delete(temporaryBundle, true);
                AssetDatabase.Refresh();
            }
        }

        private static void CreateFixtureBundle(string directory)
        {
            var texture = new Texture2D(
                FixtureWidth, FixtureHeight, TextureFormat.RGBA32, false);
            var colors = new Color[FixtureWidth * FixtureHeight];
            for (int y = 0; y < FixtureHeight; y++)
            {
                for (int x = 0; x < FixtureWidth; x++)
                {
                    colors[y * FixtureWidth + x] = new Color(
                        x / (float)(FixtureWidth - 1),
                        y / (float)(FixtureHeight - 1),
                        0.25f,
                        1f);
                }
            }
            texture.SetPixels(colors);
            texture.Apply();
            File.WriteAllBytes(Path.Combine(directory, "ground.png"), texture.EncodeToPNG());

            for (int index = 0; index < colors.Length; index++)
                colors[index] = Color.white;
            texture.SetPixels(colors);
            texture.Apply();
            File.WriteAllBytes(Path.Combine(directory, "valid_mask.png"), texture.EncodeToPNG());
            UnityEngine.Object.DestroyImmediate(texture);

            var report = new Stage1ReportData
            {
                schema_version = "stage1-ground-v1",
                stage = 1,
                pass = true,
                failure_reasons = Array.Empty<string>(),
                input = new Stage1InputData
                {
                    path = "selftest.mp4",
                    sha256 = new string('a', 64),
                    width = FixtureWidth,
                    height = FixtureHeight,
                },
                reference = new Stage1ReferenceData { video_frame = 10, time_s = 0.5f },
                outputs = new Stage1OutputData
                {
                    ground = "ground.png",
                    valid_mask = "valid_mask.png",
                },
            };
            File.WriteAllText(Path.Combine(directory, "report.json"),
                              JsonUtility.ToJson(report, true));
        }

        private static void Require(bool condition, string message)
        {
            if (!condition)
                throw new InvalidOperationException(message);
        }
    }
}
