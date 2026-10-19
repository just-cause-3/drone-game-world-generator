using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEngine;

namespace DroneWorld.Stage1.Editor
{
    public static class Stage1BundleImporter
    {
        private const string ExpectedSchema = "stage1-ground-v1";
        private const string GeneratedRoot = "Assets/Generated";

        [MenuItem("Drone World/Import Passing Stage 1 Bundle...")]
        private static void ImportWithFolderPicker()
        {
            string selected = EditorUtility.OpenFolderPanel(
                "Select a passing Stage 1 bundle", Directory.GetParent(Application.dataPath)?.FullName, "");
            if (string.IsNullOrEmpty(selected))
                return;

            try
            {
                ImportBundle(selected);
            }
            catch (Exception exception)
            {
                Debug.LogException(exception);
                EditorUtility.DisplayDialog("Stage 1 import rejected", exception.Message, "OK");
            }
        }

        /// <summary>Entry point for CI/batch validation.</summary>
        public static void ImportFromCommandLine()
        {
            try
            {
                string[] arguments = Environment.GetCommandLineArgs();
                int option = Array.IndexOf(arguments, "-stage1Bundle");
                if (option < 0 || option + 1 >= arguments.Length)
                    throw new ArgumentException("Missing command-line option: -stage1Bundle <directory>");
                ImportBundle(arguments[option + 1]);
            }
            catch (Exception exception)
            {
                Debug.LogException(exception);
                throw;
            }
        }

        internal static Stage1ArtifactManifest ImportBundle(string bundleDirectory)
        {
            string bundleRoot = Path.GetFullPath(bundleDirectory);
            string reportPath = Path.Combine(bundleRoot, "report.json");
            if (!File.Exists(reportPath))
                throw new FileNotFoundException("The selected directory has no report.json.", reportPath);

            Stage1ReportData report = JsonUtility.FromJson<Stage1ReportData>(File.ReadAllText(reportPath));
            ValidateReport(report);

            string groundSource = ResolveArtifact(bundleRoot, report.outputs.ground, "ground");
            string validMaskSource = ResolveArtifact(bundleRoot, report.outputs.valid_mask, "valid_mask");
            Dictionary<string, string> optionalImages = ResolveOptionalImages(bundleRoot, report.outputs);

            string bundleId = string.IsNullOrEmpty(report.input.sha256)
                ? Guid.NewGuid().ToString("N").Substring(0, 12)
                : report.input.sha256.Substring(0, Math.Min(12, report.input.sha256.Length));
            string assetFolder = $"{GeneratedRoot}/Stage1_{bundleId}";
            EnsureAssetFolder(assetFolder);

            CopyExternalFile(groundSource, $"{assetFolder}/ground.png");
            CopyExternalFile(validMaskSource, $"{assetFolder}/valid_mask.png");
            foreach (KeyValuePair<string, string> image in optionalImages)
                CopyExternalFile(image.Value, $"{assetFolder}/{image.Key}.png");
            CopyExternalFile(reportPath, $"{assetFolder}/source_report.json");
            AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);

            ConfigureTexture($"{assetFolder}/ground.png", true,
                             report.input.width, report.input.height);
            ConfigureTexture($"{assetFolder}/valid_mask.png", false,
                             report.input.width, report.input.height);
            foreach (string name in optionalImages.Keys)
                ConfigureTexture($"{assetFolder}/{name}.png", false,
                                 report.input.width, report.input.height);
            ValidateImportedDimensions($"{assetFolder}/ground.png",
                                       report.input.width, report.input.height, "ground");
            ValidateImportedDimensions($"{assetFolder}/valid_mask.png",
                                       report.input.width, report.input.height, "valid_mask");

            string manifestPath = $"{assetFolder}/Stage1Manifest.asset";
            Stage1ArtifactManifest manifest = AssetDatabase.LoadAssetAtPath<Stage1ArtifactManifest>(manifestPath);
            if (manifest == null)
            {
                manifest = ScriptableObject.CreateInstance<Stage1ArtifactManifest>();
                AssetDatabase.CreateAsset(manifest, manifestPath);
            }

            manifest.SchemaVersion = report.schema_version;
            manifest.SourceBundlePath = bundleRoot;
            manifest.SourceVideoPath = report.input.path;
            manifest.SourceVideoSha256 = report.input.sha256;
            manifest.PixelWidth = report.input.width;
            manifest.PixelHeight = report.input.height;
            manifest.ReferenceVideoFrame = report.reference.video_frame;
            manifest.ReferenceTimeSeconds = report.reference.time_s;
            manifest.GroundTextureAssetPath = $"{assetFolder}/ground.png";
            manifest.ValidMaskAssetPath = $"{assetFolder}/valid_mask.png";
            manifest.HasMetricCalibration = false;
            EditorUtility.SetDirty(manifest);
            AssetDatabase.SaveAssets();

            Stage1SceneBuilder.BuildPreviewScene(manifestPath);
            // Opening a new scene may unload the native object represented by the
            // earlier C# wrapper. Return a fresh Asset Database reference so a
            // same-call Stage 2 import never receives Unity's fake-null object.
            manifest = AssetDatabase.LoadAssetAtPath<Stage1ArtifactManifest>(manifestPath);
            if (manifest == null)
                throw new InvalidDataException(
                    $"Unity could not reload the Stage 1 manifest '{manifestPath}'.");
            Debug.Log($"Imported passing Stage 1 bundle: {bundleRoot}");
            return manifest;
        }

        private static void ValidateReport(Stage1ReportData report)
        {
            if (report == null)
                throw new InvalidDataException("report.json is not valid JSON for the Stage 1 schema.");
            if (!string.Equals(report.schema_version, ExpectedSchema, StringComparison.Ordinal))
                throw new InvalidDataException(
                    $"Unsupported Stage 1 schema '{report.schema_version}'. Expected '{ExpectedSchema}'.");
            if (report.stage != 1)
                throw new InvalidDataException($"Expected stage 1, but report declares stage {report.stage}.");
            if (!report.pass)
            {
                string reasons = report.failure_reasons == null
                    ? "No reasons were recorded."
                    : string.Join("; ", report.failure_reasons);
                throw new InvalidDataException($"Stage 1 report failed its quality gate: {reasons}");
            }
            if (report.input == null || report.reference == null || report.outputs == null)
                throw new InvalidDataException("Stage 1 report is missing input, reference, or outputs data.");
            if (report.input.width <= 0 || report.input.height <= 0)
                throw new InvalidDataException("Stage 1 report contains invalid image dimensions.");
        }

        private static Dictionary<string, string> ResolveOptionalImages(
            string root, Stage1OutputData outputs)
        {
            var candidates = new Dictionary<string, string>
            {
                { "temporal_variation", outputs.temporal_variation },
                { "foreground_frequency", outputs.foreground_frequency },
                { "semantic_exclusion_mask", outputs.semantic_exclusion },
                { "inpaint_mask", outputs.inpaint_mask },
                { "learned_inpaint_mask", outputs.learned_inpaint_mask },
            };
            return candidates
                .Where(pair => !string.IsNullOrWhiteSpace(pair.Value))
                .ToDictionary(pair => pair.Key,
                              pair => ResolveArtifact(root, pair.Value, pair.Key));
        }

        private static string ResolveArtifact(string root, string relativePath, string label)
        {
            if (string.IsNullOrWhiteSpace(relativePath))
                throw new InvalidDataException($"Stage 1 report has no '{label}' output path.");
            if (Path.IsPathRooted(relativePath))
                throw new InvalidDataException($"Artifact '{label}' must use a relative path.");

            string rootPrefix = root.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar)
                                + Path.DirectorySeparatorChar;
            string resolved = Path.GetFullPath(Path.Combine(root, relativePath));
            if (!resolved.StartsWith(rootPrefix, StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException($"Artifact '{label}' escapes its Stage 1 bundle.");
            if (!File.Exists(resolved))
                throw new FileNotFoundException($"Stage 1 artifact '{label}' is missing.", resolved);
            return resolved;
        }

        private static void EnsureAssetFolder(string assetFolder)
        {
            string current = "Assets";
            foreach (string segment in assetFolder.Substring("Assets/".Length).Split('/'))
            {
                string child = $"{current}/{segment}";
                if (!AssetDatabase.IsValidFolder(child))
                    AssetDatabase.CreateFolder(current, segment);
                current = child;
            }
        }

        private static void CopyExternalFile(string source, string assetPath)
        {
            string destination = Path.GetFullPath(Path.Combine(
                Directory.GetParent(Application.dataPath)!.FullName,
                assetPath.Replace('/', Path.DirectorySeparatorChar)));
            File.Copy(source, destination, true);
        }

        private static void ConfigureTexture(string assetPath, bool colorTexture,
                                             int expectedWidth, int expectedHeight)
        {
            if (!(AssetImporter.GetAtPath(assetPath) is TextureImporter importer))
                throw new InvalidDataException($"Unity could not import texture '{assetPath}'.");

            const int maximumSupportedDimension = 16384;
            int largestDimension = Math.Max(expectedWidth, expectedHeight);
            if (largestDimension <= 0 || largestDimension > maximumSupportedDimension)
            {
                throw new InvalidDataException(
                    $"Texture '{assetPath}' declares unsupported dimensions " +
                    $"{expectedWidth}x{expectedHeight}; maximum is " +
                    $"{maximumSupportedDimension} pixels per side.");
            }

            importer.textureType = TextureImporterType.Default;
            importer.textureShape = TextureImporterShape.Texture2D;
            // Stage 1 works in exact video-pixel coordinates. Unity's default
            // ToNearest policy would silently turn 1280x720 into 1024x512.
            importer.npotScale = TextureImporterNPOTScale.None;
            importer.maxTextureSize = Math.Max(32, Mathf.NextPowerOfTwo(largestDimension));
            importer.sRGBTexture = colorTexture;
            importer.alphaIsTransparency = colorTexture;
            importer.mipmapEnabled = false;
            importer.wrapMode = TextureWrapMode.Clamp;
            importer.filterMode = colorTexture ? FilterMode.Bilinear : FilterMode.Point;
            importer.textureCompression = TextureImporterCompression.Uncompressed;
            importer.SaveAndReimport();
        }

        private static void ValidateImportedDimensions(string assetPath, int expectedWidth,
                                                       int expectedHeight, string label)
        {
            Texture2D texture = AssetDatabase.LoadAssetAtPath<Texture2D>(assetPath);
            if (texture == null || texture.width != expectedWidth || texture.height != expectedHeight)
            {
                string actual = texture == null ? "unreadable" : $"{texture.width}x{texture.height}";
                throw new InvalidDataException(
                    $"Artifact '{label}' is {actual}; report declares {expectedWidth}x{expectedHeight}.");
            }
        }
    }
}
