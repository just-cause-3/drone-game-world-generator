using System;
using System.IO;
using System.Security.Cryptography;
using DroneWorld.Stage1;
using DroneWorld.Stage2;
using DroneWorld.Stage4;
using UnityEditor;
using UnityEngine;

namespace DroneWorld.Stage4.Editor
{
    internal static class Stage4BundleImporter
    {
        private const string ExpectedSchema = "stage4-semantics-v1";
        private const string ExpectedSubstage = "semantic_proposal";
        private const string AcceptanceSchema = "stage4-semantic-acceptance-v1";

        [MenuItem("Drone World/Stage 4/Import Stage 4A Semantic Proposal...")]
        private static void ImportWithFolderPicker()
        {
            try
            {
                Stage2CalibrationAsset calibration = FindCurrentCalibration();
                string selected = EditorUtility.OpenFolderPanel(
                    "Select a passing Stage 4A semantic bundle",
                    Path.GetDirectoryName(calibration.SourceCalibrationPath), "");
                if (string.IsNullOrEmpty(selected))
                    return;
                ImportBundle(selected, calibration);
            }
            catch (Exception exception)
            {
                Debug.LogException(exception);
                EditorUtility.DisplayDialog(
                    "Stage 4A import rejected", exception.Message, "OK");
            }
        }

        public static void ImportFromCommandLine()
        {
            string[] arguments = Environment.GetCommandLineArgs();
            int option = Array.IndexOf(arguments, "-stage4Bundle");
            if (option < 0 || option + 1 >= arguments.Length)
                throw new ArgumentException(
                    "Missing command-line option: -stage4Bundle <directory>");
            ImportBundle(arguments[option + 1], FindCurrentCalibration());
        }

        internal static Stage4SemanticAsset ImportBundle(
            string bundleDirectory, Stage2CalibrationAsset calibration)
        {
            if (calibration == null || !calibration.IsUsable)
                throw new InvalidDataException(
                    "Import a usable Stage 2 calibration before Stage 4A.");

            Stage1ArtifactManifest manifest = calibration.SourceManifest;
            string bundleRoot = Path.GetFullPath(bundleDirectory);
            string reportPath = Path.Combine(bundleRoot, "report.json");
            if (!File.Exists(reportPath))
                throw new FileNotFoundException(
                    "The selected directory has no report.json.", reportPath);
            Stage4ReportData report = JsonUtility.FromJson<Stage4ReportData>(
                File.ReadAllText(reportPath));
            ValidateReport(report);
            ValidateProvenance(report, manifest, calibration);

            string labelsPath = ResolveArtifact(
                bundleRoot, report.outputs.semantic_labels, "semantic_labels");
            string overlayPath = ResolveArtifact(
                bundleRoot, report.outputs.semantic_overlay, "semantic_overlay");
            string componentsPath = ResolveArtifact(
                bundleRoot, report.outputs.components, "components");
            ValidateArtifactHash(
                labelsPath, report.artifact_hashes.semantic_labels_sha256,
                "semantic labels");
            ValidateArtifactHash(
                overlayPath, report.artifact_hashes.semantic_overlay_sha256,
                "semantic overlay");
            ValidateArtifactHash(
                componentsPath, report.artifact_hashes.components_sha256,
                "components");

            string assetFolder = Path.GetDirectoryName(
                AssetDatabase.GetAssetPath(manifest))!.Replace('\\', '/');
            string importedReportPath = $"{assetFolder}/stage4_report.json";
            string importedLabelsPath = $"{assetFolder}/stage4_semantic_labels.png";
            string importedOverlayPath = $"{assetFolder}/stage4_semantic_overlay.png";
            string importedComponentsPath = $"{assetFolder}/stage4_components.json";
            CopyExternalFile(reportPath, importedReportPath);
            CopyExternalFile(labelsPath, importedLabelsPath);
            CopyExternalFile(overlayPath, importedOverlayPath);
            CopyExternalFile(componentsPath, importedComponentsPath);
            AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
            ConfigureTexture(importedLabelsPath, false, FilterMode.Point);
            ConfigureTexture(importedOverlayPath, true, FilterMode.Bilinear);

            Texture2D labelsTexture =
                AssetDatabase.LoadAssetAtPath<Texture2D>(importedLabelsPath);
            Texture2D overlayTexture =
                AssetDatabase.LoadAssetAtPath<Texture2D>(importedOverlayPath);
            ValidateTextureDimensions(
                labelsTexture, report.quality.width, report.quality.height,
                "semantic labels");
            ValidateTextureDimensions(
                overlayTexture, report.quality.width, report.quality.height,
                "semantic overlay");

            string assetPath = $"{assetFolder}/Stage4Semantics.asset";
            Stage4SemanticAsset asset =
                AssetDatabase.LoadAssetAtPath<Stage4SemanticAsset>(assetPath);
            if (asset == null)
            {
                asset = ScriptableObject.CreateInstance<Stage4SemanticAsset>();
                AssetDatabase.CreateAsset(asset, assetPath);
            }
            asset.SchemaVersion = report.schema_version;
            asset.SourceManifest = manifest;
            asset.SourceCalibration = calibration;
            asset.SourceBundlePath = bundleRoot;
            asset.SourceReportSha256 = ComputeSha256(reportPath);
            asset.SourceLabelsSha256 = ComputeSha256(labelsPath);
            asset.SourceOverlaySha256 = ComputeSha256(overlayPath);
            asset.PixelWidth = report.quality.width;
            asset.PixelHeight = report.quality.height;
            asset.KnownValidAreaPercent = report.quality.known_valid_area_pct;
            asset.BuildingAreaPercent = Coverage(report, "building");
            asset.ProviderImplementation = report.semantic_provider.implementation;
            asset.ProviderDomain = report.semantic_provider.domain;
            asset.RequiresHumanReview = true;
            asset.SemanticLabelsAssetPath = importedLabelsPath;
            asset.SemanticOverlayAssetPath = importedOverlayPath;
            RefreshAcceptanceState(asset);
            EditorUtility.SetDirty(asset);
            AssetDatabase.SaveAssets();

            Stage4SceneBuilder.BuildReviewScene(assetPath);
            asset = AssetDatabase.LoadAssetAtPath<Stage4SemanticAsset>(assetPath);
            if (asset == null)
                throw new InvalidDataException(
                    $"Unity could not reload the Stage 4 semantic asset '{assetPath}'.");
            Debug.Log(
                $"Imported Stage 4A semantic proposal: {bundleRoot}; " +
                $"known={asset.KnownValidAreaPercent:F2}%; " +
                $"accepted={asset.AcceptedForGeometry}");
            return asset;
        }

        internal static Stage2CalibrationAsset FindCurrentCalibration()
        {
            Stage2MetricGround ground =
                UnityEngine.Object.FindAnyObjectByType<Stage2MetricGround>();
            if (ground != null && ground.Calibration != null)
                return ground.Calibration;
            string[] identifiers = AssetDatabase.FindAssets(
                "t:Stage2CalibrationAsset", new[] { "Assets/Generated" });
            if (identifiers.Length == 1)
                return AssetDatabase.LoadAssetAtPath<Stage2CalibrationAsset>(
                    AssetDatabase.GUIDToAssetPath(identifiers[0]));
            throw new InvalidOperationException(
                "Open the Stage 2 metric scene or ensure exactly one Stage 2 calibration exists.");
        }

        internal static void RefreshAcceptanceState(Stage4SemanticAsset asset)
        {
            if (asset == null)
                return;
            asset.AcceptedForGeometry = false;
            asset.AcceptanceRecordPath = string.Empty;
            if (string.IsNullOrWhiteSpace(asset.SourceBundlePath))
                return;
            string path = Path.Combine(
                asset.SourceBundlePath, "semantic_acceptance.json");
            if (!File.Exists(path))
                return;
            Stage4AcceptanceData acceptance =
                JsonUtility.FromJson<Stage4AcceptanceData>(File.ReadAllText(path));
            if (acceptance == null || !acceptance.accepted ||
                !string.Equals(acceptance.schema_version, AcceptanceSchema,
                               StringComparison.Ordinal) || acceptance.stage != 4 ||
                !string.Equals(acceptance.stage4_report_sha256,
                               asset.SourceReportSha256,
                               StringComparison.OrdinalIgnoreCase) ||
                !string.Equals(acceptance.semantic_labels_sha256,
                               asset.SourceLabelsSha256,
                               StringComparison.OrdinalIgnoreCase))
                return;
            asset.AcceptedForGeometry = true;
            asset.AcceptanceRecordPath = path;
        }

        internal static string ComputeSha256(string path)
        {
            using SHA256 hash = SHA256.Create();
            using FileStream stream = File.OpenRead(path);
            return BitConverter.ToString(hash.ComputeHash(stream))
                .Replace("-", string.Empty).ToLowerInvariant();
        }

        private static void ValidateReport(Stage4ReportData report)
        {
            if (report == null ||
                !string.Equals(report.schema_version, ExpectedSchema,
                               StringComparison.Ordinal) || report.stage != 4 ||
                !string.Equals(report.substage, ExpectedSubstage,
                               StringComparison.Ordinal))
                throw new InvalidDataException(
                    $"Expected a {ExpectedSchema} Stage 4A semantic proposal.");
            if (!report.pass)
                throw new InvalidDataException(
                    "Stage 4A structural quality gate failed: " +
                    string.Join("; ", report.failure_reasons ?? Array.Empty<string>()));
            if (!report.requires_human_review || report.ready_for_geometry)
                throw new InvalidDataException(
                    "Stage 4A must require human review and must not claim geometry readiness.");
            if (report.source_stage1 == null || report.source_stage2 == null ||
                report.semantic_provider == null || report.quality == null ||
                report.outputs == null || report.artifact_hashes == null)
                throw new InvalidDataException(
                    "Stage 4A report is missing provenance, provider, quality, or outputs.");
            if (report.quality.width <= 0 || report.quality.height <= 0 ||
                !float.IsFinite(report.quality.known_valid_area_pct) ||
                report.quality.known_valid_area_pct < 0f ||
                report.quality.known_valid_area_pct > 100f)
                throw new InvalidDataException(
                    "Stage 4A quality dimensions or coverage are invalid.");
        }

        private static void ValidateProvenance(
            Stage4ReportData report, Stage1ArtifactManifest manifest,
            Stage2CalibrationAsset calibration)
        {
            if (report.source_stage1.pixel_width != manifest.PixelWidth ||
                report.source_stage1.pixel_height != manifest.PixelHeight ||
                report.quality.width != manifest.PixelWidth ||
                report.quality.height != manifest.PixelHeight)
                throw new InvalidDataException(
                    "Stage 4A dimensions do not match the selected Stage 1 ground.");
            if (!string.Equals(report.source_stage1.source_video_sha256,
                               manifest.SourceVideoSha256,
                               StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException(
                    "Stage 4A was created from a different source video.");

            string stage1ReportPath = Path.Combine(
                manifest.SourceBundlePath, "report.json");
            if (!File.Exists(stage1ReportPath) ||
                !string.Equals(ComputeSha256(stage1ReportPath),
                               report.source_stage1.report_sha256,
                               StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException(
                    "Stage 1 report changed after Stage 4A was created.");
            ValidateArtifactHash(
                AssetPathToAbsolute(manifest.GroundTextureAssetPath),
                report.source_stage1.ground_sha256, "Stage 1 ground");
            ValidateArtifactHash(
                AssetPathToAbsolute(manifest.ValidMaskAssetPath),
                report.source_stage1.valid_mask_sha256, "Stage 1 valid mask");

            string calibrationPath = Path.Combine(
                calibration.SourceCalibrationPath, "world_calibration.json");
            ValidateArtifactHash(
                calibrationPath, report.source_stage2.calibration_sha256,
                "Stage 2 calibration");
        }

        private static float Coverage(Stage4ReportData report, string name)
        {
            foreach (Stage4ClassCoverageData item in
                     report.quality.class_coverage ?? Array.Empty<Stage4ClassCoverageData>())
            {
                if (string.Equals(item.name, name, StringComparison.Ordinal))
                    return item.valid_area_pct;
            }
            return 0f;
        }

        private static void ValidateArtifactHash(
            string path, string expected, string label)
        {
            if (!File.Exists(path))
                throw new FileNotFoundException($"{label} is missing.", path);
            if (string.IsNullOrWhiteSpace(expected) ||
                !string.Equals(ComputeSha256(path), expected,
                               StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException(
                    $"{label} hash does not match the Stage 4A report.");
        }

        private static string ResolveArtifact(
            string root, string relativePath, string label)
        {
            if (string.IsNullOrWhiteSpace(relativePath) ||
                Path.IsPathRooted(relativePath))
                throw new InvalidDataException(
                    $"Artifact '{label}' must use a non-empty relative path.");
            string prefix = root.TrimEnd(
                Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar) +
                Path.DirectorySeparatorChar;
            string resolved = Path.GetFullPath(Path.Combine(root, relativePath));
            if (!resolved.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException(
                    $"Artifact '{label}' escapes its bundle.");
            if (!File.Exists(resolved))
                throw new FileNotFoundException(
                    $"Artifact '{label}' is missing.", resolved);
            return resolved;
        }

        private static void ConfigureTexture(
            string assetPath, bool sRgb, FilterMode filter)
        {
            if (!(AssetImporter.GetAtPath(assetPath) is TextureImporter importer))
                throw new InvalidDataException(
                    $"Unity could not import texture: {assetPath}");
            importer.textureType = TextureImporterType.Default;
            importer.textureShape = TextureImporterShape.Texture2D;
            importer.npotScale = TextureImporterNPOTScale.None;
            importer.maxTextureSize = 16384;
            importer.sRGBTexture = sRgb;
            importer.mipmapEnabled = false;
            importer.wrapMode = TextureWrapMode.Clamp;
            importer.filterMode = filter;
            importer.textureCompression = TextureImporterCompression.Uncompressed;
            importer.SaveAndReimport();
        }

        private static void ValidateTextureDimensions(
            Texture2D texture, int width, int height, string label)
        {
            if (texture == null || texture.width != width || texture.height != height)
                throw new InvalidDataException(
                    $"Imported {label} dimensions do not match the Stage 4A report.");
        }

        private static string AssetPathToAbsolute(string assetPath)
        {
            return Path.GetFullPath(Path.Combine(
                Directory.GetParent(Application.dataPath)!.FullName,
                assetPath.Replace('/', Path.DirectorySeparatorChar)));
        }

        private static void CopyExternalFile(string source, string assetPath)
        {
            File.Copy(source, AssetPathToAbsolute(assetPath), true);
        }
    }
}
