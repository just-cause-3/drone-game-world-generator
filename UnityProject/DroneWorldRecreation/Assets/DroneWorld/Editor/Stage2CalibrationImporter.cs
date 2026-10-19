using System;
using System.IO;
using System.Security.Cryptography;
using DroneWorld.Stage1;
using DroneWorld.Stage2;
using UnityEditor;
using UnityEngine;

namespace DroneWorld.Stage2.Editor
{
    internal static class Stage2CalibrationImporter
    {
        private const string ExpectedSchema = "stage2-calibration-v1";

        [MenuItem("Drone World/Advanced/Import Passing Stage 2 Calibration...")]
        private static void ImportWithFolderPicker()
        {
            try
            {
                Stage1ArtifactManifest manifest = FindCurrentStage1Manifest();
                string selected = EditorUtility.OpenFolderPanel(
                    "Select a passing Stage 2 calibration bundle",
                    Path.GetDirectoryName(manifest.SourceBundlePath), "");
                if (string.IsNullOrEmpty(selected))
                    return;
                ImportCalibration(selected, manifest);
            }
            catch (Exception exception)
            {
                Debug.LogException(exception);
                EditorUtility.DisplayDialog(
                    "Stage 2 import rejected", exception.Message, "OK");
            }
        }

        internal static Stage2CalibrationAsset ImportCalibration(
            string bundleDirectory, Stage1ArtifactManifest manifest)
        {
            if (manifest == null)
                throw new ArgumentNullException(nameof(manifest));
            string bundleRoot = Path.GetFullPath(bundleDirectory);
            string calibrationPath = Path.Combine(
                bundleRoot, "world_calibration.json");
            if (!File.Exists(calibrationPath))
                throw new FileNotFoundException(
                    "The selected directory has no world_calibration.json.",
                    calibrationPath);
            Stage2CalibrationData data = JsonUtility.FromJson<Stage2CalibrationData>(
                File.ReadAllText(calibrationPath));
            Validate(data, manifest);

            string assetFolder = Path.GetDirectoryName(
                AssetDatabase.GetAssetPath(manifest))!.Replace('\\', '/');
            string importedJsonPath = $"{assetFolder}/world_calibration.json";
            string overlaySource = Path.Combine(
                bundleRoot, "calibration_overlay.png");
            string overlayAssetPath = $"{assetFolder}/calibration_overlay.png";
            CopyExternalFile(calibrationPath, importedJsonPath);
            if (File.Exists(overlaySource))
                CopyExternalFile(overlaySource, overlayAssetPath);
            AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
            if (File.Exists(overlaySource))
                ConfigureOverlay(overlayAssetPath);

            string assetPath = $"{assetFolder}/Stage2Calibration.asset";
            Stage2CalibrationAsset asset =
                AssetDatabase.LoadAssetAtPath<Stage2CalibrationAsset>(assetPath);
            if (asset == null)
            {
                asset = ScriptableObject.CreateInstance<Stage2CalibrationAsset>();
                AssetDatabase.CreateAsset(asset, assetPath);
            }
            asset.SchemaVersion = data.schema_version;
            asset.SourceManifest = manifest;
            asset.SourceCalibrationPath = bundleRoot;
            asset.SourceReportSha256 = data.source_stage1.report_sha256;
            asset.SourceGroundSha256 = data.source_stage1.ground_sha256;
            asset.Confidence = data.confidence;
            asset.IsApproximateScale =
                data.coordinate_system.metric_scale_is_assumed;
            asset.AssumedWorldWidthMetres =
                data.coordinate_system.assumed_world_width_m;
            asset.GroundYMetres = data.coordinate_system.ground_y_m;
            asset.ReprojectionRmseMetres = data.quality.reprojection_rmse_m;
            asset.ReprojectionMaxErrorMetres =
                data.quality.reprojection_max_error_m;
            asset.ImageHullCoveragePercent =
                data.quality.image_convex_hull_area_pct;
            asset.ReferencePixelToWorldXZ =
                (float[])data.reference_pixel_to_world_xz_flat.Clone();
            asset.ImageCornersWorldXZ = new Vector2[4];
            for (int index = 0; index < 4; index++)
            {
                Stage2CornerData corner = data.image_corners_world_xz[index];
                asset.ImageCornersWorldXZ[index] =
                    new Vector2(corner.x, corner.z);
            }
            asset.WorldBoundsMin = new Vector2(
                data.world_bounds.min_x, data.world_bounds.min_z);
            asset.WorldBoundsMax = new Vector2(
                data.world_bounds.max_x, data.world_bounds.max_z);
            EditorUtility.SetDirty(asset);
            manifest.HasMetricCalibration = true;
            EditorUtility.SetDirty(manifest);
            AssetDatabase.SaveAssets();

            Stage2SceneBuilder.BuildMetricScene(assetPath);
            asset = AssetDatabase.LoadAssetAtPath<Stage2CalibrationAsset>(assetPath);
            if (asset == null)
                throw new InvalidDataException(
                    $"Unity could not reload the Stage 2 calibration '{assetPath}'.");
            Debug.Log(
                $"Imported passing Stage 2 calibration: {bundleRoot}; " +
                $"confidence={data.confidence}; " +
                $"RMSE={data.quality.reprojection_rmse_m:F4} m");
            return asset;
        }

        internal static Stage1ArtifactManifest FindCurrentStage1Manifest()
        {
            if (Selection.activeObject is Stage1ArtifactManifest selected)
                return selected;
            Stage1GroundView view =
                UnityEngine.Object.FindAnyObjectByType<Stage1GroundView>();
            if (view != null && view.Manifest != null)
                return view.Manifest;
            string[] identifiers = AssetDatabase.FindAssets(
                "t:Stage1ArtifactManifest", new[] { "Assets/Generated" });
            if (identifiers.Length == 1)
            {
                return AssetDatabase.LoadAssetAtPath<Stage1ArtifactManifest>(
                    AssetDatabase.GUIDToAssetPath(identifiers[0]));
            }
            throw new InvalidOperationException(
                "Import a passing Stage 1 bundle first. If multiple manifests exist, " +
                "select the intended Stage1Manifest asset in the Project window.");
        }

        private static void Validate(
            Stage2CalibrationData data, Stage1ArtifactManifest manifest)
        {
            if (data == null)
                throw new InvalidDataException(
                    "world_calibration.json is not valid Stage 2 JSON.");
            if (!string.Equals(
                    data.schema_version, ExpectedSchema, StringComparison.Ordinal) ||
                data.stage != 2)
                throw new InvalidDataException(
                    $"Expected {ExpectedSchema}, stage 2.");
            if (!data.pass)
                throw new InvalidDataException(
                    "Stage 2 quality gate failed: " +
                    string.Join("; ", data.failure_reasons ?? Array.Empty<string>()));
            if (data.source_stage1 == null ||
                data.coordinate_system == null ||
                data.quality == null ||
                data.world_bounds == null)
                throw new InvalidDataException(
                    "Stage 2 calibration is missing required sections.");
            if (!string.Equals(
                    data.coordinate_system.units, "meters",
                    StringComparison.Ordinal))
                throw new InvalidDataException(
                    "Stage 2 output units must be metres.");
            if (data.reference_pixel_to_world_xz_flat == null ||
                data.reference_pixel_to_world_xz_flat.Length != 9)
                throw new InvalidDataException(
                    "Stage 2 homography must contain exactly nine values.");
            foreach (float value in data.reference_pixel_to_world_xz_flat)
            {
                if (float.IsNaN(value) || float.IsInfinity(value))
                    throw new InvalidDataException(
                        "Stage 2 homography contains a non-finite value.");
            }
            if (data.image_corners_world_xz == null ||
                data.image_corners_world_xz.Length != 4)
                throw new InvalidDataException(
                    "Stage 2 must contain four mapped image corners.");
            if (data.source_stage1.pixel_width != manifest.PixelWidth ||
                data.source_stage1.pixel_height != manifest.PixelHeight)
                throw new InvalidDataException(
                    "Stage 2 pixel dimensions do not match the selected Stage 1 manifest.");
            if (!string.Equals(
                    data.source_stage1.source_video_sha256,
                    manifest.SourceVideoSha256,
                    StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException(
                    "Stage 2 was created for a different source video.");

            string sourceReport = Path.Combine(
                manifest.SourceBundlePath, "report.json");
            if (!File.Exists(sourceReport) ||
                !string.Equals(
                    Sha256(sourceReport), data.source_stage1.report_sha256,
                    StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException(
                    "Stage 1 report changed after this calibration was created.");
            string importedGround = AssetPathToAbsolute(
                manifest.GroundTextureAssetPath);
            if (!File.Exists(importedGround) ||
                !string.Equals(
                    Sha256(importedGround), data.source_stage1.ground_sha256,
                    StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException(
                    "Stage 1 ground changed after this calibration was created.");
        }

        private static string Sha256(string path)
        {
            using SHA256 hash = SHA256.Create();
            using FileStream stream = File.OpenRead(path);
            return BitConverter.ToString(hash.ComputeHash(stream))
                .Replace("-", string.Empty)
                .ToLowerInvariant();
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

        private static void ConfigureOverlay(string assetPath)
        {
            if (!(AssetImporter.GetAtPath(assetPath) is TextureImporter importer))
                throw new InvalidDataException(
                    "Unity could not import the Stage 2 calibration overlay.");
            importer.textureType = TextureImporterType.Default;
            importer.textureShape = TextureImporterShape.Texture2D;
            importer.npotScale = TextureImporterNPOTScale.None;
            importer.maxTextureSize = 16384;
            importer.sRGBTexture = true;
            importer.mipmapEnabled = false;
            importer.wrapMode = TextureWrapMode.Clamp;
            importer.filterMode = FilterMode.Bilinear;
            importer.textureCompression = TextureImporterCompression.Uncompressed;
            importer.SaveAndReimport();
        }
    }
}
