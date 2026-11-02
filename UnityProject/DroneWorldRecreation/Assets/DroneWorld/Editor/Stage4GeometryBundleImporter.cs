using System;
using System.Collections.Generic;
using System.IO;
using DroneWorld.Stage4;
using DroneWorld.Stage4.Editor;
using DroneWorld.Stage4Geometry;
using UnityEditor;
using UnityEngine;

namespace DroneWorld.Stage4Geometry.Editor
{
    internal static class Stage4GeometryBundleImporter
    {
        private const string ExpectedSchema = "stage4-geometry-v1";
        private const string ExpectedSubstage = "accepted_geometry";

        [MenuItem("Drone World/Stage 4/Import Accepted Geometry Bundle...")]
        private static void ImportWithFolderPicker()
        {
            try
            {
                Stage4SemanticAsset semantics = FindCurrentSemantics();
                string selected = EditorUtility.OpenFolderPanel(
                    "Select a passing Stage 4B geometry bundle",
                    semantics.SourceBundlePath, "");
                if (!string.IsNullOrEmpty(selected))
                    ImportBundle(selected, semantics);
            }
            catch (Exception exception)
            {
                Debug.LogException(exception);
                EditorUtility.DisplayDialog(
                    "Stage 4B import rejected", exception.Message, "OK");
            }
        }

        internal static Stage4GeometryAsset ImportBundle(
            string bundleDirectory, Stage4SemanticAsset semantics)
        {
            ValidateAcceptedSemantics(semantics);
            string root = Path.GetFullPath(bundleDirectory);
            string reportPath = Path.Combine(root, "report.json");
            if (!File.Exists(reportPath))
                throw new FileNotFoundException(
                    "The selected geometry directory has no report.json.", reportPath);
            Stage4GeometryReportData report =
                JsonUtility.FromJson<Stage4GeometryReportData>(
                    File.ReadAllText(reportPath));
            ValidateReport(report);
            ValidateProvenance(report, semantics);

            string geometryPath = ResolveArtifact(
                root, report.outputs.geometry, "geometry");
            string overlayPath = ResolveArtifact(
                root, report.outputs.geometry_overlay, "geometry overlay");
            ValidateHash(
                geometryPath, report.artifact_hashes.geometry_sha256, "geometry");
            ValidateHash(
                overlayPath, report.artifact_hashes.geometry_overlay_sha256,
                "geometry overlay");
            Stage4GeometryFileData geometry =
                JsonUtility.FromJson<Stage4GeometryFileData>(
                    File.ReadAllText(geometryPath));
            ValidateGeometry(geometry, report, semantics);

            string assetFolder = Path.GetDirectoryName(
                AssetDatabase.GetAssetPath(semantics))!.Replace('\\', '/');
            string importedReportPath = $"{assetFolder}/stage4_geometry_report.json";
            string importedGeometryPath = $"{assetFolder}/stage4_geometry.json";
            string importedOverlayPath = $"{assetFolder}/stage4_geometry_overlay.png";
            CopyExternalFile(reportPath, importedReportPath);
            CopyExternalFile(geometryPath, importedGeometryPath);
            CopyExternalFile(overlayPath, importedOverlayPath);
            AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
            ConfigureOverlay(importedOverlayPath);

            string assetPath = $"{assetFolder}/Stage4Geometry.asset";
            Stage4GeometryAsset asset =
                AssetDatabase.LoadAssetAtPath<Stage4GeometryAsset>(assetPath);
            if (asset == null)
            {
                asset = ScriptableObject.CreateInstance<Stage4GeometryAsset>();
                AssetDatabase.CreateAsset(asset, assetPath);
            }
            asset.SchemaVersion = report.schema_version;
            asset.SourceSemantics = semantics;
            asset.SourceBundlePath = root;
            asset.SourceReportSha256 = Stage4BundleImporter.ComputeSha256(reportPath);
            asset.SourceGeometrySha256 =
                Stage4BundleImporter.ComputeSha256(geometryPath);
            asset.SourceAcceptanceSha256 = report.source_stage4a.acceptance_sha256;
            asset.HeightPolicy = geometry.height_policy.name;
            asset.AllHeightsAreAssumed = geometry.height_policy.all_heights_are_assumed;
            asset.GroundYMetres = geometry.coordinate_system.ground_y_m;
            asset.MetresPerPixelMedian = report.quality.metres_per_pixel_median;
            asset.SurfaceRegionCount = report.quality.surface_region_count;
            asset.Warnings = report.warnings ?? Array.Empty<string>();
            asset.GeometryOverlayAssetPath = importedOverlayPath;
            asset.Buildings = ConvertBuildings(geometry.buildings);
            asset.VegetationProxies = ConvertVegetation(geometry.vegetation_proxies);
            EditorUtility.SetDirty(asset);
            AssetDatabase.SaveAssets();

            Stage4GeometrySceneBuilder.BuildGeneratedScene(assetPath);
            asset = AssetDatabase.LoadAssetAtPath<Stage4GeometryAsset>(assetPath);
            if (asset == null)
                throw new InvalidDataException(
                    $"Unity could not reload the geometry asset '{assetPath}'.");
            Debug.Log(
                $"Imported Stage 4B/C geometry: buildings={asset.Buildings.Length}; " +
                $"vegetation={asset.VegetationProxies.Length}; " +
                $"height policy={asset.HeightPolicy}");
            return asset;
        }

        internal static Stage4SemanticAsset FindCurrentSemantics()
        {
            Stage4SemanticReview review =
                UnityEngine.Object.FindAnyObjectByType<Stage4SemanticReview>();
            if (review != null && review.SemanticAsset != null)
                return review.SemanticAsset;
            Stage4GeneratedWorld generated =
                UnityEngine.Object.FindAnyObjectByType<Stage4GeneratedWorld>();
            if (generated != null && generated.GeometryAsset != null)
                return generated.GeometryAsset.SourceSemantics;
            string[] identifiers = AssetDatabase.FindAssets(
                "t:Stage4SemanticAsset", new[] { "Assets/Generated" });
            if (identifiers.Length == 1)
                return AssetDatabase.LoadAssetAtPath<Stage4SemanticAsset>(
                    AssetDatabase.GUIDToAssetPath(identifiers[0]));
            throw new InvalidOperationException(
                "Open the Stage 4A review scene or ensure exactly one semantic asset exists.");
        }

        internal static void ValidateAcceptedSemantics(Stage4SemanticAsset semantics)
        {
            if (semantics == null || !semantics.IsUsable)
                throw new InvalidDataException(
                    "Import a usable Stage 4A semantic proposal first.");
            Stage4BundleImporter.RefreshAcceptanceState(semantics);
            if (!semantics.AcceptedForGeometry ||
                string.IsNullOrWhiteSpace(semantics.AcceptanceRecordPath) ||
                !File.Exists(semantics.AcceptanceRecordPath))
                throw new InvalidDataException(
                    "Stage 4B is blocked. Review and accept the exact Stage 4A semantic " +
                    "map in the Stage4SemanticReview inspector first.");
        }

        private static Stage4BuildingGeometry[] ConvertBuildings(
            Stage4BuildingData[] source)
        {
            source ??= Array.Empty<Stage4BuildingData>();
            var result = new Stage4BuildingGeometry[source.Length];
            for (int index = 0; index < source.Length; index++)
            {
                Stage4BuildingData item = source[index];
                var footprint = new Vector2[item.footprint_xz.Length];
                for (int point = 0; point < footprint.Length; point++)
                    footprint[point] = new Vector2(
                        item.footprint_xz[point].x, item.footprint_xz[point].z);
                result[index] = new Stage4BuildingGeometry
                {
                    Id = item.id,
                    AreaSquareMetres = item.area_m2,
                    HeightMetres = item.height_m,
                    HeightSource = item.height_source,
                    HeightIsAssumed = item.height_is_assumed,
                    FootprintXZ = footprint,
                    RoofTriangles = (int[])item.roof_triangles.Clone(),
                };
            }
            return result;
        }

        private static Stage4VegetationProxy[] ConvertVegetation(
            Stage4VegetationData[] source)
        {
            source ??= Array.Empty<Stage4VegetationData>();
            var result = new Stage4VegetationProxy[source.Length];
            for (int index = 0; index < source.Length; index++)
            {
                Stage4VegetationData item = source[index];
                result[index] = new Stage4VegetationProxy
                {
                    Id = item.id,
                    X = item.x,
                    Z = item.z,
                    CanopyRadiusMetres = item.canopy_radius_m,
                    HeightMetres = item.height_m,
                    DimensionsAreAssumed = item.dimensions_are_assumed,
                };
            }
            return result;
        }

        private static void ValidateReport(Stage4GeometryReportData report)
        {
            if (report == null ||
                !string.Equals(report.schema_version, ExpectedSchema,
                               StringComparison.Ordinal) ||
                report.stage != 4 ||
                !string.Equals(report.substage, ExpectedSubstage,
                               StringComparison.Ordinal))
                throw new InvalidDataException(
                    $"Expected a passing {ExpectedSchema} geometry report.");
            if (!report.pass || !report.ready_for_unity_geometry)
                throw new InvalidDataException(
                    "Stage 4B geometry quality gate failed: " +
                    string.Join("; ", report.failure_reasons ?? Array.Empty<string>()));
            if (report.source_stage4a == null || report.source_stage2 == null ||
                report.quality == null || report.outputs == null ||
                report.artifact_hashes == null)
                throw new InvalidDataException(
                    "Geometry report is missing provenance, quality, outputs, or hashes.");
            if (report.quality.building_count < 0 ||
                report.quality.building_vertex_count < 0 ||
                report.quality.surface_region_count < 0 ||
                report.quality.vegetation_proxy_count < 0 ||
                !float.IsFinite(report.quality.metres_per_pixel_median) ||
                report.quality.metres_per_pixel_median <= 0f)
                throw new InvalidDataException(
                    "Geometry report contains invalid quality metrics.");
        }

        private static void ValidateProvenance(
            Stage4GeometryReportData report, Stage4SemanticAsset semantics)
        {
            if (!string.Equals(
                    report.source_stage4a.report_sha256,
                    semantics.SourceReportSha256,
                    StringComparison.OrdinalIgnoreCase) ||
                !string.Equals(
                    report.source_stage4a.semantic_labels_sha256,
                    semantics.SourceLabelsSha256,
                    StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException(
                    "Geometry was generated from a different or stale Stage 4A map.");
            string acceptanceHash = Stage4BundleImporter.ComputeSha256(
                semantics.AcceptanceRecordPath);
            if (!string.Equals(
                    report.source_stage4a.acceptance_sha256, acceptanceHash,
                    StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException(
                    "Geometry acceptance hash is stale or belongs to another review.");
            string calibrationPath = Path.Combine(
                semantics.SourceCalibration.SourceCalibrationPath,
                "world_calibration.json");
            ValidateHash(
                calibrationPath, report.source_stage2.calibration_sha256,
                "Stage 2 calibration");
        }

        private static void ValidateGeometry(
            Stage4GeometryFileData geometry, Stage4GeometryReportData report,
            Stage4SemanticAsset semantics)
        {
            if (geometry == null ||
                !string.Equals(geometry.schema_version, ExpectedSchema,
                               StringComparison.Ordinal) ||
                geometry.stage != 4 ||
                !string.Equals(geometry.substage, ExpectedSubstage,
                               StringComparison.Ordinal) ||
                geometry.coordinate_system == null || geometry.height_policy == null)
                throw new InvalidDataException("geometry.json has the wrong schema.");
            if (!string.Equals(geometry.coordinate_system.space, "unity_world_xz",
                               StringComparison.Ordinal) ||
                !string.Equals(geometry.coordinate_system.horizontal_units, "metres",
                               StringComparison.Ordinal) ||
                Mathf.Abs(geometry.coordinate_system.ground_y_m -
                          semantics.SourceCalibration.GroundYMetres) > 0.001f)
                throw new InvalidDataException(
                    "Geometry coordinate system differs from Stage 2.");
            if (!geometry.height_policy.all_heights_are_assumed)
                throw new InvalidDataException(
                    "This footage provides no measured heights; geometry must mark them assumed.");

            geometry.buildings ??= Array.Empty<Stage4BuildingData>();
            geometry.surface_regions ??= Array.Empty<Stage4SurfaceRegionData>();
            geometry.vegetation_proxies ??= Array.Empty<Stage4VegetationData>();
            if (geometry.buildings.Length != report.quality.building_count ||
                geometry.surface_regions.Length != report.quality.surface_region_count ||
                geometry.vegetation_proxies.Length !=
                    report.quality.vegetation_proxy_count)
                throw new InvalidDataException(
                    "Geometry arrays do not match the report counts.");

            var identifiers = new HashSet<string>(StringComparer.Ordinal);
            int vertexCount = 0;
            Vector2 minimum = semantics.SourceCalibration.WorldBoundsMin;
            Vector2 maximum = semantics.SourceCalibration.WorldBoundsMax;
            float tolerance = Mathf.Max(1f, report.quality.metres_per_pixel_median * 4f);
            foreach (Stage4BuildingData building in geometry.buildings)
            {
                if (building == null || string.IsNullOrWhiteSpace(building.id) ||
                    !identifiers.Add(building.id) ||
                    !float.IsFinite(building.area_m2) || building.area_m2 <= 0f ||
                    !float.IsFinite(building.height_m) || building.height_m <= 0f ||
                    !building.height_is_assumed ||
                    building.footprint_xz == null || building.footprint_xz.Length < 3 ||
                    building.roof_triangles == null ||
                    building.roof_triangles.Length !=
                        3 * (building.footprint_xz.Length - 2))
                    throw new InvalidDataException(
                        "A building has invalid identity, dimensions, or triangulation.");
                foreach (Stage4PointXZData point in building.footprint_xz)
                {
                    if (point == null || !float.IsFinite(point.x) ||
                        !float.IsFinite(point.z) || point.x < minimum.x - tolerance ||
                        point.x > maximum.x + tolerance || point.z < minimum.y - tolerance ||
                        point.z > maximum.y + tolerance)
                        throw new InvalidDataException(
                            $"Building {building.id} lies outside the calibrated world.");
                }
                foreach (int triangleIndex in building.roof_triangles)
                {
                    if (triangleIndex < 0 || triangleIndex >= building.footprint_xz.Length)
                        throw new InvalidDataException(
                            $"Building {building.id} has an invalid roof index.");
                }
                vertexCount += building.footprint_xz.Length;
            }
            if (vertexCount != report.quality.building_vertex_count)
                throw new InvalidDataException(
                    "Building vertex count differs from the geometry report.");
            foreach (Stage4VegetationData vegetation in geometry.vegetation_proxies)
            {
                if (vegetation == null || string.IsNullOrWhiteSpace(vegetation.id) ||
                    !identifiers.Add(vegetation.id) || !float.IsFinite(vegetation.x) ||
                    !float.IsFinite(vegetation.z) ||
                    !float.IsFinite(vegetation.canopy_radius_m) ||
                    vegetation.canopy_radius_m <= 0f ||
                    !float.IsFinite(vegetation.height_m) || vegetation.height_m <= 0f ||
                    !vegetation.dimensions_are_assumed)
                    throw new InvalidDataException(
                        "A vegetation proxy has invalid identity or dimensions.");
            }
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

        private static void ValidateHash(string path, string expected, string label)
        {
            if (!File.Exists(path))
                throw new FileNotFoundException($"{label} is missing.", path);
            if (string.IsNullOrWhiteSpace(expected) ||
                !string.Equals(Stage4BundleImporter.ComputeSha256(path), expected,
                               StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException($"{label} hash does not match its report.");
        }

        private static void ConfigureOverlay(string assetPath)
        {
            if (!(AssetImporter.GetAtPath(assetPath) is TextureImporter importer))
                throw new InvalidDataException(
                    $"Unity could not import geometry overlay: {assetPath}");
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

        private static void CopyExternalFile(string source, string assetPath)
        {
            File.Copy(source, AssetPathToAbsolute(assetPath), true);
        }

        private static string AssetPathToAbsolute(string assetPath)
        {
            return Path.GetFullPath(Path.Combine(
                Directory.GetParent(Application.dataPath)!.FullName,
                assetPath.Replace('/', Path.DirectorySeparatorChar)));
        }
    }
}
