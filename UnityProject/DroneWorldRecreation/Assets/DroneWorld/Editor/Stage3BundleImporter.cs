using System;
using System.Collections.Generic;
using System.IO;
using System.Security.Cryptography;
using DroneWorld.Stage1;
using DroneWorld.Stage2;
using DroneWorld.Stage3;
using UnityEditor;
using UnityEngine;

namespace DroneWorld.Stage3.Editor
{
    public static class Stage3BundleImporter
    {
        private const string ExpectedSchema = "stage3-tracks-v1";

        [MenuItem("Drone World/Stage 3/Import Passing Stage 3 Bundle...")]
        private static void ImportWithFolderPicker()
        {
            try
            {
                Stage2CalibrationAsset calibration = FindCurrentCalibration();
                string selected = EditorUtility.OpenFolderPanel(
                    "Select a passing Stage 3 tracking bundle",
                    Directory.GetParent(calibration.SourceCalibrationPath)?.FullName,
                    "");
                if (string.IsNullOrEmpty(selected))
                    return;
                ImportBundle(selected, calibration);
            }
            catch (Exception exception)
            {
                Debug.LogException(exception);
                EditorUtility.DisplayDialog(
                    "Stage 3 import rejected", exception.Message, "OK");
            }
        }

        public static void ImportFromCommandLine()
        {
            string[] arguments = Environment.GetCommandLineArgs();
            int option = Array.IndexOf(arguments, "-stage3Bundle");
            if (option < 0 || option + 1 >= arguments.Length)
                throw new ArgumentException(
                    "Missing command-line option: -stage3Bundle <directory>");
            ImportBundle(arguments[option + 1], FindCurrentCalibration());
        }

        internal static Stage3TrackAsset ImportBundle(
            string bundleDirectory, Stage2CalibrationAsset calibration)
        {
            if (calibration == null || !calibration.IsUsable)
                throw new InvalidDataException(
                    "Import a usable Stage 2 calibration before Stage 3.");
            Stage1ArtifactManifest manifest = calibration.SourceManifest;
            string bundleRoot = Path.GetFullPath(bundleDirectory);
            string reportPath = Path.Combine(bundleRoot, "report.json");
            if (!File.Exists(reportPath))
                throw new FileNotFoundException(
                    "The selected directory has no report.json.", reportPath);
            Stage3ReportData report = JsonUtility.FromJson<Stage3ReportData>(
                File.ReadAllText(reportPath));
            ValidateReport(report);
            string tracksPath = ResolveArtifact(
                bundleRoot, report.outputs.tracks, "tracks");
            if (report.artifact_hashes != null && !string.Equals(
                    Sha256(tracksPath), report.artifact_hashes.tracks_sha256,
                    StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException("Stage 3 tracks changed after processing.");
            string mapPath = null;
            if (!string.IsNullOrWhiteSpace(report.outputs.track_map))
            {
                mapPath = ResolveArtifact(bundleRoot, report.outputs.track_map, "track_map");
                if (report.artifact_hashes != null && !string.Equals(
                        Sha256(mapPath), report.artifact_hashes.track_map_sha256,
                        StringComparison.OrdinalIgnoreCase))
                    throw new InvalidDataException("Stage 3 track map changed after processing.");
            }
            Stage3TracksData tracks = JsonUtility.FromJson<Stage3TracksData>(
                File.ReadAllText(tracksPath));
            ValidateProvenance(report, manifest, calibration);
            ValidateTracks(tracks, report, calibration);

            string assetFolder = Path.GetDirectoryName(
                AssetDatabase.GetAssetPath(manifest))!.Replace('\\', '/');
            string importedTracksPath = $"{assetFolder}/stage3_tracks.json";
            string importedReportPath = $"{assetFolder}/stage3_report.json";
            CopyExternalFile(tracksPath, importedTracksPath);
            CopyExternalFile(reportPath, importedReportPath);
            if (mapPath != null)
            {
                CopyExternalFile(mapPath, $"{assetFolder}/stage3_track_map.png");
            }
            AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);

            string assetPath = $"{assetFolder}/Stage3Tracks.asset";
            Stage3TrackAsset asset =
                AssetDatabase.LoadAssetAtPath<Stage3TrackAsset>(assetPath);
            if (asset == null)
            {
                asset = ScriptableObject.CreateInstance<Stage3TrackAsset>();
                AssetDatabase.CreateAsset(asset, assetPath);
            }
            asset.SchemaVersion = tracks.schema_version;
            asset.SourceManifest = manifest;
            asset.SourceCalibration = calibration;
            asset.SourceBundlePath = bundleRoot;
            asset.SourceReportSha256 = Sha256(reportPath);
            asset.SourceTracksSha256 = Sha256(tracksPath);
            asset.SourceCfrVideoSha256 = report.source_stage1.cfr_video_sha256;
            asset.SourceFrameTransformsSha256 =
                report.source_stage1.frame_transforms_sha256;
            asset.SourceCalibrationSha256 =
                report.source_stage2.calibration_sha256;
            asset.FramesPerSecond = tracks.fps;
            asset.FrameCount = tracks.frame_count;
            asset.DurationSeconds = tracks.duration_s;
            asset.GroundY = tracks.ground_y;
            asset.ScaleConfidence = tracks.scale_confidence;
            asset.Tracks = ConvertTracks(tracks.tracks);
            EditorUtility.SetDirty(asset);
            AssetDatabase.SaveAssets();
            Stage3SceneBuilder.BuildPlaybackScene(assetPath);
            asset = AssetDatabase.LoadAssetAtPath<Stage3TrackAsset>(assetPath);
            if (asset == null)
                throw new InvalidDataException(
                    $"Unity could not reload the Stage 3 track asset '{assetPath}'.");
            Debug.Log(
                $"Imported passing Stage 3 bundle: {bundleRoot}; " +
                $"tracks={asset.Tracks.Length}; confidence={asset.ScaleConfidence}");
            return asset;
        }

        private static Stage3ActorTrack[] ConvertTracks(Stage3TrackData[] source)
        {
            source ??= Array.Empty<Stage3TrackData>();
            var result = new Stage3ActorTrack[source.Length];
            for (int trackIndex = 0; trackIndex < source.Length; trackIndex++)
            {
                Stage3TrackData item = source[trackIndex];
                var track = new Stage3ActorTrack
                {
                    Id = item.id,
                    Kind = string.Equals(item.kind, "car", StringComparison.Ordinal)
                        ? Stage3ActorKind.Car : Stage3ActorKind.Person,
                    FirstFrame = item.first_frame,
                    LastFrame = item.last_frame,
                    ObservedFrames = item.observed_frames,
                    MeanConfidence = item.mean_confidence,
                    Samples = new Stage3TrackSample[item.samples.Length],
                };
                for (int sampleIndex = 0; sampleIndex < item.samples.Length; sampleIndex++)
                {
                    Stage3SampleData sample = item.samples[sampleIndex];
                    track.Samples[sampleIndex] = new Stage3TrackSample
                    {
                        Frame = sample.frame,
                        TimeSeconds = sample.t,
                        X = sample.x,
                        Z = sample.z,
                        HeadingDegrees = sample.heading_deg,
                        SpeedUnitsPerSecond = sample.speed_units_per_s,
                        Observed = sample.observed,
                        Confidence = sample.confidence,
                    };
                }
                result[trackIndex] = track;
            }
            return result;
        }

        private static void ValidateReport(Stage3ReportData report)
        {
            if (report == null ||
                !string.Equals(report.schema_version, ExpectedSchema,
                               StringComparison.Ordinal) || report.stage != 3)
                throw new InvalidDataException(
                    $"Expected a {ExpectedSchema}, stage 3 report.");
            if (!report.pass)
                throw new InvalidDataException(
                    "Stage 3 quality gate failed: " +
                    string.Join("; ", report.failure_reasons ?? Array.Empty<string>()));
            if (report.source_stage1 == null || report.source_stage2 == null ||
                report.timeline == null || report.outputs == null)
                throw new InvalidDataException(
                    "Stage 3 report is missing provenance, timeline, or outputs.");
        }

        private static void ValidateProvenance(
            Stage3ReportData report, Stage1ArtifactManifest manifest,
            Stage2CalibrationAsset calibration)
        {
            string stage1ReportPath = Path.Combine(
                manifest.SourceBundlePath, "report.json");
            if (!File.Exists(stage1ReportPath))
                throw new FileNotFoundException(
                    "The source Stage 1 report is missing.", stage1ReportPath);
            Stage3Stage1ReportData stage1 =
                JsonUtility.FromJson<Stage3Stage1ReportData>(
                File.ReadAllText(stage1ReportPath));
            if (stage1 == null || stage1.outputs == null)
                throw new InvalidDataException(
                    "The Stage 1 report no longer contains output paths.");
            if (!string.Equals(
                    Sha256(stage1ReportPath), report.source_stage1.report_sha256,
                    StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException(
                    "Stage 1 report changed after Stage 3 was created.");
            string cfrPath = ResolveArtifact(
                manifest.SourceBundlePath, stage1.outputs.video_cfr, "video_cfr");
            string transformsPath = ResolveArtifact(
                manifest.SourceBundlePath, stage1.outputs.frame_transforms,
                "frame_transforms");
            if (!string.Equals(
                    Sha256(cfrPath), report.source_stage1.cfr_video_sha256,
                    StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException(
                    "Stage 1 CFR video changed after Stage 3 was created.");
            if (!string.Equals(
                    Sha256(transformsPath),
                    report.source_stage1.frame_transforms_sha256,
                    StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException(
                    "Stage 1 frame transforms changed after Stage 3 was created.");
            string calibrationPath = Path.Combine(
                calibration.SourceCalibrationPath, "world_calibration.json");
            if (!File.Exists(calibrationPath) || !string.Equals(
                    Sha256(calibrationPath), report.source_stage2.calibration_sha256,
                    StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException(
                    "Stage 2 calibration changed after Stage 3 was created.");
        }

        private static void ValidateTracks(
            Stage3TracksData data, Stage3ReportData report,
            Stage2CalibrationAsset calibration)
        {
            if (data == null || !string.Equals(
                    data.schema_version, ExpectedSchema, StringComparison.Ordinal))
                throw new InvalidDataException("tracks.json has the wrong schema.");
            if (data.fps <= 0f || data.frame_count <= 0 || data.duration_s <= 0f)
                throw new InvalidDataException("tracks.json has an invalid timeline.");
            if (Mathf.Abs(data.fps - report.timeline.fps) > 0.001f ||
                data.frame_count != report.timeline.frame_count)
                throw new InvalidDataException(
                    "tracks.json timeline differs from the Stage 3 report.");
            float exactDuration = data.frame_count / data.fps;
            if (!float.IsFinite(data.duration_s) ||
                Mathf.Abs(data.duration_s - exactDuration) > 0.001f ||
                Mathf.Abs(data.duration_s - report.timeline.duration_s) > 0.001f)
                throw new InvalidDataException(
                    "Stage 3 duration is not derived from frame_count / fps.");
            if (Mathf.Abs(data.ground_y - calibration.GroundYMetres) > 0.001f)
                throw new InvalidDataException(
                    "Stage 3 ground elevation differs from Stage 2.");
            data.tracks ??= Array.Empty<Stage3TrackData>();
            var identifiers = new HashSet<int>();
            foreach (Stage3TrackData track in data.tracks)
            {
                if (track == null || track.id <= 0 || !identifiers.Add(track.id))
                    throw new InvalidDataException(
                        "Stage 3 track IDs must be unique positive integers.");
                if (track.kind != "car" && track.kind != "person")
                    throw new InvalidDataException(
                        $"Unsupported Stage 3 actor kind: {track.kind}");
                if (track.samples == null || track.samples.Length == 0)
                    throw new InvalidDataException(
                        $"Stage 3 track {track.id} has no samples.");
                if (track.first_frame != track.samples[0].frame ||
                    track.last_frame != track.samples[track.samples.Length - 1].frame ||
                    track.observed_frames < 1 ||
                    track.observed_frames > track.samples.Length ||
                    !float.IsFinite(track.mean_confidence) ||
                    track.mean_confidence < 0f || track.mean_confidence > 1f)
                    throw new InvalidDataException(
                        $"Stage 3 track {track.id} summary does not match its samples.");
                int previousFrame = -1;
                int observedFrames = 0;
                foreach (Stage3SampleData sample in track.samples)
                {
                    if (sample.frame <= previousFrame || sample.frame < 0 ||
                        sample.frame >= data.frame_count)
                        throw new InvalidDataException(
                            $"Stage 3 track {track.id} has invalid frame ordering.");
                    if (!float.IsFinite(sample.t) || !float.IsFinite(sample.x) ||
                        !float.IsFinite(sample.z) ||
                        !float.IsFinite(sample.heading_deg) ||
                        !float.IsFinite(sample.speed_units_per_s) ||
                        !float.IsFinite(sample.confidence))
                        throw new InvalidDataException(
                            $"Stage 3 track {track.id} contains non-finite values.");
                    float exactTime = sample.frame / data.fps;
                    if (Mathf.Abs(sample.t - exactTime) > 0.001f)
                        throw new InvalidDataException(
                            $"Stage 3 track {track.id} has a non-CFR timestamp.");
                    if (sample.confidence < 0f || sample.confidence > 1f ||
                        sample.speed_units_per_s < 0f)
                        throw new InvalidDataException(
                            $"Stage 3 track {track.id} has an invalid sample value.");
                    if (sample.observed)
                        observedFrames++;
                    previousFrame = sample.frame;
                }
                if (observedFrames != track.observed_frames)
                    throw new InvalidDataException(
                        $"Stage 3 track {track.id} observed-frame count is inconsistent.");
            }
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

        private static void CopyExternalFile(string source, string assetPath)
        {
            string destination = Path.GetFullPath(Path.Combine(
                Directory.GetParent(Application.dataPath)!.FullName,
                assetPath.Replace('/', Path.DirectorySeparatorChar)));
            File.Copy(source, destination, true);
        }

        private static string Sha256(string path)
        {
            using SHA256 hash = SHA256.Create();
            using FileStream stream = File.OpenRead(path);
            return BitConverter.ToString(hash.ComputeHash(stream))
                .Replace("-", string.Empty).ToLowerInvariant();
        }
    }
}
