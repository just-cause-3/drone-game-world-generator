using System;
using System.IO;
using System.Security.Cryptography;
using UnityEngine;

namespace DroneWorld.FullPipeline.Editor
{
    /// <summary>Read-only cross-stage preflight before any Unity assets are changed.</summary>
    internal static class FullPipelineValidation
    {
        internal static void Validate(string root)
        {
            var pipeline = Read<FullPipelineReportData>(Path.Combine(root, "pipeline_report.json"));
            if (pipeline.schema_version != "droneworld-full-pipeline-v1" ||
                !pipeline.pass || pipeline.status != "pass_review_required" || pipeline.stages == null)
                throw new InvalidDataException("The Stage 1-4A processing report is incomplete or does not pass.");
            var summaries = new[] { pipeline.stages.stage1, pipeline.stages.stage2,
                                   pipeline.stages.stage3, pipeline.stages.stage4 };
            var schemas = new[] { "stage1-ground-v1", "stage2-calibration-v1",
                                  "stage3-tracks-v1", "stage4-semantics-v1" };
            var paths = new string[4];
            var reports = new Bundle[4];
            for (int i = 0; i < 4; i++)
            {
                string directory = Path.Combine(root, "stage" + (i + 1));
                paths[i] = Path.Combine(directory, i == 1 ? "world_calibration.json" : "report.json");
                if (summaries[i] == null || !summaries[i].pass)
                    throw new InvalidDataException($"The pipeline has no passing Stage {i + 1} summary.");
                HashMatches(paths[i], summaries[i].report_sha256, required: false);
                reports[i] = Read<Bundle>(paths[i]);
                if (!reports[i].pass || reports[i].schema_version != schemas[i] || reports[i].stage != i + 1)
                    throw new InvalidDataException($"Stage {i + 1} report is invalid or does not pass.");
                if (reports[i].outputs == null)
                    throw new InvalidDataException($"Stage {i + 1} declares no artifacts.");
                string[] required = i == 0 ? new[] { "ground", "valid_mask", "video_cfr", "frame_transforms" }
                    : i == 1 ? new[] { "overlay", "control_points" }
                    : i == 2 ? new[] { "tracks", "detections", "tracking_preview", "track_map", "frame_counts" }
                    : new[] { "semantic_labels", "semantic_overlay", "components" };
                foreach (string name in required)
                {
                    string relative = (string)typeof(Outputs).GetField(name).GetValue(reports[i].outputs);
                    if (string.IsNullOrWhiteSpace(relative))
                        throw new InvalidDataException($"Stage {i + 1} has no {name} artifact.");
                    string artifact = Path.GetFullPath(Path.Combine(directory, relative));
                    if (!artifact.StartsWith(Path.GetFullPath(directory) + Path.DirectorySeparatorChar,
                                             StringComparison.OrdinalIgnoreCase))
                        throw new InvalidDataException("Artifact path escapes its stage folder: " + relative);
                    if (!File.Exists(artifact)) throw new FileNotFoundException("Missing " + name, artifact);
                    var field = typeof(Hashes).GetField(name + "_sha256");
                    string expected = reports[i].artifact_hashes == null || field == null ? null
                        : (string)field.GetValue(reports[i].artifact_hashes);
                    HashMatches(artifact, expected, required: i == 3);
                }
            }
            for (int i = 1; i < 4; i++)
            {
                HashMatches(paths[0], reports[i].source_stage1?.report_sha256, required: true);
                if (i > 1) HashMatches(paths[1], reports[i].source_stage2?.calibration_sha256, required: true);
            }
        }

        private static T Read<T>(string path) where T : class
        {
            if (!File.Exists(path)) throw new FileNotFoundException("Missing pipeline artifact", path);
            return JsonUtility.FromJson<T>(File.ReadAllText(path)) ??
                throw new InvalidDataException("Invalid JSON: " + path);
        }

        private static void HashMatches(string path, string expected, bool required)
        {
            if (string.IsNullOrWhiteSpace(expected))
            {
                if (required) throw new InvalidDataException("Missing integrity hash for " + path);
                return; // Legacy reports did not snapshot every output hash.
            }
            using var stream = File.OpenRead(path);
            using var sha = SHA256.Create();
            string actual = BitConverter.ToString(sha.ComputeHash(stream)).Replace("-", "").ToLowerInvariant();
            if (!string.Equals(actual, expected, StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException("Stale or modified pipeline artifact: " + path);
        }

        [Serializable] private sealed class Bundle
        {
            public string schema_version;
            public int stage;
            public bool pass;
            public Outputs outputs;
            public Hashes artifact_hashes;
            public Source source_stage1, source_stage2;
        }
        [Serializable] private sealed class Source { public string report_sha256, calibration_sha256; }
        [Serializable] private sealed class Outputs
        {
            public string ground, valid_mask, video_cfr, frame_transforms, overlay, control_points;
            public string tracks, detections, tracking_preview, track_map, frame_counts;
            public string semantic_labels, semantic_overlay, components;
        }
        [Serializable] private sealed class Hashes
        {
            public string tracks_sha256, detections_sha256, tracking_preview_sha256, track_map_sha256, frame_counts_sha256;
            public string semantic_labels_sha256, semantic_overlay_sha256, components_sha256;
        }
    }
}
