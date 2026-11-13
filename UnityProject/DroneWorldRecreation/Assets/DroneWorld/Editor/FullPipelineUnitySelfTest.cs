using System;
using System.IO;
using System.Security.Cryptography;
using UnityEditor;
using UnityEngine;

namespace DroneWorld.FullPipeline.Editor
{
    /// <summary>Read-only production preflight and isolated fixture checks; never imports a scene.</summary>
    internal static class FullPipelineUnitySelfTest
    {
        [InitializeOnLoadMethod]
        private static void Install()
        {
            EditorApplication.delayCall += () =>
            {
                const string request = "Temp/FullPipelineSelfTest.request";
                if (!File.Exists(request)) return;
                File.Delete(request);
                try { Run(); }
                catch (Exception ex) { File.WriteAllText("Temp/FullPipelineSelfTest.result", ex.ToString()); Debug.LogException(ex); }
            };
        }

        [MenuItem("Drone World/Diagnostics/Check Pipeline Orchestration")]
        public static void Run()
        {
            string root = Path.GetFullPath(Path.Combine("Temp", "PipelineFixture-" + Guid.NewGuid().ToString("N")));
            Directory.CreateDirectory(root);
            string input = Path.Combine(root, "input.mp4");
            File.WriteAllText(input, "fixture");
            Require(PipelineRunFolder.Problem(input, root, false) != null, "Input folder must be protected.");
            string fresh = Path.Combine(root, "new_output");
            Require(PipelineRunFolder.Problem(input, fresh, false) == null && !Directory.Exists(fresh),
                "New output path should be accepted without creating it.");
            string owned = Path.Combine(root, "owned");
            Directory.CreateDirectory(owned);
            File.WriteAllText(Path.Combine(owned, ".droneworld-full-pipeline-run"),
                "{\"video\":\"" + input.Replace("\\", "\\\\") + "\",\"pipeline_mode\":\"accuracy\"}");
            Require(PipelineRunFolder.Problem(input, owned, false) == null, "Owned matching run should pass.");
            Require(PipelineRunFolder.Problem(input, owned, true) != null, "Mode mixing must fail.");
            Require(PipelineRunFolder.Problem(input + "other", owned, false) != null, "Video mixing must fail.");

            string[] schemas = { "stage1-ground-v1", "stage2-calibration-v1", "stage3-tracks-v1", "stage4-semantics-v1" };
            string[][] names = {
                new[] { "ground", "valid_mask", "video_cfr", "frame_transforms" },
                new[] { "overlay", "control_points" },
                new[] { "tracks", "detections", "tracking_preview", "track_map", "frame_counts" },
                new[] { "semantic_labels", "semantic_overlay", "components" }
            };
            string[] reports = new string[4];
            for (int i = 0; i < 4; i++)
            {
                string dir = Path.Combine(root, "stage" + (i + 1));
                Directory.CreateDirectory(dir);
                string artifact = Path.Combine(dir, "artifact.bin");
                File.WriteAllText(artifact, "fixture artifact");
                string output = string.Join(",", Array.ConvertAll(names[i], n => "\"" + n + "\":\"artifact.bin\""));
                string hashes = string.Join(",", Array.ConvertAll(names[i], n => "\"" + n + "_sha256\":\"" + Hash(artifact) + "\""));
                string sources = i == 0 ? "" : ",\"source_stage1\":{\"report_sha256\":\"" + Hash(reports[0]) + "\"}";
                if (i > 1) sources += ",\"source_stage2\":{\"calibration_sha256\":\"" + Hash(reports[1]) + "\"}";
                reports[i] = Path.Combine(dir, i == 1 ? "world_calibration.json" : "report.json");
                File.WriteAllText(reports[i], "{\"schema_version\":\"" + schemas[i] + "\",\"stage\":" + (i + 1) +
                    ",\"pass\":true,\"outputs\":{" + output + "},\"artifact_hashes\":{" + hashes + "}" + sources + "}");
            }
            File.WriteAllText(Path.Combine(root, "pipeline_report.json"),
                "{\"schema_version\":\"droneworld-full-pipeline-v1\",\"pass\":true,\"status\":\"pass_review_required\",\"stages\":{" +
                string.Join(",", Array.ConvertAll(new[] { 1, 2, 3, 4 }, i => "\"stage" + i + "\":{\"pass\":true,\"report_sha256\":\"" + Hash(reports[i - 1]) + "\"}")) + "}}");
            FullPipelineValidation.Validate(root);
            string last = Path.Combine(root, "stage4", "artifact.bin");
            File.WriteAllText(last, "tampered");
            bool rejected = false;
            try { FullPipelineValidation.Validate(root); } catch (InvalidDataException) { rejected = true; }
            Require(rejected, "Late-stage tampering must fail before import.");
            string production = Path.Combine(FullPipelineRunner.RepositoryRoot, "runs", "perfect_top_down_full");
            bool checkedProduction = File.Exists(Path.Combine(production, "pipeline_report.json"));
            if (checkedProduction) FullPipelineValidation.Validate(production);
            File.WriteAllText("Temp/FullPipelineSelfTest.result", "PASS: folder ownership, new paths, modes, video identity, report snapshots, artifact integrity; existing run checked=" + checkedProduction);
            Debug.Log("FULL PIPELINE UNITY SELFTEST PASS");
        }

        private static string Hash(string path)
        {
            using var stream = File.OpenRead(path);
            using var sha = SHA256.Create();
            return BitConverter.ToString(sha.ComputeHash(stream)).Replace("-", "").ToLowerInvariant();
        }
        private static void Require(bool condition, string message)
        { if (!condition) throw new InvalidOperationException(message); }
    }
}
