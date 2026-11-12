using System;
using System.IO;
using UnityEditor;
using UnityEngine;

namespace DroneWorld.FullPipeline.Editor
{
    internal static class PipelineRunFolder
    {
        private const string Marker = ".droneworld-full-pipeline-run";
        [Serializable] private sealed class Owner { public string video, pipeline_mode; }

        internal static string Problem(string video, string folder, bool fast)
        {
            if (string.IsNullOrWhiteSpace(folder)) return "No output folder was selected.";
            string root = Path.GetFullPath(folder);
            if (File.Exists(root)) return "The output path is a file, not a folder.";
            if (string.Equals(root.TrimEnd('\\', '/'), Path.GetPathRoot(root)?.TrimEnd('\\', '/'), StringComparison.OrdinalIgnoreCase))
                return "A drive root cannot be an output folder.";
            if (!Directory.Exists(root)) return null;
            if (Directory.GetFileSystemEntries(root).Length == 0) return null;
            string marker = Path.Combine(root, Marker);
            if (!File.Exists(marker))
                return "This folder already contains other files (possibly your input videos). " +
                    "It is not an isolated pipeline output folder.";
            try
            {
                var owner = JsonUtility.FromJson<Owner>(File.ReadAllText(marker));
                if (owner == null || string.IsNullOrWhiteSpace(owner.video) ||
                    !string.Equals(Path.GetFullPath(owner.video), Path.GetFullPath(video), StringComparison.OrdinalIgnoreCase))
                    return "This output folder belongs to another video.";
                if ((string.IsNullOrEmpty(owner.pipeline_mode) ? "accuracy" : owner.pipeline_mode) != (fast ? "fast_balanced" : "accuracy"))
                    return "Normal and fast processing must use separate output folders.";
            }
            catch (Exception) { return "This folder's pipeline ownership marker cannot be read."; }
            return null;
        }

        internal static string Suggested(string video, bool fast)
        {
            string name = string.IsNullOrWhiteSpace(video) ? "new_video" : Path.GetFileNameWithoutExtension(video);
            string candidate = Path.Combine(FullPipelineRunner.RepositoryRoot, "runs", name + (fast ? "_fast" : "_full"));
            string basePath = candidate;
            int suffix = 2;
            while (File.Exists(candidate) || Directory.Exists(candidate)) candidate = basePath + "_" + suffix++;
            return candidate;
        }

        internal static string Confirm(string video, string folder, bool fast)
        {
            string problem = Problem(video, folder, fast);
            if (problem == null) return Path.GetFullPath(folder);
            string suggested = Suggested(video, fast);
            return EditorUtility.DisplayDialog("Choose a dedicated output folder",
                problem + "\n\nYour existing files will not be changed.\nUse this new folder instead?\n" + suggested,
                "Use New Run Folder", "Cancel") ? suggested : null;
        }
    }
}
