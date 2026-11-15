using System;
using System.IO;
using UnityEditor;
using UnityEngine;

namespace DroneWorld.FullPipeline.Editor
{
    public sealed class FastPipelineWindow : EditorWindow
    {
        private const string VideoPreference = "DroneWorld.FastPipeline.Video";
        private const string RunPreference = "DroneWorld.FastPipeline.RunRoot";
        private const string WidthPreference = "DroneWorld.FastPipeline.WorldWidth";
        private string videoPath;
        private string runPath;
        private float worldWidth;
        private Vector2 logScroll;
        private bool importScheduled;

        [MenuItem("Drone World/Run Full Pipeline - Fast Mode (Stages 1-4A)...", priority = 2)]
        public static void ShowWindow()
        {
            GetWindow<FastPipelineWindow>("Fast Drone World Pipeline");
        }

        private void OnEnable()
        {
            string repository = FullPipelineRunner.RepositoryRoot;
            videoPath = EditorPrefs.GetString(
                VideoPreference, Path.Combine(repository, "sample_vids", "crossroads.mp4"));
            runPath = EditorPrefs.GetString(
                RunPreference, Path.Combine(repository, "runs", "crossroads_fast"));
            worldWidth = EditorPrefs.GetFloat(WidthPreference, 40f);
            FullPipelineRunner.Changed += OnRunnerChanged;
        }

        private void OnDisable()
        {
            FullPipelineRunner.Changed -= OnRunnerChanged;
            if (importScheduled)
            {
                EditorApplication.delayCall -= ExecuteScheduledImport;
                importScheduled = false;
            }
        }

        private void OnRunnerChanged()
        {
            logScroll.y = float.MaxValue;
            Repaint();
        }

        private void OnGUI()
        {
            EditorGUILayout.Space(8f);
            EditorGUILayout.LabelField(
                "Fast video -> ground -> scale -> actors -> semantic review",
                EditorStyles.boldLabel);
            EditorGUILayout.HelpBox(
                "Balanced fast mode keeps the original resolution, overlapping aerial " +
                "tiles, and test-time augmentation. Stage 3 detects alternate frames, " +
                "then tracks/interpolates actors on the full source frame clock. Use a " +
                "separate *_fast run folder. Stops at semantic review (4A); approval " +
                "and Stage 4B-C geometry are still required for the final world.",
                MessageType.Info);

            using (new EditorGUI.DisabledScope(FullPipelineRunner.IsRunning || importScheduled || EditorApplication.isPlayingOrWillChangePlaymode))
            {
                DrawVideoField();
                DrawRunField();
                worldWidth = EditorGUILayout.FloatField(
                    "Approx. image width (m)", worldWidth);
                EditorGUILayout.HelpBox("Output folder, NOT the folder containing videos. A new path is created automatically.", MessageType.None);
                if (GUILayout.Button("Use New Run Folder"))
                    runPath = PipelineRunFolder.Suggested(videoPath, true);
                if (GUILayout.Button("Run Stages 1-4A - Fast", GUILayout.Height(36f)))
                    RunPipeline();
                if (GUILayout.Button(
                        "Import Existing Passing Fast Run", GUILayout.Height(28f)))
                    SchedulePassingRunImport();
            }

            EditorGUILayout.Space(8f);
            EditorGUILayout.LabelField("Status", EditorStyles.boldLabel);
            MessageType type = FullPipelineRunner.Status.StartsWith("PASS")
                ? MessageType.Info
                : FullPipelineRunner.Status.Contains("FAIL")
                    ? MessageType.Error : MessageType.None;
            EditorGUILayout.HelpBox(FullPipelineRunner.Status, type);
            FullPipelineWindow.DrawRemainingSteps();
            foreach (string reason in FullPipelineRunner.FailureReasons)
                EditorGUILayout.HelpBox(reason, MessageType.Error);

            EditorGUILayout.LabelField("Pipeline log", EditorStyles.boldLabel);
            logScroll = EditorGUILayout.BeginScrollView(
                logScroll, GUILayout.MinHeight(220f));
            string logText = string.Join("\n", FullPipelineRunner.LogLines);
            var style = new GUIStyle(EditorStyles.textArea) { wordWrap = true };
            float width = Mathf.Max(100f, position.width - 38f);
            float height = Mathf.Max(220f, style.CalcHeight(new GUIContent(logText), width));
            EditorGUILayout.SelectableLabel(logText, style, GUILayout.Height(height));
            EditorGUILayout.EndScrollView();
        }

        private void DrawVideoField()
        {
            EditorGUILayout.BeginHorizontal();
            videoPath = EditorGUILayout.TextField("Input video", videoPath);
            if (GUILayout.Button("Browse", GUILayout.Width(72f)))
            {
                string selected = EditorUtility.OpenFilePanel(
                    "Select stationary top-down video",
                    string.IsNullOrWhiteSpace(videoPath)
                        ? FullPipelineRunner.RepositoryRoot
                        : Path.GetDirectoryName(videoPath),
                    "mp4,mov,avi,mkv");
                if (!string.IsNullOrEmpty(selected))
                {
                    videoPath = selected;
                    string name = Path.GetFileNameWithoutExtension(selected);
                    runPath = Path.Combine(
                        FullPipelineRunner.RepositoryRoot, "runs", name + "_fast");
                }
            }
            EditorGUILayout.EndHorizontal();
        }

        private void DrawRunField()
        {
            EditorGUILayout.BeginHorizontal();
            runPath = EditorGUILayout.TextField("Separate fast run folder", runPath);
            if (GUILayout.Button("Browse", GUILayout.Width(72f)))
            {
                string selected = EditorUtility.OpenFolderPanel(
                    "Select a separate fast-mode run folder", runPath, "");
                if (!string.IsNullOrEmpty(selected))
                    runPath = selected;
            }
            EditorGUILayout.EndHorizontal();
        }

        private void RunPipeline()
        {
            try
            {
                string confirmed = PipelineRunFolder.Confirm(videoPath, runPath, true);
                if (confirmed == null) return;
                runPath = confirmed;
                EditorPrefs.SetString(VideoPreference, videoPath);
                EditorPrefs.SetString(RunPreference, runPath);
                EditorPrefs.SetFloat(WidthPreference, worldWidth);
                FullPipelineRunner.StartFast(videoPath, runPath, worldWidth);
            }
            catch (Exception exception)
            {
                Debug.LogException(exception);
                EditorUtility.DisplayDialog(
                    "Could not start fast pipeline", exception.Message, "OK");
            }
        }

        private void SchedulePassingRunImport()
        {
            if (importScheduled)
                return;
            EditorPrefs.SetString(RunPreference, runPath);
            importScheduled = true;
            EditorApplication.delayCall += ExecuteScheduledImport;
        }

        private void ExecuteScheduledImport()
        {
            importScheduled = false;
            try
            {
                FullPipelineRunner.ImportPassingRun(runPath);
            }
            catch (Exception exception)
            {
                Debug.LogException(exception);
                EditorUtility.DisplayDialog(
                    "Could not import passing run", exception.Message, "OK");
            }
        }
    }
}
