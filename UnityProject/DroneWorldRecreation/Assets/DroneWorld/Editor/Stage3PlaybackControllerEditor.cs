using DroneWorld.Stage3;
using UnityEditor;
using UnityEngine;

namespace DroneWorld.Stage3.Editor
{
    [CustomEditor(typeof(Stage3PlaybackController))]
    public sealed class Stage3PlaybackControllerEditor : UnityEditor.Editor
    {
        public override void OnInspectorGUI()
        {
            DrawDefaultInspector();
            var controller = (Stage3PlaybackController)target;
            Stage3TrackAsset tracks = controller.TrackAsset;

            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Playback", EditorStyles.boldLabel);
            if (!Application.isPlaying)
            {
                EditorGUILayout.HelpBox(
                    "Enter Play Mode to instantiate and replay the validated tracks.",
                    MessageType.Info);
                return;
            }
            if (tracks == null || !tracks.IsUsable)
            {
                EditorGUILayout.HelpBox(
                    "No usable Stage 3 track asset is configured.", MessageType.Error);
                return;
            }

            float requested = EditorGUILayout.Slider(
                "Timeline (seconds)", controller.PlaybackTimeSeconds,
                0f, tracks.DurationSeconds);
            if (!Mathf.Approximately(requested, controller.PlaybackTimeSeconds))
            {
                controller.Seek(requested);
                EditorApplication.QueuePlayerLoopUpdate();
            }

            using (new EditorGUILayout.HorizontalScope())
            {
                if (GUILayout.Button(controller.IsPlaying ? "Pause" : "Play"))
                {
                    if (controller.IsPlaying)
                        controller.Pause();
                    else
                        controller.Play();
                }
                if (GUILayout.Button("Reset"))
                {
                    controller.ResetPlayback();
                    EditorApplication.QueuePlayerLoopUpdate();
                }
            }
            EditorGUILayout.LabelField(
                "Source clock",
                $"{tracks.FramesPerSecond:0.###} FPS / {tracks.FrameCount} frames");
        }

        public override bool RequiresConstantRepaint()
        {
            return Application.isPlaying &&
                   ((Stage3PlaybackController)target).IsPlaying;
        }
    }
}
