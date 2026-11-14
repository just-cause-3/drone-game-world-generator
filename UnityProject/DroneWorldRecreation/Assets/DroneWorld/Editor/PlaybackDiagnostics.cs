using System;
using System.Collections.Generic;
using System.IO;
using DroneWorld.Stage3;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace DroneWorld.Diagnostics.Editor
{
    /// <summary>Opt-in real-scene Play Mode smoke test. Never saves or replaces scenes.</summary>
    [InitializeOnLoad]
    public static class PlaybackDiagnostics
    {
        private const string Key = "DroneWorld.PlaybackDiagnostics.Active";
        private static readonly string Output = Path.GetFullPath("Temp/DroneWorldPlaybackCheck");
        private static double started;
        private static float lastTime;
        private static int updates;
        private static int wraps;
        private static readonly List<string> errors = new List<string>();
        private static readonly List<string> captures = new List<string>();
        private static Stage3PlaybackController controller;

        static PlaybackDiagnostics()
        {
            EditorApplication.update += Tick;
            EditorApplication.playModeStateChanged += OnState;
            Application.logMessageReceived += OnLog;
        }

        [MenuItem("Drone World/Diagnostics/Check Current Scene Playback")]
        public static void Begin()
        {
            if (EditorApplication.isPlayingOrWillChangePlaymode || EditorApplication.isCompiling)
                throw new InvalidOperationException("Finish compilation and exit Play Mode first.");
            // Entering/exiting Play Mode preserves the editor's unsaved scene
            // state. The self-test uses its own temporary additive scene.
            var playback = UnityEngine.Object.FindAnyObjectByType<Stage3PlaybackController>();
            if (playback == null || playback.TrackAsset == null || !playback.TrackAsset.IsUsable)
                throw new InvalidOperationException("Open a generated scene with usable Stage 3 playback first.");
            Directory.CreateDirectory(Output);
            string blocked = Path.Combine(Output, "blocked.txt");
            if (File.Exists(blocked)) File.Delete(blocked);
            errors.Clear();
            captures.Clear();
            if (File.Exists(Path.Combine(Output, "report.json")))
                File.Delete(Path.Combine(Output, "report.json"));
            Stage3.Editor.Stage3UnitySelfTest.Run();
            SessionState.SetBool(Key, true);
            SessionState.SetString(Key + ".scene", SceneManager.GetActiveScene().path);
            SessionState.SetString(Key + ".source", playback.TrackAsset.SourceBundlePath);
            EditorApplication.isPlaying = true;
        }

        [MenuItem("Drone World/Diagnostics/Refresh Tracks In Current World...")]
        public static void RefreshTracksMenu()
        {
            string bundle = EditorUtility.OpenFolderPanel("Select passing Stage 3 bundle", "", "");
            if (!string.IsNullOrWhiteSpace(bundle)) RefreshTracks(bundle);
        }

        private static void RefreshTracks(string bundle)
        {
            if (EditorApplication.isPlayingOrWillChangePlaymode || SceneManager.GetActiveScene().isDirty)
                throw new InvalidOperationException("Save scene changes and exit Play Mode before refreshing tracks.");
            var playback = UnityEngine.Object.FindAnyObjectByType<Stage3PlaybackController>();
            if (playback == null || playback.TrackAsset == null)
                throw new InvalidOperationException("The current world has no playback tracks.");
            Stage3.Editor.Stage3BundleImporter.ImportBundle(
                bundle, playback.TrackAsset.SourceCalibration, rebuildScene: false);
            // Old generated scenes serialized this as false. Make the corrected
            // visibility policy explicit when the operator requests this refresh.
            Undo.RecordObject(playback, "Include confirmed stationary pedestrians");
            var serialized = new SerializedObject(playback);
            serialized.FindProperty("showStationaryPeople").boolValue = true;
            serialized.ApplyModifiedProperties();
            EditorSceneManager.MarkSceneDirty(playback.gameObject.scene);
            EditorSceneManager.SaveScene(playback.gameObject.scene);
        }

        private static void OnState(PlayModeStateChange state)
        {
            if (!SessionState.GetBool(Key, false)) return;
            if (state == PlayModeStateChange.EnteredPlayMode)
            {
                started = EditorApplication.timeSinceStartup;
                updates = wraps = 0;
                lastTime = 0;
                errors.Clear();
                captures.Clear();
            }
            else if (state == PlayModeStateChange.ExitingPlayMode)
            {
                if (!File.Exists(Path.Combine(Output, "report.json")))
                    Finish("Playback interrupted before completion.");
                SessionState.SetBool(Key, false);
            }
        }

        private static void OnLog(string message, string stack, LogType type)
        {
            if (SessionState.GetBool(Key, false) &&
                (type == LogType.Error || type == LogType.Exception || type == LogType.Assert))
                errors.Add(message);
        }

        private static void Tick()
        {
            string refresh = Path.Combine(Output, "refresh.request");
            if (!SessionState.GetBool(Key, false) && File.Exists(refresh) &&
                !EditorApplication.isCompiling && !EditorApplication.isUpdating)
            {
                string bundle = File.ReadAllText(refresh).Trim();
                File.Delete(refresh);
                try { RefreshTracks(bundle); }
                catch (Exception ex) { File.WriteAllText(Path.Combine(Output, "blocked.txt"), ex.Message); return; }
            }
            string request = Path.Combine(Output, "run.request");
            if (!SessionState.GetBool(Key, false) && File.Exists(request) &&
                !EditorApplication.isCompiling && !EditorApplication.isUpdating)
            {
                File.Delete(request);
                try { Begin(); }
                catch (Exception ex) { Directory.CreateDirectory(Output); File.WriteAllText(Path.Combine(Output, "blocked.txt"), ex.Message); }
            }
            if (!SessionState.GetBool(Key, false) || !EditorApplication.isPlaying) return;
            // A script refresh can reload this class while already in Play Mode,
            // without another EnteredPlayMode event. Restart the observation clock.
            if (started <= 0) started = EditorApplication.timeSinceStartup;
            try
            {
                controller = UnityEngine.Object.FindAnyObjectByType<Stage3PlaybackController>();
                if (controller == null) throw new InvalidOperationException("Playback controller disappeared.");
                // Start() runs after entering Play Mode, so allow its first frame.
                if (controller.transform.childCount == 0 && EditorApplication.timeSinceStartup - started < 3) return;
                if (controller.PlaybackTimeSeconds != lastTime) updates++;
                if (controller.PlaybackTimeSeconds < lastTime) wraps++;
                lastTime = controller.PlaybackTimeSeconds;
                if (captures.Count < 5 && lastTime >= captures.Count * 2f)
                    Capture("live_" + captures.Count, lastTime);
                double elapsed = EditorApplication.timeSinceStartup - started;
                if (elapsed < controller.TrackAsset.DurationSeconds + 2 && elapsed < 60) return;
                if (updates < 5) throw new InvalidOperationException("Playback clock did not advance.");
                controller.Pause();
                foreach (float time in new[] { 0f, 2f, 4f, 6f, 8f })
                {
                    if (time >= controller.TrackAsset.DurationSeconds) continue;
                    controller.Seek(time);
                    if (Mathf.Abs(controller.PlaybackTimeSeconds - time) > 0.001f)
                        throw new InvalidOperationException("Seek did not reach requested time.");
                    Capture("seek_" + time.ToString("0", System.Globalization.CultureInfo.InvariantCulture), time);
                    Capture("topdown_" + time.ToString("0", System.Globalization.CultureInfo.InvariantCulture), time, true);
                }
                controller.ResetPlayback();
                if (controller.PlaybackTimeSeconds != 0) throw new InvalidOperationException("Reset failed.");
                Finish(null);
            }
            catch (Exception ex) { Finish(ex.ToString()); }
        }

        private static void Capture(string name, float time, bool topDown = false)
        {
            Camera camera = Camera.main;
            if (camera == null) throw new InvalidOperationException("No MainCamera for playback capture.");
            var target = RenderTexture.GetTemporary(1280, 720, 24);
            var previous = camera.targetTexture;
            var active = RenderTexture.active;
            var position = camera.transform.position;
            var rotation = camera.transform.rotation;
            bool orthographic = camera.orthographic;
            float orthoSize = camera.orthographicSize;
            float aspect = camera.aspect;
            var texture = new Texture2D(1280, 720, TextureFormat.RGB24, false);
            try
            {
                camera.targetTexture = target;
                camera.aspect = 1280f / 720f;
                if (topDown)
                {
                    var calibration = controller.TrackAsset.SourceCalibration;
                    Vector2 min = calibration.WorldBoundsMin, max = calibration.WorldBoundsMax;
                    Vector2 centre = (min + max) * 0.5f;
                    camera.transform.SetPositionAndRotation(
                        new Vector3(centre.x, calibration.GroundYMetres + 50f, centre.y), Quaternion.Euler(90f, 0f, 0f));
                    camera.orthographic = true;
                    camera.orthographicSize = Mathf.Max((max.y - min.y) / 2f, (max.x - min.x) / (2f * camera.aspect));
                }
                camera.Render();
                RenderTexture.active = target;
                texture.ReadPixels(new Rect(0, 0, 1280, 720), 0, 0);
                texture.Apply();
                File.WriteAllBytes(Path.Combine(Output, name + ".png"), texture.EncodeToPNG());
                int visible = 0;
                foreach (Transform actor in controller.transform)
                    if (actor.gameObject.activeInHierarchy) visible++;
                captures.Add($"{name}.png: time={time:F4}, visibleActors={visible}");
            }
            finally
            {
                camera.targetTexture = previous;
                camera.transform.SetPositionAndRotation(position, rotation);
                camera.orthographic = orthographic;
                camera.orthographicSize = orthoSize;
                camera.aspect = aspect;
                RenderTexture.active = active;
                RenderTexture.ReleaseTemporary(target);
                UnityEngine.Object.DestroyImmediate(texture);
            }
        }

        private static void Finish(string failure)
        {
            if (failure != null) errors.Add(failure);
            var report = new Report
            {
                passed = errors.Count == 0, scene = SessionState.GetString(Key + ".scene", ""),
                trackBundle = SessionState.GetString(Key + ".source", ""),
                clockUpdates = updates, loopWraps = wraps,
                captures = captures.ToArray(), errors = errors.ToArray(),
            };
            File.WriteAllText(Path.Combine(Output, "report.json"), JsonUtility.ToJson(report, true));
            SessionState.SetBool(Key, false);
            EditorApplication.isPlaying = false;
            Debug.Log("DRONE WORLD PLAYBACK CHECK " + (report.passed ? "PASS" : "FAIL") + ": " + Output);
        }

        [Serializable] private sealed class Report
        {
            public bool passed;
            public string scene, trackBundle;
            public int clockUpdates, loopWraps;
            public string[] captures, errors;
        }
    }
}
