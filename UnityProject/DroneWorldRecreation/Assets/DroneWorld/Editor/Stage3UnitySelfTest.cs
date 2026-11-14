using System;
using System.Reflection;
using DroneWorld.Stage3;
using UnityEngine;
using UnityEditor.SceneManagement;
using UnityEngine.SceneManagement;

namespace DroneWorld.Stage3.Editor
{
    /// <summary>Run with -executeMethod DroneWorld.Stage3.Editor.Stage3UnitySelfTest.Run.</summary>
    public static class Stage3UnitySelfTest
    {
        public static void Run()
        {
            if (Application.isPlaying)
                throw new InvalidOperationException("Run the self-test outside Play Mode.");
            Scene originalScene = SceneManager.GetActiveScene();
            Scene testScene = EditorSceneManager.NewScene(
                NewSceneSetup.EmptyScene, NewSceneMode.Additive);
            SceneManager.SetActiveScene(testScene);
            var data = ScriptableObject.CreateInstance<Stage3TrackAsset>();
            var root = new GameObject("Stage3 playback test");
            var template = new GameObject("Actor test template");
            try
            {
                data.FramesPerSecond = 10f;
                data.FrameCount = 10;
                data.DurationSeconds = 1f;
                data.GroundY = 2f;
                data.Tracks = new[]
                {
                    new Stage3ActorTrack
                    {
                        Id = 1,
                        Kind = Stage3ActorKind.Car,
                        Samples = new[]
                        {
                            Sample(0, 0f, 170f), Sample(1, 1f, -170f),
                            Sample(5, 5f, -170f), Sample(6, 6f, -170f),
                        },
                    },
                    new Stage3ActorTrack
                    {
                        Id = 2, Kind = Stage3ActorKind.Person,
                        Samples = new[] { Sample(0, 3f, 0f), Sample(1, 3f, 0f) },
                    },
                    new Stage3ActorTrack
                    {
                        Id = 3, Kind = Stage3ActorKind.Car,
                        Samples = new[] { Sample(0, 8f, 0f), Sample(1, 8f, 0f) },
                    },
                };
                var controller = root.AddComponent<Stage3PlaybackController>();
                controller.Configure(data, template, template);
                typeof(Stage3PlaybackController).GetMethod(
                    "BuildActors", BindingFlags.NonPublic | BindingFlags.Instance)!
                    .Invoke(controller, null);
                Require(root.transform.childCount == 2,
                    "Keep the moving car and waiting pedestrian; parked cars remain opt-in.");
                Transform actor = root.transform.GetChild(0);

                controller.Seek(0.05f);
                Require(root.transform.GetChild(1).gameObject.activeSelf,
                    "A confirmed pedestrian must not disappear solely for standing still.");
                Require(actor.gameObject.activeSelf, "Consecutive observations must be visible.");
                Require(Mathf.Abs(actor.position.x - 0.5f) < 1e-5f, "Consecutive samples must interpolate.");
                Require(Mathf.Abs(actor.position.y - 2f) < 1e-5f, "Ground elevation must be preserved.");
                Require(Mathf.Abs(Mathf.DeltaAngle(actor.eulerAngles.y, 180f)) < 0.001f,
                    "Heading must interpolate through the shortest angle.");

                controller.Seek(0.15f);
                Require(actor.gameObject.activeSelf && Mathf.Abs(actor.position.x - 1f) < 1e-5f,
                    "Last supported frame before a gap must hold its measured position.");
                controller.Seek(0.3f);
                Require(!actor.gameObject.activeSelf, "An unfilled occlusion must stay hidden.");
                controller.Seek(0.5f);
                Require(actor.gameObject.activeSelf && Mathf.Abs(actor.position.x - 5f) < 1e-5f,
                    "Actor must return at the next supported sample.");
                controller.Seek(0.8f);
                Require(!actor.gameObject.activeSelf, "Actor must hide after the last frame.");
                Debug.Log("STAGE3 UNITY SELFTEST PASS");
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(root);
                UnityEngine.Object.DestroyImmediate(template);
                UnityEngine.Object.DestroyImmediate(data);
                if (originalScene.IsValid()) SceneManager.SetActiveScene(originalScene);
                EditorSceneManager.CloseScene(testScene, true);
            }
        }

        private static Stage3TrackSample Sample(int frame, float x, float heading) =>
            new Stage3TrackSample
            {
                Frame = frame, TimeSeconds = frame / 10f, X = x, Z = 0f,
                HeadingDegrees = heading, Observed = true, Confidence = 0.9f,
            };

        private static void Require(bool condition, string message)
        {
            if (!condition)
                throw new InvalidOperationException(message);
        }
    }
}
