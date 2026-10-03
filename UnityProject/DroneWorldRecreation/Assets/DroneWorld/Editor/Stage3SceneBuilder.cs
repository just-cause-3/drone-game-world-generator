using System;
using System.IO;
using DroneWorld.Stage3;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace DroneWorld.Stage3.Editor
{
    internal static class Stage3SceneBuilder
    {
        private const string Stage2ScenePath = "Assets/Scenes/Stage2MetricWorld.unity";
        private const string Stage3ScenePath = "Assets/Scenes/Stage3ActorPlayback.unity";
        private const string CarPrefabPath =
            "Assets/DroneWorld/Stage3/Prefabs/SampleCar.prefab";
        private const string PersonPrefabPath =
            "Assets/DroneWorld/Stage3/Prefabs/SamplePerson.prefab";

        internal static void BuildPlaybackScene(string trackAssetPath)
        {
            Stage3TrackAsset tracks =
                AssetDatabase.LoadAssetAtPath<Stage3TrackAsset>(trackAssetPath);
            if (tracks == null || !tracks.IsUsable)
                throw new InvalidDataException(
                    "Unity could not load a usable Stage 3 track asset.");
            if (!File.Exists(AssetPathToAbsolute(Stage2ScenePath)))
                throw new FileNotFoundException(
                    "Build the Stage 2 metric scene before importing Stage 3.",
                    Stage2ScenePath);
            if (!Application.isBatchMode &&
                !EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo())
                throw new OperationCanceledException(
                    "Stage 3 scene creation was cancelled to preserve unsaved changes.");

            Stage3SampleModelBuilder.CreateSampleActorAssets();
            GameObject carPrefab =
                AssetDatabase.LoadAssetAtPath<GameObject>(CarPrefabPath);
            GameObject personPrefab =
                AssetDatabase.LoadAssetAtPath<GameObject>(PersonPrefabPath);
            if (carPrefab == null || personPrefab == null)
                throw new InvalidDataException(
                    "Stage 3 sample actor prefabs could not be loaded.");

            Scene scene = EditorSceneManager.OpenScene(
                Stage2ScenePath, OpenSceneMode.Single);
            EditorSceneManager.SaveScene(scene, Stage3ScenePath, false);
            DestroyRootNamed("Stage 3 Sample Actors");
            DestroyRootNamed("Stage 3 Real Track Playback");

            GameObject root = new GameObject("Stage 3 Real Track Playback");
            Stage3PlaybackController controller =
                root.AddComponent<Stage3PlaybackController>();
            controller.Configure(tracks, carPrefab, personPrefab);
            Selection.activeGameObject = root;
            EditorSceneManager.MarkSceneDirty(scene);
            EditorSceneManager.SaveScene(scene, Stage3ScenePath);
            EditorBuildSettings.scenes = new[]
            {
                new EditorBuildSettingsScene(Stage3ScenePath, true),
            };
            AssetDatabase.SaveAssets();
            Debug.Log(
                $"Built Stage 3 playback scene: {Stage3ScenePath}; " +
                $"tracks={tracks.Tracks.Length}");
        }

        private static void DestroyRootNamed(string name)
        {
            GameObject target = GameObject.Find(name);
            if (target != null && target.transform.parent == null)
                UnityEngine.Object.DestroyImmediate(target);
        }

        private static string AssetPathToAbsolute(string assetPath)
        {
            return Path.GetFullPath(Path.Combine(
                Directory.GetParent(Application.dataPath)!.FullName,
                assetPath.Replace('/', Path.DirectorySeparatorChar)));
        }
    }
}
