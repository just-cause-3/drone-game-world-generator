using System.IO;
using DroneWorld.Stage2;
using DroneWorld.Stage3;
using DroneWorld.Stage4;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace DroneWorld.Stage4.Editor
{
    internal static class Stage4SceneBuilder
    {
        private const string ReviewScenePath =
            "Assets/Scenes/Stage4SemanticReview.unity";
        private const string MetricScenePath =
            "Assets/Scenes/Stage2MetricWorld.unity";
        private const string PlaybackScenePath =
            "Assets/Scenes/Stage3ActorPlayback.unity";

        internal static void BuildReviewScene(string semanticAssetPath)
        {
            Stage4SemanticAsset semantic =
                AssetDatabase.LoadAssetAtPath<Stage4SemanticAsset>(semanticAssetPath);
            if (semantic == null || !semantic.IsUsable)
                throw new InvalidDataException(
                    "Unity could not load a usable Stage 4A semantic asset.");
            if (!Application.isBatchMode &&
                !EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo())
                throw new System.OperationCanceledException(
                    "Stage 4A review scene creation was cancelled to preserve unsaved changes.");

            if (!File.Exists(AssetPathToAbsolute(MetricScenePath)))
                throw new FileNotFoundException(
                    "Build the Stage 2 metric scene before importing Stage 4A.",
                    MetricScenePath);

            string assetFolder = Path.GetDirectoryName(semanticAssetPath)!
                .Replace('\\', '/');
            string trackAssetPath = $"{assetFolder}/Stage3Tracks.asset";
            Stage3TrackAsset tracks =
                AssetDatabase.LoadAssetAtPath<Stage3TrackAsset>(trackAssetPath);
            bool includeActorPlayback =
                tracks != null && tracks.IsUsable &&
                tracks.SourceManifest == semantic.SourceManifest &&
                tracks.SourceCalibration == semantic.SourceCalibration &&
                File.Exists(AssetPathToAbsolute(PlaybackScenePath));
            string sourceScenePath = includeActorPlayback
                ? PlaybackScenePath : MetricScenePath;
            Scene scene = EditorSceneManager.OpenScene(
                sourceScenePath, OpenSceneMode.Single);
            Stage2MetricGround ground =
                Object.FindAnyObjectByType<Stage2MetricGround>();
            if (ground == null || ground.GroundRenderer == null ||
                ground.Calibration != semantic.SourceCalibration ||
                ground.Stage1Manifest != semantic.SourceManifest)
                throw new InvalidDataException(
                    "The Stage 2 scene does not match this Stage 4A proposal.");
            MeshFilter groundMesh = ground.GetComponent<MeshFilter>();
            if (groundMesh == null || groundMesh.sharedMesh == null)
                throw new InvalidDataException(
                    "Stage 2 metric ground has no mesh.");
            if (includeActorPlayback)
            {
                Stage3PlaybackController playback =
                    Object.FindAnyObjectByType<Stage3PlaybackController>();
                if (playback == null || playback.TrackAsset != tracks)
                    throw new InvalidDataException(
                        "The Stage 3 playback scene does not match the imported tracks.");
            }

            Texture2D overlayTexture = AssetDatabase.LoadAssetAtPath<Texture2D>(
                semantic.SemanticOverlayAssetPath);
            if (overlayTexture == null)
                throw new FileNotFoundException(
                    "The imported semantic overlay is missing.",
                    semantic.SemanticOverlayAssetPath);

            GameObject root = new GameObject(
                "Drone World - Stage 4A Semantic Review (NO GEOMETRY)");
            Stage4SemanticReview review = root.AddComponent<Stage4SemanticReview>();
            GameObject overlay = new GameObject(
                "Semantic Proposal Overlay - REVIEW REQUIRED");
            overlay.transform.SetParent(root.transform, false);
            overlay.transform.position = new Vector3(0f, 0.02f, 0f);
            MeshFilter filter = overlay.AddComponent<MeshFilter>();
            MeshRenderer renderer = overlay.AddComponent<MeshRenderer>();
            filter.sharedMesh = groundMesh.sharedMesh;

            string materialPath = $"{assetFolder}/Stage4SemanticOverlay.mat";
            Shader shader = Shader.Find("Unlit/Texture");
            if (shader == null)
                throw new InvalidDataException(
                    "The built-in Unlit/Texture shader is unavailable.");
            Material material =
                AssetDatabase.LoadAssetAtPath<Material>(materialPath);
            if (material == null)
            {
                material = new Material(shader);
                AssetDatabase.CreateAsset(material, materialPath);
            }
            else
            {
                material.shader = shader;
            }
            material.mainTexture = overlayTexture;
            EditorUtility.SetDirty(material);
            renderer.sharedMaterial = material;
            review.Configure(semantic, renderer);

            Selection.activeGameObject = root;
            EditorSceneManager.MarkSceneDirty(scene);
            EditorSceneManager.SaveScene(scene, ReviewScenePath, false);
            EditorBuildSettings.scenes = new[]
            {
                new EditorBuildSettingsScene(ReviewScenePath, true),
            };
            AssetDatabase.SaveAssets();
            Debug.Log(
                $"Built Stage 4A review scene: {ReviewScenePath}. " +
                (includeActorPlayback
                    ? $"Included Stage 3 playback with {tracks.Tracks.Length} tracks. "
                    : "No matching Stage 3 playback was available. ") +
                "No geometry was generated; operator acceptance is required.");
        }

        private static string AssetPathToAbsolute(string assetPath)
        {
            return Path.GetFullPath(Path.Combine(
                Directory.GetParent(Application.dataPath)!.FullName,
                assetPath.Replace('/', Path.DirectorySeparatorChar)));
        }
    }
}
