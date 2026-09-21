using System.IO;
using DroneWorld.Stage1;
using DroneWorld.Stage2;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace DroneWorld.Stage2.Editor
{
    internal static class Stage2SceneBuilder
    {
        private const string MetricScenePath = "Assets/Scenes/Stage2MetricWorld.unity";

        internal static void BuildMetricScene(string calibrationAssetPath)
        {
            Stage2CalibrationAsset calibration =
                AssetDatabase.LoadAssetAtPath<Stage2CalibrationAsset>(
                    calibrationAssetPath);
            if (calibration == null || !calibration.IsUsable)
                throw new InvalidDataException(
                    "Unity could not load a usable Stage 2 calibration asset.");
            Stage1ArtifactManifest manifest = calibration.SourceManifest;
            Texture2D groundTexture =
                AssetDatabase.LoadAssetAtPath<Texture2D>(
                    manifest.GroundTextureAssetPath);
            if (groundTexture == null)
                throw new FileNotFoundException(
                    "Unity could not load the Stage 1 ground texture.",
                    manifest.GroundTextureAssetPath);
            if (calibration.ImageCornersWorldXZ == null ||
                calibration.ImageCornersWorldXZ.Length != 4)
                throw new InvalidDataException(
                    "Stage 2 calibration does not contain four image corners.");

            Scene scene = EditorSceneManager.NewScene(
                NewSceneSetup.EmptyScene, NewSceneMode.Single);
            GameObject root = new GameObject(
                "Drone World - Stage 2 Metric Coordinates");
            GameObject ground = new GameObject(
                "Metric Static Ground (1 Unity unit = 1 metre)");
            ground.transform.SetParent(root.transform, false);

            MeshFilter meshFilter = ground.AddComponent<MeshFilter>();
            MeshRenderer meshRenderer = ground.AddComponent<MeshRenderer>();
            MeshCollider meshCollider = ground.AddComponent<MeshCollider>();
            Stage2MetricGround metricGround = ground.AddComponent<Stage2MetricGround>();

            string assetFolder = Path.GetDirectoryName(
                calibrationAssetPath)!.Replace('\\', '/');
            string meshPath = $"{assetFolder}/MetricGroundMesh.asset";
            string materialPath = $"{assetFolder}/MetricGroundMaterial.mat";
            Mesh mesh = CreateMetricMesh(calibration.ImageCornersWorldXZ,
                                         calibration.GroundYMetres);
            AssetDatabase.DeleteAsset(meshPath);
            AssetDatabase.CreateAsset(mesh, meshPath);
            meshFilter.sharedMesh = mesh;
            meshCollider.sharedMesh = mesh;

            Shader shader = Shader.Find("Unlit/Transparent");
            if (shader == null)
                throw new InvalidDataException(
                    "The built-in Unlit/Transparent shader is unavailable.");
            Material material = AssetDatabase.LoadAssetAtPath<Material>(materialPath);
            if (material == null)
            {
                material = new Material(shader);
                AssetDatabase.CreateAsset(material, materialPath);
            }
            else
            {
                material.shader = shader;
            }
            material.mainTexture = groundTexture;
            EditorUtility.SetDirty(material);
            meshRenderer.sharedMaterial = material;
            metricGround.Configure(manifest, calibration, meshRenderer, meshCollider);

            CreateMetricCamera(root.transform, calibration);
            Selection.activeGameObject = ground;
            EditorSceneManager.SaveScene(scene, MetricScenePath);
            EditorBuildSettings.scenes = new[]
            {
                new EditorBuildSettingsScene(MetricScenePath, true),
            };
            AssetDatabase.SaveAssets();
            Debug.Log(
                $"Built Stage 2 metric scene: {MetricScenePath}; " +
                $"RMSE={calibration.ReprojectionRmseMetres:F4} m");
        }

        private static Mesh CreateMetricMesh(Vector2[] corners, float groundY)
        {
            // Python emits TL, TR, BR, BL. Unity UV Y=1 is the image top.
            var mesh = new Mesh { name = "Stage2MetricGroundMesh" };
            mesh.vertices = new[]
            {
                new Vector3(corners[3].x, groundY, corners[3].y), // BL
                new Vector3(corners[2].x, groundY, corners[2].y), // BR
                new Vector3(corners[0].x, groundY, corners[0].y), // TL
                new Vector3(corners[1].x, groundY, corners[1].y), // TR
            };
            mesh.uv = new[]
            {
                new Vector2(0f, 0f),
                new Vector2(1f, 0f),
                new Vector2(0f, 1f),
                new Vector2(1f, 1f),
            };
            Vector3[] vertices = mesh.vertices;
            float proposedNormalY = Vector3.Cross(
                vertices[2] - vertices[0],
                vertices[1] - vertices[0]).y;
            mesh.triangles = proposedNormalY >= 0f
                ? new[] { 0, 2, 1, 2, 3, 1 }
                : new[] { 0, 1, 2, 2, 1, 3 };
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();
            return mesh;
        }

        private static void CreateMetricCamera(
            Transform parent, Stage2CalibrationAsset calibration)
        {
            Vector2 min = calibration.WorldBoundsMin;
            Vector2 max = calibration.WorldBoundsMax;
            float width = Mathf.Max(1f, max.x - min.x);
            float depth = Mathf.Max(1f, max.y - min.y);
            Vector2 center = (min + max) * 0.5f;
            GameObject cameraObject = new GameObject("Top-down Metric Camera");
            cameraObject.transform.SetParent(parent, false);
            Camera camera = cameraObject.AddComponent<Camera>();
            camera.orthographic = true;
            camera.orthographicSize = Mathf.Max(depth, width / (16f / 9f)) * 0.58f;
            camera.clearFlags = CameraClearFlags.SolidColor;
            camera.backgroundColor = new Color(0.035f, 0.045f, 0.06f, 1f);
            cameraObject.transform.position = new Vector3(
                center.x,
                calibration.GroundYMetres + Mathf.Max(width, depth),
                center.y);
            cameraObject.transform.rotation = Quaternion.Euler(90f, 0f, 0f);
            camera.tag = "MainCamera";
        }
    }
}
