using System.IO;
using DroneWorld.Stage1;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace DroneWorld.Stage1.Editor
{
    internal static class Stage1SceneBuilder
    {
        private const string PreviewScenePath = "Assets/Scenes/Stage1Preview.unity";
        private const float PreviewWidth = 10f;

        private const string RebuildMenu =
            "Drone World/Advanced/Rebuild Preview From Selected Manifest";

        [MenuItem(RebuildMenu)]
        private static void RebuildSelectedManifest()
        {
            Stage1ArtifactManifest manifest = Selection.activeObject as Stage1ArtifactManifest;
            if (manifest == null)
            {
                EditorUtility.DisplayDialog(
                    "No Stage 1 manifest selected",
                    "Select a generated Stage1ArtifactManifest asset before rebuilding its preview.",
                    "OK");
                return;
            }
            BuildPreviewScene(AssetDatabase.GetAssetPath(manifest));
        }

        [MenuItem(RebuildMenu, true)]
        private static bool CanRebuildSelectedManifest()
        {
            return Selection.activeObject is Stage1ArtifactManifest;
        }

        internal static void BuildPreviewScene(string manifestPath)
        {
            Stage1ArtifactManifest manifest =
                AssetDatabase.LoadAssetAtPath<Stage1ArtifactManifest>(manifestPath);
            if (manifest == null)
                throw new FileNotFoundException("Unity could not load the Stage 1 manifest.", manifestPath);
            Texture2D groundTexture =
                AssetDatabase.LoadAssetAtPath<Texture2D>(manifest.GroundTextureAssetPath);
            if (groundTexture == null)
                throw new FileNotFoundException("Unity could not load the Stage 1 ground texture.",
                                                manifest.GroundTextureAssetPath);

            Scene scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            GameObject root = new GameObject("Drone World - Stage 1 Pixel Preview");
            float depth = PreviewWidth * manifest.PixelHeight / manifest.PixelWidth;

            GameObject ground = new GameObject("Validated Static Ground (preview units, not meters)");
            ground.transform.SetParent(root.transform, false);
            MeshFilter meshFilter = ground.AddComponent<MeshFilter>();
            MeshRenderer meshRenderer = ground.AddComponent<MeshRenderer>();
            Stage1GroundView groundView = ground.AddComponent<Stage1GroundView>();

            string assetFolder = Path.GetDirectoryName(manifestPath)!.Replace('\\', '/');
            string meshPath = $"{assetFolder}/GroundPreviewMesh.asset";
            string materialPath = $"{assetFolder}/GroundPreviewMaterial.mat";
            Mesh mesh = CreateGroundMesh(PreviewWidth, depth);
            AssetDatabase.DeleteAsset(meshPath);
            AssetDatabase.CreateAsset(mesh, meshPath);
            meshFilter.sharedMesh = mesh;

            Shader shader = Shader.Find("Unlit/Transparent");
            if (shader == null)
                throw new InvalidDataException("The built-in Unlit/Transparent shader is unavailable.");
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
            groundView.Configure(manifest, meshRenderer, PreviewWidth);

            GameObject cameraObject = new GameObject("Top-down Preview Camera");
            cameraObject.transform.SetParent(root.transform, false);
            Camera camera = cameraObject.AddComponent<Camera>();
            camera.orthographic = true;
            camera.orthographicSize = depth * 0.58f;
            camera.clearFlags = CameraClearFlags.SolidColor;
            camera.backgroundColor = new Color(0.035f, 0.045f, 0.06f, 1f);
            cameraObject.transform.position = new Vector3(0f, 10f, 0f);
            cameraObject.transform.rotation = Quaternion.Euler(90f, 0f, 0f);
            camera.tag = "MainCamera";

            Selection.activeGameObject = ground;
            EditorSceneManager.SaveScene(scene, PreviewScenePath);
            EditorBuildSettings.scenes = new[] { new EditorBuildSettingsScene(PreviewScenePath, true) };
            AssetDatabase.SaveAssets();
            Debug.Log($"Built Stage 1 preview scene: {PreviewScenePath}");
        }

        private static Mesh CreateGroundMesh(float width, float depth)
        {
            float halfWidth = width * 0.5f;
            float halfDepth = depth * 0.5f;
            var mesh = new Mesh { name = "Stage1ReferencePixelPlane" };
            mesh.vertices = new[]
            {
                new Vector3(-halfWidth, 0f, -halfDepth),
                new Vector3( halfWidth, 0f, -halfDepth),
                new Vector3(-halfWidth, 0f,  halfDepth),
                new Vector3( halfWidth, 0f,  halfDepth),
            };
            mesh.uv = new[]
            {
                new Vector2(0f, 0f),
                new Vector2(1f, 0f),
                new Vector2(0f, 1f),
                new Vector2(1f, 1f),
            };
            mesh.triangles = new[] { 0, 2, 1, 2, 3, 1 };
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();
            return mesh;
        }
    }
}
