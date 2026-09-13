using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace DroneWorld.Stage1.Editor
{
    public static class Stage1ProjectBootstrap
    {
        private const string WaitingScenePath = "Assets/Scenes/Stage1AwaitingValidInput.unity";

        [InitializeOnLoadMethod]
        private static void OpenWaitingSceneForFreshEditor()
        {
            EditorApplication.delayCall += () =>
            {
                if (EditorApplication.isPlayingOrWillChangePlaymode)
                    return;
                Scene active = SceneManager.GetActiveScene();
                if (IsUntouchedDefaultScene(active))
                    InitializeProject();
            };
        }

        private static bool IsUntouchedDefaultScene(Scene scene)
        {
            if (scene.name != "Untitled")
                return false;
            GameObject[] roots = scene.GetRootGameObjects();
            if (roots.Length > 2)
                return false;
            foreach (GameObject root in roots)
            {
                if (root.name != "Main Camera" && root.name != "Directional Light")
                    return false;
            }
            return true;
        }

        /// <summary>Creates the fail-closed scene present before any bundle passes.</summary>
        [MenuItem("Drone World/Reset to Waiting Scene")]
        public static void InitializeProject()
        {
            PlayerSettings.companyName = "Drone World Reconstruction";
            PlayerSettings.productName = "Drone World Recreation";

            Scene scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            GameObject root = new GameObject("STAGE 1 BLOCKED - Import a passing artifact bundle");

            GameObject platform = GameObject.CreatePrimitive(PrimitiveType.Cube);
            platform.name = "Waiting Platform (not generated world geometry)";
            platform.transform.SetParent(root.transform, false);
            platform.transform.position = new Vector3(0f, -0.25f, 0f);
            platform.transform.localScale = new Vector3(8f, 0.4f, 4f);

            GameObject messageObject = new GameObject("Stage 1 status message");
            messageObject.transform.SetParent(root.transform, false);
            messageObject.transform.position = new Vector3(0f, 1.25f, 0f);
            TextMesh message = messageObject.AddComponent<TextMesh>();
            message.text = "STAGE 1 WAITING\nNo passing ground bundle has been imported";
            message.anchor = TextAnchor.MiddleCenter;
            message.alignment = TextAlignment.Center;
            message.fontSize = 48;
            message.characterSize = 0.16f;
            message.color = new Color(1f, 0.65f, 0.15f, 1f);

            GameObject cameraObject = new GameObject("Preview Camera");
            cameraObject.transform.SetParent(root.transform, false);
            cameraObject.transform.position = new Vector3(0f, 4.5f, -8f);
            cameraObject.transform.LookAt(new Vector3(0f, 0.5f, 0f));
            Camera camera = cameraObject.AddComponent<Camera>();
            camera.clearFlags = CameraClearFlags.SolidColor;
            camera.backgroundColor = new Color(0.035f, 0.045f, 0.06f, 1f);
            camera.tag = "MainCamera";

            EditorSceneManager.SaveScene(scene, WaitingScenePath);
            EditorBuildSettings.scenes = new[] { new EditorBuildSettingsScene(WaitingScenePath, true) };
            AssetDatabase.SaveAssets();
            Debug.Log("Drone World project initialized. Stage 1 remains blocked until a passing bundle is imported.");
        }
    }
}
