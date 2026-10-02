using System.IO;
using DroneWorld.Stage2;
using DroneWorld.Stage3;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace DroneWorld.Stage3.Editor
{
    /// <summary>Creates readable low-poly placeholders without external model files.</summary>
    public static class Stage3SampleModelBuilder
    {
        private const string RootFolder = "Assets/DroneWorld/Stage3";
        private const string MaterialFolder = RootFolder + "/Materials";
        private const string PrefabFolder = RootFolder + "/Prefabs";
        private const string CarPrefabPath = PrefabFolder + "/SampleCar.prefab";
        private const string PersonPrefabPath = PrefabFolder + "/SamplePerson.prefab";
        private const string ShowcaseName = "Stage 3 Sample Actors";

        [MenuItem("Drone World/Stage 3/Create Sample Actor Prefabs")]
        public static void CreateSampleActorAssets()
        {
            if (!RequireEditMode())
                return;
            EnsureFolder("Assets/DroneWorld", "Stage3");
            EnsureFolder(RootFolder, "Materials");
            EnsureFolder(RootFolder, "Prefabs");

            Material carPaint = CreateMaterial(
                MaterialFolder + "/SampleCarPaint.mat",
                new Color(0.06f, 0.32f, 0.85f, 1f), 0.65f, 0.72f);
            Material glass = CreateMaterial(
                MaterialFolder + "/SampleGlass.mat",
                new Color(0.025f, 0.07f, 0.11f, 1f), 0.25f, 0.9f);
            Material rubber = CreateMaterial(
                MaterialFolder + "/SampleRubber.mat",
                new Color(0.018f, 0.021f, 0.025f, 1f), 0.0f, 0.18f);
            Material marker = CreateMaterial(
                MaterialFolder + "/SampleForwardMarker.mat",
                new Color(1f, 0.24f, 0.035f, 1f), 0.0f, 0.4f);
            Material clothing = CreateMaterial(
                MaterialFolder + "/SamplePersonClothing.mat",
                new Color(1f, 0.52f, 0.06f, 1f), 0.0f, 0.45f);
            Material skin = CreateMaterial(
                MaterialFolder + "/SamplePersonSkin.mat",
                new Color(0.76f, 0.48f, 0.31f, 1f), 0.0f, 0.35f);

            SaveCarPrefab(carPaint, glass, rubber, marker);
            SavePersonPrefab(clothing, skin, marker);
            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();
            Debug.Log($"Created Stage 3 sample prefabs: {CarPrefabPath}, {PersonPrefabPath}");
        }

        [MenuItem("Drone World/Stage 3/Create Sample Actor Prefabs", true)]
        private static bool CanCreateSampleActorAssets()
        {
            return !EditorApplication.isPlayingOrWillChangePlaymode;
        }

        [MenuItem("Drone World/Stage 3/Place Sample Actors In Current Scene")]
        public static void PlaceSampleActorsInCurrentScene()
        {
            if (!RequireEditMode())
                return;
            CreateSampleActorAssets();
            GameObject existing = GameObject.Find(ShowcaseName);
            if (existing != null)
            {
                ConfigurePlayback(existing, calibration: null);
                Selection.activeGameObject = existing;
                EditorGUIUtility.PingObject(existing);
                EditorUtility.DisplayDialog(
                    "Stage 3 sample playback updated",
                    "The existing sample actors now have looping playback. Press the Unity Play button to run it.",
                    "OK");
                return;
            }

            GameObject carPrefab = AssetDatabase.LoadAssetAtPath<GameObject>(CarPrefabPath);
            GameObject personPrefab = AssetDatabase.LoadAssetAtPath<GameObject>(PersonPrefabPath);
            if (carPrefab == null || personPrefab == null)
                throw new InvalidDataException("Stage 3 sample prefabs could not be loaded.");

            Stage2MetricGround metricGround =
                Object.FindAnyObjectByType<Stage2MetricGround>();
            Stage2CalibrationAsset calibration = metricGround != null
                ? metricGround.Calibration
                : null;
            float groundY = calibration != null ? calibration.GroundYMetres : 0f;
            Vector2 centre = calibration != null
                ? (calibration.WorldBoundsMin + calibration.WorldBoundsMax) * 0.5f
                : Vector2.zero;
            float width = calibration != null
                ? Mathf.Max(1f, calibration.WorldBoundsMax.x - calibration.WorldBoundsMin.x)
                : 12f;

            GameObject root = new GameObject(ShowcaseName);
            Undo.RegisterCreatedObjectUndo(root, "Place Stage 3 sample actors");
            GameObject car = (GameObject)PrefabUtility.InstantiatePrefab(carPrefab);
            GameObject person = (GameObject)PrefabUtility.InstantiatePrefab(personPrefab);
            Undo.RegisterCreatedObjectUndo(car, "Place sample car");
            Undo.RegisterCreatedObjectUndo(person, "Place sample person");
            car.transform.SetParent(root.transform, true);
            person.transform.SetParent(root.transform, true);
            root.transform.position = new Vector3(centre.x, groundY, centre.y);
            ConfigurePlayback(root, calibration);

            Selection.activeGameObject = root;
            EditorSceneManager.MarkSceneDirty(root.scene);
            Debug.Log(
                calibration == null
                    ? "Placed Stage 3 sample actors at the scene origin (no Stage 2 ground was found)."
                    : "Placed Stage 3 sample actors on the Stage 2 ground. Their local +Z is forward.");
        }

        private static void ConfigurePlayback(
            GameObject root, Stage2CalibrationAsset calibration)
        {
            Stage3SampleActorVisual[] actors =
                root.GetComponentsInChildren<Stage3SampleActorVisual>(true);
            Transform car = null;
            Transform person = null;
            foreach (Stage3SampleActorVisual actor in actors)
            {
                if (actor.Kind == Stage3ActorKind.Car)
                    car = actor.transform;
                else if (actor.Kind == Stage3ActorKind.Person)
                    person = actor.transform;
            }
            if (car == null || person == null)
                throw new InvalidDataException(
                    "The Stage 3 sample root must contain one car and one person.");

            if (calibration == null)
            {
                Stage2MetricGround metricGround =
                    Object.FindAnyObjectByType<Stage2MetricGround>();
                calibration = metricGround != null ? metricGround.Calibration : null;
            }
            float width = calibration != null
                ? Mathf.Max(4f, calibration.WorldBoundsMax.x - calibration.WorldBoundsMin.x)
                : 12f;
            float depth = calibration != null
                ? Mathf.Max(4f, calibration.WorldBoundsMax.y - calibration.WorldBoundsMin.y)
                : 8f;
            if (calibration != null)
            {
                Vector2 centre =
                    (calibration.WorldBoundsMin + calibration.WorldBoundsMax) * 0.5f;
                root.transform.position = new Vector3(
                    centre.x, calibration.GroundYMetres, centre.y);
            }

            Stage3SamplePlayback playback =
                root.GetComponent<Stage3SamplePlayback>();
            if (playback == null)
                playback = Undo.AddComponent<Stage3SamplePlayback>(root);
            Undo.RecordObject(playback, "Configure Stage 3 sample playback");
            playback.Configure(car, person, width, depth);
            EditorUtility.SetDirty(playback);
            if (!EditorApplication.isPlaying && root.scene.IsValid())
                EditorSceneManager.MarkSceneDirty(root.scene);
        }

        [MenuItem("Drone World/Stage 3/Place Sample Actors In Current Scene", true)]
        private static bool CanPlaceSampleActorsInCurrentScene()
        {
            return !EditorApplication.isPlayingOrWillChangePlaymode;
        }

        private static bool RequireEditMode()
        {
            if (!EditorApplication.isPlayingOrWillChangePlaymode)
                return true;
            EditorUtility.DisplayDialog(
                "Exit Play Mode first",
                "Stage 3 sample assets and scene setup can only be changed in Edit Mode. Exit Play Mode, run this command, save the scene, and then press Play.",
                "OK");
            return false;
        }

        private static void SaveCarPrefab(
            Material paint, Material glass, Material rubber, Material marker)
        {
            GameObject root = new GameObject("SampleCar");
            try
            {
                root.AddComponent<Stage3SampleActorVisual>().Configure(
                    Stage3ActorKind.Car, new Vector3(1.8f, 1.5f, 4.2f));
                BoxCollider collider = root.AddComponent<BoxCollider>();
                collider.center = new Vector3(0f, 0.75f, 0f);
                collider.size = new Vector3(1.8f, 1.5f, 4.2f);

                AddPrimitive(root.transform, PrimitiveType.Cube, "Lower Body", paint,
                    new Vector3(0f, 0.48f, 0f), new Vector3(1.8f, 0.72f, 4.2f));
                AddPrimitive(root.transform, PrimitiveType.Cube, "Cabin", paint,
                    new Vector3(0f, 1.02f, -0.15f), new Vector3(1.52f, 0.55f, 2.15f));
                AddPrimitive(root.transform, PrimitiveType.Cube, "Windshield", glass,
                    new Vector3(0f, 1.22f, 0.79f), new Vector3(1.34f, 0.16f, 0.48f));
                AddPrimitive(root.transform, PrimitiveType.Cube, "Rear Glass", glass,
                    new Vector3(0f, 1.22f, -1.08f), new Vector3(1.34f, 0.16f, 0.38f));

                float[] wheelXs = { -0.93f, 0.93f };
                float[] wheelZs = { -1.28f, 1.28f };
                foreach (float x in wheelXs)
                foreach (float z in wheelZs)
                {
                    GameObject wheel = AddPrimitive(
                        root.transform, PrimitiveType.Cylinder, $"Wheel {x:F2} {z:F2}",
                        rubber, new Vector3(x, 0.34f, z), new Vector3(0.31f, 0.13f, 0.31f));
                    wheel.transform.localRotation = Quaternion.Euler(0f, 0f, 90f);
                }

                AddPrimitive(root.transform, PrimitiveType.Cube, "Forward Marker", marker,
                    new Vector3(0f, 0.88f, 2.13f), new Vector3(0.5f, 0.1f, 0.12f));
                PrefabUtility.SaveAsPrefabAsset(root, CarPrefabPath);
            }
            finally
            {
                Object.DestroyImmediate(root);
            }
        }

        private static void SavePersonPrefab(
            Material clothing, Material skin, Material marker)
        {
            GameObject root = new GameObject("SamplePerson");
            try
            {
                root.AddComponent<Stage3SampleActorVisual>().Configure(
                    Stage3ActorKind.Person, new Vector3(0.58f, 1.75f, 0.58f));
                CapsuleCollider collider = root.AddComponent<CapsuleCollider>();
                collider.center = new Vector3(0f, 0.875f, 0f);
                collider.height = 1.75f;
                collider.radius = 0.29f;

                AddPrimitive(root.transform, PrimitiveType.Capsule, "Body", clothing,
                    new Vector3(0f, 0.82f, 0f), new Vector3(0.56f, 0.65f, 0.56f));
                AddPrimitive(root.transform, PrimitiveType.Sphere, "Head", skin,
                    new Vector3(0f, 1.57f, 0f), new Vector3(0.42f, 0.42f, 0.42f));
                AddPrimitive(root.transform, PrimitiveType.Cube, "Forward Marker", marker,
                    new Vector3(0f, 1.13f, 0.3f), new Vector3(0.22f, 0.14f, 0.12f));
                PrefabUtility.SaveAsPrefabAsset(root, PersonPrefabPath);
            }
            finally
            {
                Object.DestroyImmediate(root);
            }
        }

        private static GameObject AddPrimitive(
            Transform parent, PrimitiveType type, string name, Material material,
            Vector3 localPosition, Vector3 localScale)
        {
            GameObject part = GameObject.CreatePrimitive(type);
            part.name = name;
            part.transform.SetParent(parent, false);
            part.transform.localPosition = localPosition;
            part.transform.localScale = localScale;
            Collider primitiveCollider = part.GetComponent<Collider>();
            if (primitiveCollider != null)
                Object.DestroyImmediate(primitiveCollider);
            MeshRenderer renderer = part.GetComponent<MeshRenderer>();
            renderer.sharedMaterial = material;
            return part;
        }

        private static Material CreateMaterial(
            string path, Color color, float metallic, float smoothness)
        {
            Shader shader = Shader.Find("Universal Render Pipeline/Lit") ??
                            Shader.Find("Standard") ??
                            Shader.Find("Unlit/Color");
            if (shader == null)
                throw new InvalidDataException("No supported sample-model shader was found.");
            Material material = AssetDatabase.LoadAssetAtPath<Material>(path);
            if (material == null)
            {
                material = new Material(shader);
                AssetDatabase.CreateAsset(material, path);
            }
            else
            {
                material.shader = shader;
            }

            if (material.HasProperty("_BaseColor"))
                material.SetColor("_BaseColor", color);
            if (material.HasProperty("_Color"))
                material.SetColor("_Color", color);
            if (material.HasProperty("_Metallic"))
                material.SetFloat("_Metallic", metallic);
            if (material.HasProperty("_Smoothness"))
                material.SetFloat("_Smoothness", smoothness);
            if (material.HasProperty("_Glossiness"))
                material.SetFloat("_Glossiness", smoothness);
            EditorUtility.SetDirty(material);
            return material;
        }

        private static void EnsureFolder(string parent, string name)
        {
            string path = parent + "/" + name;
            if (!AssetDatabase.IsValidFolder(path))
                AssetDatabase.CreateFolder(parent, name);
        }
    }
}
