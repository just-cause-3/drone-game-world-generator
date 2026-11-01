using System;
using System.Collections.Generic;
using System.IO;
using DroneWorld.Stage2;
using DroneWorld.Stage4;
using DroneWorld.Stage4Geometry;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace DroneWorld.Stage4Geometry.Editor
{
    internal static class Stage4GeometrySceneBuilder
    {
        internal const string GeneratedScenePath =
            "Assets/Scenes/Stage4GeneratedWorld.unity";
        private const string ReviewScenePath =
            "Assets/Scenes/Stage4SemanticReview.unity";

        internal static void BuildGeneratedScene(string geometryAssetPath)
        {
            Stage4GeometryAsset geometry =
                AssetDatabase.LoadAssetAtPath<Stage4GeometryAsset>(geometryAssetPath);
            if (geometry == null || !geometry.IsUsable)
                throw new InvalidDataException(
                    "Unity could not load a usable Stage 4B geometry asset.");
            Stage4GeometryBundleImporter.ValidateAcceptedSemantics(
                geometry.SourceSemantics);
            if (!Application.isBatchMode &&
                !EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo())
                throw new OperationCanceledException(
                    "Geometry scene creation was cancelled to preserve unsaved changes.");
            if (!File.Exists(AssetPathToAbsolute(ReviewScenePath)))
                throw new FileNotFoundException(
                    "Open or rebuild the matching Stage 4A review scene first.",
                    ReviewScenePath);

            Scene scene = EditorSceneManager.OpenScene(
                ReviewScenePath, OpenSceneMode.Single);
            Stage4SemanticReview review =
                UnityEngine.Object.FindAnyObjectByType<Stage4SemanticReview>();
            Stage2MetricGround ground =
                UnityEngine.Object.FindAnyObjectByType<Stage2MetricGround>();
            if (review == null || review.SemanticAsset != geometry.SourceSemantics ||
                ground == null ||
                ground.Calibration != geometry.SourceSemantics.SourceCalibration)
                throw new InvalidDataException(
                    "The Stage 4A review scene does not match this geometry bundle.");

            DestroyRootNamed("Drone World - Stage 4B/C Generated Geometry");
            string assetFolder = Path.GetDirectoryName(geometryAssetPath)!
                .Replace('\\', '/');
            string meshFolder = $"{assetFolder}/Stage4Meshes";
            if (AssetDatabase.IsValidFolder(meshFolder))
                AssetDatabase.DeleteAsset(meshFolder);
            AssetDatabase.CreateFolder(assetFolder, "Stage4Meshes");

            Material wallMaterial = GetOrCreateMaterial(
                $"{assetFolder}/Stage4BuildingWalls.mat",
                new Color(0.55f, 0.58f, 0.62f, 1f), 0.15f);
            Material roofMaterial = GetOrCreateMaterial(
                $"{assetFolder}/Stage4BuildingRoofs.mat",
                new Color(0.75f, 0.77f, 0.80f, 1f), 0.05f);
            Material trunkMaterial = GetOrCreateMaterial(
                $"{assetFolder}/Stage4TreeTrunks.mat",
                new Color(0.28f, 0.16f, 0.08f, 1f), 0.7f);
            Material canopyMaterial = GetOrCreateMaterial(
                $"{assetFolder}/Stage4TreeCanopies.mat",
                new Color(0.12f, 0.43f, 0.16f, 1f), 0.85f);

            GameObject root = new GameObject(
                "Drone World - Stage 4B/C Generated Geometry");
            Stage4GeneratedWorld marker = root.AddComponent<Stage4GeneratedWorld>();
            GameObject buildingRoot = new GameObject(
                $"Buildings ({geometry.Buildings.Length}) - HEIGHTS ASSUMED");
            buildingRoot.transform.SetParent(root.transform, false);
            GameObject vegetationRoot = new GameObject(
                $"Vegetation Proxies ({geometry.VegetationProxies.Length}) - ASSUMED");
            vegetationRoot.transform.SetParent(root.transform, false);

            foreach (Stage4BuildingGeometry building in geometry.Buildings)
                CreateBuilding(
                    building, geometry.GroundYMetres, buildingRoot.transform,
                    meshFolder, wallMaterial, roofMaterial);
            foreach (Stage4VegetationProxy vegetation in geometry.VegetationProxies)
                CreateVegetation(
                    vegetation, geometry.GroundYMetres, vegetationRoot.transform,
                    trunkMaterial, canopyMaterial);

            review.SetOverlayVisible(false);
            review.gameObject.name = "Stage 4A Accepted Semantic Source";
            marker.Configure(
                geometry, buildingRoot.transform, vegetationRoot.transform, review);
            CreatePreviewLighting(root.transform);
            CreatePerspectiveCamera(root.transform, geometry.SourceSemantics.SourceCalibration);

            Selection.activeGameObject = root;
            EditorSceneManager.MarkSceneDirty(scene);
            EditorSceneManager.SaveScene(scene, GeneratedScenePath, false);
            EditorBuildSettings.scenes = new[]
            {
                new EditorBuildSettingsScene(GeneratedScenePath, true),
            };
            AssetDatabase.SaveAssets();
            Debug.Log(
                $"Built provisional Stage 4 generated world: {GeneratedScenePath}; " +
                $"buildings={geometry.Buildings.Length}; " +
                $"vegetation={geometry.VegetationProxies.Length}. " +
                "All vertical dimensions remain explicit assumptions.");
        }

        private static void CreateBuilding(
            Stage4BuildingGeometry building, float groundY, Transform parent,
            string meshFolder, Material wallMaterial, Material roofMaterial)
        {
            Vector2[] footprint = building.FootprintXZ;
            int count = footprint.Length;
            var vertices = new Vector3[count * 2];
            float bottom = groundY + 0.03f;
            float top = bottom + building.HeightMetres;
            for (int index = 0; index < count; index++)
            {
                vertices[index] = new Vector3(
                    footprint[index].x, bottom, footprint[index].y);
                vertices[index + count] = new Vector3(
                    footprint[index].x, top, footprint[index].y);
            }

            float signedArea = 0f;
            for (int index = 0; index < count; index++)
            {
                Vector2 current = footprint[index];
                Vector2 next = footprint[(index + 1) % count];
                signedArea += current.x * next.y - next.x * current.y;
            }
            var sides = new List<int>(count * 6);
            for (int index = 0; index < count; index++)
            {
                int next = (index + 1) % count;
                int bottomA = index;
                int bottomB = next;
                int topA = index + count;
                int topB = next + count;
                if (signedArea > 0f)
                {
                    sides.AddRange(new[]
                    {
                        bottomA, topA, bottomB,
                        bottomB, topA, topB,
                    });
                }
                else
                {
                    sides.AddRange(new[]
                    {
                        bottomA, bottomB, topA,
                        bottomB, topB, topA,
                    });
                }
            }

            var roof = new List<int>(building.RoofTriangles.Length);
            for (int index = 0; index < building.RoofTriangles.Length; index += 3)
            {
                int a = building.RoofTriangles[index] + count;
                int b = building.RoofTriangles[index + 1] + count;
                int c = building.RoofTriangles[index + 2] + count;
                if (Vector3.Cross(vertices[b] - vertices[a],
                                  vertices[c] - vertices[a]).y < 0f)
                    (b, c) = (c, b);
                roof.AddRange(new[] { a, b, c });
            }

            var mesh = new Mesh
            {
                name = $"{building.Id}_ProvisionalMesh",
                indexFormat = vertices.Length > 65535
                    ? UnityEngine.Rendering.IndexFormat.UInt32
                    : UnityEngine.Rendering.IndexFormat.UInt16,
                vertices = vertices,
                subMeshCount = 2,
            };
            mesh.SetTriangles(sides, 0);
            mesh.SetTriangles(roof, 1);
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();
            string meshPath = $"{meshFolder}/{building.Id}.asset";
            AssetDatabase.CreateAsset(mesh, meshPath);

            GameObject gameObject = new GameObject(
                $"{building.Id} - {building.HeightMetres:F1}m ASSUMED");
            gameObject.transform.SetParent(parent, false);
            MeshFilter filter = gameObject.AddComponent<MeshFilter>();
            MeshRenderer renderer = gameObject.AddComponent<MeshRenderer>();
            MeshCollider collider = gameObject.AddComponent<MeshCollider>();
            filter.sharedMesh = mesh;
            renderer.sharedMaterials = new[] { wallMaterial, roofMaterial };
            collider.sharedMesh = mesh;
            gameObject.isStatic = true;
        }

        private static void CreateVegetation(
            Stage4VegetationProxy vegetation, float groundY, Transform parent,
            Material trunkMaterial, Material canopyMaterial)
        {
            GameObject root = new GameObject(
                $"{vegetation.Id} - PROVISIONAL");
            root.transform.SetParent(parent, false);
            root.transform.position = new Vector3(vegetation.X, groundY, vegetation.Z);
            float trunkHeight = Mathf.Max(1f, vegetation.HeightMetres * 0.55f);
            float trunkRadius = Mathf.Max(0.12f, vegetation.CanopyRadiusMetres * 0.13f);
            GameObject trunk = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            trunk.name = "Assumed Trunk";
            trunk.transform.SetParent(root.transform, false);
            trunk.transform.localPosition = new Vector3(0f, trunkHeight * 0.5f, 0f);
            trunk.transform.localScale = new Vector3(
                trunkRadius * 2f, trunkHeight * 0.5f, trunkRadius * 2f);
            trunk.GetComponent<MeshRenderer>().sharedMaterial = trunkMaterial;
            GameObject canopy = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            canopy.name = "Assumed Canopy";
            canopy.transform.SetParent(root.transform, false);
            canopy.transform.localPosition = new Vector3(
                0f, Mathf.Max(trunkHeight, vegetation.HeightMetres * 0.72f), 0f);
            canopy.transform.localScale = Vector3.one *
                (vegetation.CanopyRadiusMetres * 2f);
            canopy.GetComponent<MeshRenderer>().sharedMaterial = canopyMaterial;
            root.isStatic = true;
        }

        private static Material GetOrCreateMaterial(
            string assetPath, Color color, float smoothness)
        {
            Shader shader = Shader.Find("Universal Render Pipeline/Lit") ??
                            Shader.Find("Standard") ??
                            Shader.Find("Unlit/Color");
            if (shader == null)
                throw new InvalidDataException(
                    "No supported shader is available for generated geometry.");
            Material material = AssetDatabase.LoadAssetAtPath<Material>(assetPath);
            if (material == null)
            {
                material = new Material(shader);
                AssetDatabase.CreateAsset(material, assetPath);
            }
            else
            {
                material.shader = shader;
            }
            material.color = color;
            if (material.HasProperty("_Smoothness"))
                material.SetFloat("_Smoothness", smoothness);
            if (material.HasProperty("_Glossiness"))
                material.SetFloat("_Glossiness", smoothness);
            EditorUtility.SetDirty(material);
            return material;
        }

        private static void CreatePreviewLighting(Transform parent)
        {
            foreach (Light light in UnityEngine.Object.FindObjectsByType<Light>())
                light.enabled = false;
            GameObject lightObject = new GameObject("Generated World Sun");
            lightObject.transform.SetParent(parent, false);
            lightObject.transform.rotation = Quaternion.Euler(48f, -32f, 0f);
            Light sun = lightObject.AddComponent<Light>();
            sun.type = LightType.Directional;
            sun.intensity = 1.15f;
            sun.shadows = LightShadows.Soft;
        }

        private static void CreatePerspectiveCamera(
            Transform parent, Stage2CalibrationAsset calibration)
        {
            foreach (Camera existingCamera in
                     UnityEngine.Object.FindObjectsByType<Camera>())
            {
                existingCamera.enabled = false;
                if (existingCamera.CompareTag("MainCamera"))
                    existingCamera.tag = "Untagged";
            }
            Vector2 minimum = calibration.WorldBoundsMin;
            Vector2 maximum = calibration.WorldBoundsMax;
            Vector2 centre = (minimum + maximum) * 0.5f;
            float size = Mathf.Max(maximum.x - minimum.x, maximum.y - minimum.y);
            size = Mathf.Max(size, 10f);
            GameObject cameraObject = new GameObject("Generated World Perspective Camera");
            cameraObject.transform.SetParent(parent, false);
            cameraObject.transform.position = new Vector3(
                centre.x, calibration.GroundYMetres + size * 0.72f,
                centre.y - size * 0.62f);
            cameraObject.transform.LookAt(new Vector3(
                centre.x, calibration.GroundYMetres + size * 0.08f, centre.y));
            Camera camera = cameraObject.AddComponent<Camera>();
            camera.fieldOfView = 52f;
            camera.nearClipPlane = 0.05f;
            camera.farClipPlane = size * 8f;
            camera.clearFlags = CameraClearFlags.Skybox;
            camera.tag = "MainCamera";
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
