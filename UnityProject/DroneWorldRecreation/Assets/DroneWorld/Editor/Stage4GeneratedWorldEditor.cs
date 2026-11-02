using DroneWorld.Stage4Geometry;
using UnityEditor;
using UnityEngine;

namespace DroneWorld.Stage4Geometry.Editor
{
    [CustomEditor(typeof(Stage4GeneratedWorld))]
    internal sealed class Stage4GeneratedWorldEditor : UnityEditor.Editor
    {
        public override void OnInspectorGUI()
        {
            Stage4GeneratedWorld world = (Stage4GeneratedWorld)target;
            Stage4GeometryAsset asset = world.GeometryAsset;
            EditorGUILayout.HelpBox(
                "Provisional generated world. Building heights, vegetation dimensions, " +
                "and approximate-scale measurements are assumptions, not recovered " +
                "physical truth.", MessageType.Warning);
            if (asset == null)
            {
                EditorGUILayout.HelpBox("Geometry asset is missing.", MessageType.Error);
                return;
            }
            EditorGUILayout.LabelField("Buildings", asset.Buildings.Length.ToString());
            EditorGUILayout.LabelField(
                "Vegetation proxies", asset.VegetationProxies.Length.ToString());
            EditorGUILayout.LabelField("Height policy", asset.HeightPolicy);
            EditorGUILayout.LabelField(
                "Median scale", $"{asset.MetresPerPixelMedian:F4} m/pixel");
            EditorGUILayout.Space(6f);
            EditorGUILayout.BeginHorizontal();
            if (GUILayout.Button("Show Buildings"))
                world.SetBuildingsVisible(true);
            if (GUILayout.Button("Hide Buildings"))
                world.SetBuildingsVisible(false);
            EditorGUILayout.EndHorizontal();
            EditorGUILayout.BeginHorizontal();
            if (GUILayout.Button("Show Vegetation"))
                world.SetVegetationVisible(true);
            if (GUILayout.Button("Hide Vegetation"))
                world.SetVegetationVisible(false);
            EditorGUILayout.EndHorizontal();
            EditorGUILayout.BeginHorizontal();
            if (GUILayout.Button("Show Semantic Overlay"))
                world.SetSemanticOverlayVisible(true);
            if (GUILayout.Button("Hide Semantic Overlay"))
                world.SetSemanticOverlayVisible(false);
            EditorGUILayout.EndHorizontal();
        }
    }
}
