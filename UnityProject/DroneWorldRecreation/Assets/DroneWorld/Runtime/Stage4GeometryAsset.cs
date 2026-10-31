using System;
using DroneWorld.Stage4;
using UnityEngine;

namespace DroneWorld.Stage4Geometry
{
    [Serializable]
    public sealed class Stage4BuildingGeometry
    {
        public string Id;
        public float AreaSquareMetres;
        public float HeightMetres;
        public string HeightSource;
        public bool HeightIsAssumed;
        public Vector2[] FootprintXZ = Array.Empty<Vector2>();
        public int[] RoofTriangles = Array.Empty<int>();
    }

    [Serializable]
    public sealed class Stage4VegetationProxy
    {
        public string Id;
        public float X;
        public float Z;
        public float CanopyRadiusMetres;
        public float HeightMetres;
        public bool DimensionsAreAssumed;
    }

    /// <summary>
    /// Hash-bound provisional geometry generated from an operator-accepted Stage 4A map.
    /// Heights remain assumptions unless a future evidence source replaces them.
    /// </summary>
    public sealed class Stage4GeometryAsset : ScriptableObject
    {
        public string SchemaVersion;
        public Stage4SemanticAsset SourceSemantics;
        public string SourceBundlePath;
        public string SourceReportSha256;
        public string SourceGeometrySha256;
        public string SourceAcceptanceSha256;
        public string HeightPolicy;
        public bool AllHeightsAreAssumed = true;
        public float GroundYMetres;
        public float MetresPerPixelMedian;
        public int SurfaceRegionCount;
        public string[] Warnings = Array.Empty<string>();
        public string GeometryOverlayAssetPath;
        public Stage4BuildingGeometry[] Buildings = Array.Empty<Stage4BuildingGeometry>();
        public Stage4VegetationProxy[] VegetationProxies =
            Array.Empty<Stage4VegetationProxy>();

        public bool IsUsable =>
            string.Equals(SchemaVersion, "stage4-geometry-v1",
                          StringComparison.Ordinal) &&
            SourceSemantics != null && SourceSemantics.IsUsable &&
            !string.IsNullOrWhiteSpace(SourceReportSha256) &&
            !string.IsNullOrWhiteSpace(SourceGeometrySha256) &&
            Buildings != null && VegetationProxies != null;
    }
}
