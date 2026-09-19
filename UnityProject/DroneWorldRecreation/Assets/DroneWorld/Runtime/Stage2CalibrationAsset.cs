using System;
using DroneWorld.Stage1;
using UnityEngine;

namespace DroneWorld.Stage2
{
    /// <summary>
    /// Validated mapping from Stage 1 reference-image pixels to metric Unity X/Z.
    /// One Unity unit is one metre.
    /// </summary>
    public sealed class Stage2CalibrationAsset : ScriptableObject
    {
        public string SchemaVersion;
        public Stage1ArtifactManifest SourceManifest;
        public string SourceCalibrationPath;
        public string SourceReportSha256;
        public string SourceGroundSha256;
        public string Confidence;
        public bool IsApproximateScale;
        public float AssumedWorldWidthMetres;
        public float GroundYMetres;
        public float ReprojectionRmseMetres;
        public float ReprojectionMaxErrorMetres;
        public float ImageHullCoveragePercent;
        public float[] ReferencePixelToWorldXZ = new float[9];
        public Vector2[] ImageCornersWorldXZ = new Vector2[4];
        public Vector2 WorldBoundsMin;
        public Vector2 WorldBoundsMax;

        public bool IsUsable =>
            string.Equals(SchemaVersion, "stage2-calibration-v1",
                          StringComparison.Ordinal) &&
            SourceManifest != null &&
            ReferencePixelToWorldXZ != null &&
            ReferencePixelToWorldXZ.Length == 9;

        public Vector3 ReferencePixelToUnity(float u, float v)
        {
            if (!IsUsable)
                throw new InvalidOperationException("Stage 2 calibration is incomplete.");
            float[] h = ReferencePixelToWorldXZ;
            float denominator = h[6] * u + h[7] * v + h[8];
            if (!float.IsFinite(denominator) || Mathf.Abs(denominator) < 1e-8f)
                throw new InvalidOperationException(
                    "Pixel maps to an invalid projective coordinate.");
            float x = (h[0] * u + h[1] * v + h[2]) / denominator;
            float z = (h[3] * u + h[4] * v + h[5]) / denominator;
            return new Vector3(x, GroundYMetres, z);
        }
    }
}
