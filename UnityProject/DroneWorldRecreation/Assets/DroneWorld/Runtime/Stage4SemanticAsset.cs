using System;
using DroneWorld.Stage1;
using DroneWorld.Stage2;
using UnityEngine;

namespace DroneWorld.Stage4
{
    /// <summary>
    /// Imported Stage 4A semantic proposal. A structural pass is not operator acceptance.
    /// </summary>
    public sealed class Stage4SemanticAsset : ScriptableObject
    {
        public string SchemaVersion;
        public Stage1ArtifactManifest SourceManifest;
        public Stage2CalibrationAsset SourceCalibration;
        public string SourceBundlePath;
        public string SourceReportSha256;
        public string SourceLabelsSha256;
        public string SourceOverlaySha256;
        public int PixelWidth;
        public int PixelHeight;
        public float KnownValidAreaPercent;
        public float BuildingAreaPercent;
        public string ProviderImplementation;
        public string ProviderDomain;
        public bool RequiresHumanReview = true;
        public bool AcceptedForGeometry;
        public string AcceptanceRecordPath;
        public string SemanticLabelsAssetPath;
        public string SemanticOverlayAssetPath;

        public bool IsUsable =>
            string.Equals(SchemaVersion, "stage4-semantics-v1",
                          StringComparison.Ordinal) &&
            SourceManifest != null &&
            SourceCalibration != null &&
            PixelWidth == SourceManifest.PixelWidth &&
            PixelHeight == SourceManifest.PixelHeight &&
            !string.IsNullOrWhiteSpace(SourceReportSha256) &&
            !string.IsNullOrWhiteSpace(SourceLabelsSha256);
    }
}
