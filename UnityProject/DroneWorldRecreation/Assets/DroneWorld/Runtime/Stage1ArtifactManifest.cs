using UnityEngine;

namespace DroneWorld.Stage1
{
    /// <summary>
    /// Unity-side record of the validated Stage 1 bundle used to build the preview.
    /// Metric calibration deliberately remains unset until the next stage.
    /// </summary>
    public sealed class Stage1ArtifactManifest : ScriptableObject
    {
        public string SchemaVersion;
        public string SourceBundlePath;
        public string SourceVideoPath;
        public string SourceVideoSha256;
        public int PixelWidth;
        public int PixelHeight;
        public int ReferenceVideoFrame;
        public float ReferenceTimeSeconds;
        public string GroundTextureAssetPath;
        public string ValidMaskAssetPath;
        public bool HasMetricCalibration;
    }
}
