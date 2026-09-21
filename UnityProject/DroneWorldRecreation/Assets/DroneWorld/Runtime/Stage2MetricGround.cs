using DroneWorld.Stage1;
using UnityEngine;

namespace DroneWorld.Stage2
{
    /// <summary>Marks the metrically calibrated Stage 2 ground surface.</summary>
    public sealed class Stage2MetricGround : MonoBehaviour
    {
        [SerializeField] private Stage1ArtifactManifest stage1Manifest;
        [SerializeField] private Stage2CalibrationAsset calibration;
        [SerializeField] private MeshRenderer groundRenderer;
        [SerializeField] private MeshCollider groundCollider;

        public Stage1ArtifactManifest Stage1Manifest => stage1Manifest;
        public Stage2CalibrationAsset Calibration => calibration;
        public MeshRenderer GroundRenderer => groundRenderer;
        public MeshCollider GroundCollider => groundCollider;

        public void Configure(Stage1ArtifactManifest sourceManifest,
                              Stage2CalibrationAsset sourceCalibration,
                              MeshRenderer renderer,
                              MeshCollider collider)
        {
            stage1Manifest = sourceManifest;
            calibration = sourceCalibration;
            groundRenderer = renderer;
            groundCollider = collider;
        }
    }
}

