using UnityEngine;

namespace DroneWorld.Stage1
{
    /// <summary>
    /// Marks a preview surface created from a passing Stage 1 artifact bundle.
    /// Its dimensions are display units, not meters.
    /// </summary>
    public sealed class Stage1GroundView : MonoBehaviour
    {
        [SerializeField] private Stage1ArtifactManifest manifest;
        [SerializeField] private MeshRenderer groundRenderer;
        [SerializeField] private float previewWidth = 10f;

        public Stage1ArtifactManifest Manifest => manifest;
        public MeshRenderer GroundRenderer => groundRenderer;
        public float PreviewWidth => previewWidth;

        public void Configure(Stage1ArtifactManifest sourceManifest,
                              MeshRenderer renderer, float width)
        {
            manifest = sourceManifest;
            groundRenderer = renderer;
            previewWidth = width;
        }
    }
}
