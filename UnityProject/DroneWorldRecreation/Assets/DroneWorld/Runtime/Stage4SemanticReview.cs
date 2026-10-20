using UnityEngine;

namespace DroneWorld.Stage4
{
    /// <summary>Scene marker and overlay control for a Stage 4A proposal.</summary>
    public sealed class Stage4SemanticReview : MonoBehaviour
    {
        [SerializeField] private Stage4SemanticAsset semanticAsset;
        [SerializeField] private MeshRenderer overlayRenderer;

        public Stage4SemanticAsset SemanticAsset => semanticAsset;
        public MeshRenderer OverlayRenderer => overlayRenderer;

        public void Configure(Stage4SemanticAsset asset, MeshRenderer renderer)
        {
            semanticAsset = asset;
            overlayRenderer = renderer;
        }

        public void SetOverlayVisible(bool visible)
        {
            if (overlayRenderer != null)
                overlayRenderer.enabled = visible;
        }
    }
}
