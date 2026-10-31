using DroneWorld.Stage4;
using UnityEngine;

namespace DroneWorld.Stage4Geometry
{
    /// <summary>Scene marker and visibility controls for provisional generated geometry.</summary>
    public sealed class Stage4GeneratedWorld : MonoBehaviour
    {
        [SerializeField] private Stage4GeometryAsset geometryAsset;
        [SerializeField] private Transform buildingRoot;
        [SerializeField] private Transform vegetationRoot;
        [SerializeField] private Stage4SemanticReview semanticReview;

        public Stage4GeometryAsset GeometryAsset => geometryAsset;
        public Transform BuildingRoot => buildingRoot;
        public Transform VegetationRoot => vegetationRoot;

        public void Configure(Stage4GeometryAsset asset, Transform buildings,
                              Transform vegetation, Stage4SemanticReview review)
        {
            geometryAsset = asset;
            buildingRoot = buildings;
            vegetationRoot = vegetation;
            semanticReview = review;
        }

        public void SetBuildingsVisible(bool visible)
        {
            if (buildingRoot != null)
                buildingRoot.gameObject.SetActive(visible);
        }

        public void SetVegetationVisible(bool visible)
        {
            if (vegetationRoot != null)
                vegetationRoot.gameObject.SetActive(visible);
        }

        public void SetSemanticOverlayVisible(bool visible)
        {
            if (semanticReview != null)
                semanticReview.SetOverlayVisible(visible);
        }
    }
}
