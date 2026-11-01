using System;

namespace DroneWorld.Stage4Geometry.Editor
{
    [Serializable]
    internal sealed class Stage4GeometryReportData
    {
        public string schema_version;
        public int stage;
        public string substage;
        public bool pass;
        public bool ready_for_unity_geometry;
        public string[] failure_reasons;
        public string[] warnings;
        public Stage4GeometrySourceStage4Data source_stage4a;
        public Stage4GeometrySourceStage2Data source_stage2;
        public Stage4GeometryQualityData quality;
        public Stage4GeometryOutputsData outputs;
        public Stage4GeometryHashesData artifact_hashes;
    }

    [Serializable]
    internal sealed class Stage4GeometrySourceStage4Data
    {
        public string bundle_path;
        public string report_sha256;
        public string semantic_labels_sha256;
        public string acceptance_sha256;
    }

    [Serializable]
    internal sealed class Stage4GeometrySourceStage2Data
    {
        public string bundle_path;
        public string calibration_sha256;
        public string confidence;
        public bool metric_scale_is_assumed;
    }

    [Serializable]
    internal sealed class Stage4GeometryQualityData
    {
        public float metres_per_pixel_median;
        public int building_count;
        public int building_vertex_count;
        public int surface_region_count;
        public int vegetation_proxy_count;
        public int assumed_height_count;
    }

    [Serializable]
    internal sealed class Stage4GeometryOutputsData
    {
        public string report;
        public string geometry;
        public string geometry_overlay;
    }

    [Serializable]
    internal sealed class Stage4GeometryHashesData
    {
        public string geometry_sha256;
        public string geometry_overlay_sha256;
    }

    [Serializable]
    internal sealed class Stage4GeometryFileData
    {
        public string schema_version;
        public int stage;
        public string substage;
        public Stage4GeometryCoordinateData coordinate_system;
        public Stage4HeightPolicyData height_policy;
        public Stage4BuildingData[] buildings;
        public Stage4SurfaceRegionData[] surface_regions;
        public Stage4VegetationData[] vegetation_proxies;
    }

    [Serializable]
    internal sealed class Stage4GeometryCoordinateData
    {
        public string space;
        public string horizontal_units;
        public string vertical_axis;
        public float ground_y_m;
        public string scale_confidence;
        public bool metric_scale_is_assumed;
    }

    [Serializable]
    internal sealed class Stage4HeightPolicyData
    {
        public string name;
        public bool all_heights_are_assumed;
        public float default_building_height_m;
        public float floor_height_m;
        public float maximum_assumed_height_m;
    }

    [Serializable]
    internal sealed class Stage4BuildingData
    {
        public string id;
        public float area_m2;
        public float height_m;
        public string height_source;
        public bool height_is_assumed;
        public Stage4PointXZData[] footprint_xz;
        public int[] roof_triangles;
    }

    [Serializable]
    internal sealed class Stage4SurfaceRegionData
    {
        public string id;
        public int class_id;
        public string class_name;
        public float area_m2;
        public Stage4PointXZData[] boundary_xz;
    }

    [Serializable]
    internal sealed class Stage4VegetationData
    {
        public string id;
        public float x;
        public float z;
        public float canopy_radius_m;
        public float height_m;
        public bool dimensions_are_assumed;
    }

    [Serializable]
    internal sealed class Stage4PointXZData
    {
        public float x;
        public float z;
    }
}
