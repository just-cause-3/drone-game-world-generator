using System;

namespace DroneWorld.Stage2.Editor
{
    [Serializable]
    internal sealed class Stage2CalibrationData
    {
        public string schema_version;
        public int stage;
        public bool pass;
        public string[] failure_reasons;
        public string[] warnings;
        public string confidence;
        public Stage2SourceData source_stage1;
        public Stage2CoordinateData coordinate_system;
        public Stage2QualityData quality;
        public float[] reference_pixel_to_world_xz_flat;
        public Stage2CornerData[] image_corners_world_xz;
        public Stage2BoundsData world_bounds;
    }

    [Serializable]
    internal sealed class Stage2SourceData
    {
        public string bundle_path;
        public string report_sha256;
        public string ground_sha256;
        public string source_video_sha256;
        public int reference_video_frame;
        public int pixel_width;
        public int pixel_height;
    }

    [Serializable]
    internal sealed class Stage2CoordinateData
    {
        public string units;
        public string calibration_mode;
        public bool metric_scale_is_assumed;
        public float assumed_world_width_m;
        public float ground_y_m;
        public string origin_description;
        public string x_axis_description;
        public string z_axis_description;
    }

    [Serializable]
    internal sealed class Stage2QualityData
    {
        public int point_count;
        public int independent_residual_degrees_of_freedom;
        public float image_convex_hull_area_pct;
        public float reprojection_rmse_m;
        public float reprojection_max_error_m;
        public float max_local_anisotropy;
    }

    [Serializable]
    internal sealed class Stage2CornerData
    {
        public float u;
        public float v;
        public float x;
        public float z;
    }

    [Serializable]
    internal sealed class Stage2BoundsData
    {
        public float min_x;
        public float max_x;
        public float min_z;
        public float max_z;
    }

    [Serializable]
    internal sealed class Stage2ControlPointFileData
    {
        public string schema_version = "stage2-control-points-v1";
        public string units = "meters";
        public string calibration_mode = "measured_control_points";
        public float assumed_world_width_m;
        public string origin_description;
        public string x_axis_description;
        public string z_axis_description;
        public float ground_y_m;
        public Stage2ControlPointData[] points;
    }

    [Serializable]
    internal sealed class Stage2ControlPointData
    {
        public string id;
        public Stage2ImagePointData image;
        public Stage2WorldPointData world;
    }

    [Serializable]
    internal sealed class Stage2ImagePointData
    {
        public float u;
        public float v;
    }

    [Serializable]
    internal sealed class Stage2WorldPointData
    {
        public float x;
        public float z;
    }
}
