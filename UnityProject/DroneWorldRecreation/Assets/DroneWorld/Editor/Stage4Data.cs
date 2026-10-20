using System;

namespace DroneWorld.Stage4.Editor
{
    [Serializable]
    internal sealed class Stage4ReportData
    {
        public string schema_version;
        public int stage;
        public string substage;
        public bool pass;
        public bool requires_human_review;
        public bool ready_for_geometry;
        public string[] failure_reasons;
        public string[] warnings;
        public Stage4SourceStage1Data source_stage1;
        public Stage4SourceStage2Data source_stage2;
        public Stage4ProviderData semantic_provider;
        public Stage4QualityData quality;
        public Stage4OutputsData outputs;
        public Stage4ArtifactHashesData artifact_hashes;
    }

    [Serializable]
    internal sealed class Stage4SourceStage1Data
    {
        public string bundle_path;
        public string report_sha256;
        public string ground_sha256;
        public string valid_mask_sha256;
        public string source_video_sha256;
        public int pixel_width;
        public int pixel_height;
    }

    [Serializable]
    internal sealed class Stage4SourceStage2Data
    {
        public string bundle_path;
        public string calibration_sha256;
        public string confidence;
        public bool metric_scale_is_assumed;
    }

    [Serializable]
    internal sealed class Stage4ProviderData
    {
        public string implementation;
        public string domain;
        public bool automatic;
        public string model;
        public string model_sha256;
        public string source;
        public string source_sha256;
    }

    [Serializable]
    internal sealed class Stage4QualityData
    {
        public int width;
        public int height;
        public int valid_pixels;
        public float valid_area_pct;
        public float known_valid_area_pct;
        public int component_count;
        public float components_per_megapixel;
        public Stage4ClassCoverageData[] class_coverage;
    }

    [Serializable]
    internal sealed class Stage4ClassCoverageData
    {
        public int id;
        public string name;
        public int pixels;
        public float valid_area_pct;
    }

    [Serializable]
    internal sealed class Stage4OutputsData
    {
        public string report;
        public string semantic_labels;
        public string semantic_overlay;
        public string components;
        public string acceptance;
    }

    [Serializable]
    internal sealed class Stage4ArtifactHashesData
    {
        public string semantic_labels_sha256;
        public string semantic_overlay_sha256;
        public string components_sha256;
    }

    [Serializable]
    internal sealed class Stage4AcceptanceData
    {
        public string schema_version = "stage4-semantic-acceptance-v1";
        public int stage = 4;
        public bool accepted;
        public string stage4_report_sha256;
        public string semantic_labels_sha256;
        public string accepted_utc;
        public string note;
    }
}
