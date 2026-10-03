using System;

namespace DroneWorld.Stage3.Editor
{
    [Serializable]
    internal sealed class Stage3Stage1ReportData
    {
        public Stage3Stage1OutputData outputs;
    }

    [Serializable]
    internal sealed class Stage3Stage1OutputData
    {
        public string video_cfr;
        public string frame_transforms;
    }

    [Serializable]
    internal sealed class Stage3ReportData
    {
        public string schema_version;
        public int stage;
        public bool pass;
        public string[] failure_reasons;
        public string[] warnings;
        public Stage3SourceStage1Data source_stage1;
        public Stage3SourceStage2Data source_stage2;
        public Stage3TimelineData timeline;
        public Stage3OutputData outputs;
    }

    [Serializable]
    internal sealed class Stage3SourceStage1Data
    {
        public string report_sha256;
        public string cfr_video_sha256;
        public string frame_transforms_sha256;
        public string source_video_sha256;
    }

    [Serializable]
    internal sealed class Stage3SourceStage2Data
    {
        public string calibration_sha256;
        public string confidence;
        public bool metric_scale_is_assumed;
    }

    [Serializable]
    internal sealed class Stage3TimelineData
    {
        public float fps;
        public int frame_count;
        public float duration_s;
    }

    [Serializable]
    internal sealed class Stage3OutputData
    {
        public string report;
        public string detections;
        public string tracks;
        public string tracking_preview;
        public string track_map;
        public string frame_counts;
    }

    [Serializable]
    internal sealed class Stage3TracksData
    {
        public string schema_version;
        public float fps;
        public int frame_count;
        public float duration_s;
        public float ground_y;
        public string scale_confidence;
        public Stage3TrackData[] tracks;
    }

    [Serializable]
    internal sealed class Stage3TrackData
    {
        public int id;
        public string kind;
        public int first_frame;
        public int last_frame;
        public int observed_frames;
        public float mean_confidence;
        public Stage3SampleData[] samples;
    }

    [Serializable]
    internal sealed class Stage3SampleData
    {
        public int frame;
        public float t;
        public float x;
        public float z;
        public float heading_deg;
        public float speed_units_per_s;
        public bool observed;
        public float confidence;
    }
}
