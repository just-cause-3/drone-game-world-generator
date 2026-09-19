using System;

namespace DroneWorld.Stage1.Editor
{
    [Serializable]
    internal sealed class Stage1ReportData
    {
        public string schema_version;
        public int stage;
        public bool pass;
        public string[] failure_reasons;
        public Stage1InputData input;
        public Stage1ReferenceData reference;
        public Stage1OutputData outputs;
    }

    [Serializable]
    internal sealed class Stage1InputData
    {
        public string path;
        public string sha256;
        public int width;
        public int height;
        public float fps;
        public int frame_count;
    }

    [Serializable]
    internal sealed class Stage1ReferenceData
    {
        public int video_frame;
        public float time_s;
    }

    [Serializable]
    internal sealed class Stage1OutputData
    {
        public string ground;
        public string valid_mask;
        public string temporal_variation;
        public string foreground_frequency;
        public string semantic_exclusion;
        public string inpaint_mask;
        public string registration;
        public string video_cfr;
        public string frame_transforms;
    }
}
