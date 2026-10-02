using System;
using DroneWorld.Stage1;
using DroneWorld.Stage2;
using UnityEngine;

namespace DroneWorld.Stage3
{
    [Serializable]
    public sealed class Stage3TrackSample
    {
        public int Frame;
        public float TimeSeconds;
        public float X;
        public float Z;
        public float HeadingDegrees;
        public float SpeedUnitsPerSecond;
        public bool Observed;
        public float Confidence;
    }

    [Serializable]
    public sealed class Stage3ActorTrack
    {
        public int Id;
        public Stage3ActorKind Kind;
        public int FirstFrame;
        public int LastFrame;
        public int ObservedFrames;
        public float MeanConfidence;
        public Stage3TrackSample[] Samples = Array.Empty<Stage3TrackSample>();
    }

    /// <summary>Validated, provenance-bound Stage 3 tracks used at runtime.</summary>
    public sealed class Stage3TrackAsset : ScriptableObject
    {
        public string SchemaVersion;
        public Stage1ArtifactManifest SourceManifest;
        public Stage2CalibrationAsset SourceCalibration;
        public string SourceBundlePath;
        public string SourceReportSha256;
        public string SourceTracksSha256;
        public string SourceCfrVideoSha256;
        public string SourceFrameTransformsSha256;
        public string SourceCalibrationSha256;
        public float FramesPerSecond;
        public int FrameCount;
        public float DurationSeconds;
        public float GroundY;
        public string ScaleConfidence;
        public Stage3ActorTrack[] Tracks = Array.Empty<Stage3ActorTrack>();

        public bool IsUsable =>
            string.Equals(SchemaVersion, "stage3-tracks-v1", StringComparison.Ordinal) &&
            SourceManifest != null && SourceCalibration != null &&
            FramesPerSecond > 0f && FrameCount > 0 && DurationSeconds > 0f &&
            Tracks != null;
    }
}
