using System;
using System.Collections.Generic;
using UnityEngine;

namespace DroneWorld.Stage3
{
    /// <summary>Replays validated Stage 3 tracks using one exact CFR-derived clock.</summary>
    public sealed class Stage3PlaybackController : MonoBehaviour
    {
        [SerializeField] private Stage3TrackAsset trackAsset;
        [SerializeField] private GameObject carPrefab;
        [SerializeField] private GameObject personPrefab;
        [SerializeField] private bool playOnStart = true;
        [SerializeField] private bool loop = true;
        [SerializeField, Min(0f)] private float playbackSpeed = 1f;
        [SerializeField] private bool tintEstimatedSamples;
        [SerializeField] private bool showStationaryCars;
        [SerializeField] private bool showStationaryPeople;
        [SerializeField, Min(0f)] private float stationaryMovementThreshold = 0.5f;

        private readonly List<ActorInstance> actors = new List<ActorInstance>();
        private float playbackTimeSeconds;
        private bool isPlaying;

        public Stage3TrackAsset TrackAsset => trackAsset;
        public float PlaybackTimeSeconds => playbackTimeSeconds;
        public bool IsPlaying => isPlaying;

        public void Configure(Stage3TrackAsset asset, GameObject car, GameObject person)
        {
            trackAsset = asset;
            carPrefab = car;
            personPrefab = person;
        }

        private void Start()
        {
            if (trackAsset == null || !trackAsset.IsUsable)
            {
                Debug.LogError("Stage 3 playback has no usable track asset.", this);
                enabled = false;
                return;
            }
            BuildActors();
            playbackTimeSeconds = 0f;
            isPlaying = playOnStart;
            Evaluate();
        }

        private void Update()
        {
            if (!isPlaying || trackAsset == null)
                return;
            playbackTimeSeconds += Time.deltaTime * playbackSpeed;
            if (playbackTimeSeconds >= trackAsset.DurationSeconds)
            {
                if (loop)
                    playbackTimeSeconds = Mathf.Repeat(
                        playbackTimeSeconds, trackAsset.DurationSeconds);
                else
                {
                    playbackTimeSeconds = trackAsset.DurationSeconds;
                    isPlaying = false;
                }
            }
            Evaluate();
        }

        public void Play() => isPlaying = true;
        public void Pause() => isPlaying = false;

        public void Seek(float timeSeconds)
        {
            if (trackAsset == null)
                return;
            playbackTimeSeconds = loop
                ? Mathf.Repeat(timeSeconds, trackAsset.DurationSeconds)
                : Mathf.Clamp(timeSeconds, 0f, trackAsset.DurationSeconds);
            Evaluate();
        }

        public void ResetPlayback()
        {
            playbackTimeSeconds = 0f;
            Evaluate();
        }

        private void BuildActors()
        {
            foreach (ActorInstance actor in actors)
            {
                if (actor.Transform != null)
                    Destroy(actor.Transform.gameObject);
            }
            actors.Clear();
            foreach (Stage3ActorTrack track in trackAsset.Tracks)
            {
                bool stationary = IsStationary(track);
                if (stationary &&
                    ((track.Kind == Stage3ActorKind.Car && !showStationaryCars) ||
                     (track.Kind == Stage3ActorKind.Person && !showStationaryPeople)))
                    continue;
                GameObject prefab = track.Kind == Stage3ActorKind.Car
                    ? carPrefab : personPrefab;
                if (prefab == null)
                {
                    Debug.LogError($"No prefab configured for Stage 3 {track.Kind}.", this);
                    continue;
                }
                GameObject instance = Instantiate(prefab, transform);
                instance.name = $"{track.Kind} Track {track.Id}";
                actors.Add(new ActorInstance(
                    track, instance.transform,
                    instance.GetComponentsInChildren<Renderer>(true)));
            }
        }

        public void RebuildVisibleActors()
        {
            if (!Application.isPlaying || trackAsset == null)
                return;
            BuildActors();
            Evaluate();
        }

        private bool IsStationary(Stage3ActorTrack track)
        {
            Stage3TrackSample[] samples = track.Samples;
            if (samples == null || samples.Length < 2)
                return true;
            Vector2 start = new Vector2(samples[0].X, samples[0].Z);
            float thresholdSquared =
                stationaryMovementThreshold * stationaryMovementThreshold;
            foreach (Stage3TrackSample sample in samples)
            {
                Vector2 position = new Vector2(sample.X, sample.Z);
                if ((position - start).sqrMagnitude >= thresholdSquared)
                    return false;
            }
            return true;
        }

        private void Evaluate()
        {
            foreach (ActorInstance actor in actors)
                EvaluateActor(actor, playbackTimeSeconds);
        }

        private void EvaluateActor(ActorInstance actor, float timeSeconds)
        {
            Stage3TrackSample[] samples = actor.Track.Samples;
            float frameDuration = trackAsset.FramesPerSecond > 0f
                ? 1f / trackAsset.FramesPerSecond
                : 0f;
            if (samples == null || samples.Length == 0 ||
                timeSeconds < samples[0].TimeSeconds ||
                timeSeconds >= samples[samples.Length - 1].TimeSeconds + frameDuration)
            {
                actor.Transform.gameObject.SetActive(false);
                return;
            }
            actor.Transform.gameObject.SetActive(true);
            float evaluationTime = Mathf.Min(
                timeSeconds, samples[samples.Length - 1].TimeSeconds);
            int upper = UpperBound(samples, evaluationTime);
            int left = Mathf.Clamp(upper - 1, 0, samples.Length - 1);
            int right = Mathf.Clamp(upper, 0, samples.Length - 1);
            Stage3TrackSample first = samples[left];
            Stage3TrackSample second = samples[right];
            // Python explicitly omits gaps beyond its interpolation budget.
            // Preserve the first sample for its own video frame, then hide the
            // actor until the next supported sample instead of inventing motion.
            if (second.Frame - first.Frame > 1)
            {
                if (evaluationTime >= first.TimeSeconds + frameDuration)
                {
                    actor.Transform.gameObject.SetActive(false);
                    return;
                }
                second = first;
            }
            float interval = second.TimeSeconds - first.TimeSeconds;
            float alpha = interval > 1e-6f
                ? Mathf.Clamp01((evaluationTime - first.TimeSeconds) / interval)
                : 0f;
            actor.Transform.position = new Vector3(
                Mathf.Lerp(first.X, second.X, alpha),
                trackAsset.GroundY,
                Mathf.Lerp(first.Z, second.Z, alpha));
            float heading = Mathf.LerpAngle(
                first.HeadingDegrees, second.HeadingDegrees, alpha);
            actor.Transform.rotation = Quaternion.Euler(0f, heading, 0f);
            if (tintEstimatedSamples)
            {
                bool estimated = !first.Observed || !second.Observed;
                Color color = estimated
                    ? new Color(1f, 0.25f, 0.85f, 1f)
                    : Color.white;
                foreach (Renderer renderer in actor.Renderers)
                {
                    if (renderer.material.HasProperty("_Color"))
                        renderer.material.color = color;
                }
            }
        }

        private static int UpperBound(Stage3TrackSample[] samples, float timeSeconds)
        {
            int low = 0;
            int high = samples.Length;
            while (low < high)
            {
                int middle = low + (high - low) / 2;
                if (samples[middle].TimeSeconds <= timeSeconds)
                    low = middle + 1;
                else
                    high = middle;
            }
            return low;
        }

        private sealed class ActorInstance
        {
            public readonly Stage3ActorTrack Track;
            public readonly Transform Transform;
            public readonly Renderer[] Renderers;

            public ActorInstance(
                Stage3ActorTrack track, Transform transform, Renderer[] renderers)
            {
                Track = track;
                Transform = transform;
                Renderers = renderers ?? Array.Empty<Renderer>();
            }
        }
    }
}
