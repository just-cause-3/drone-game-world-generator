using UnityEngine;

namespace DroneWorld.Stage3
{
    /// <summary>
    /// Synthetic playback used to validate Stage 3 actor scale, pivots, heading,
    /// interpolation, and the Unity clock before real tracking data is available.
    /// </summary>
    public sealed class Stage3SamplePlayback : MonoBehaviour
    {
        [SerializeField] private Transform carActor;
        [SerializeField] private Transform personActor;
        [SerializeField, Min(0.1f)] private float loopDurationSeconds = 10f;
        [SerializeField, Min(0f)] private float playbackSpeed = 1f;
        [SerializeField] private bool playOnStart = true;
        [SerializeField] private bool loop = true;
        [SerializeField, Min(0.5f)] private float carRadiusX = 10f;
        [SerializeField, Min(0.5f)] private float carRadiusZ = 5f;
        [SerializeField] private Vector3 personPathCentre = new Vector3(7f, 0f, 0f);
        [SerializeField, Min(0.25f)] private float personPathRadius = 2.5f;

        private float playbackTimeSeconds;
        private bool isPlaying;

        public float PlaybackTimeSeconds => playbackTimeSeconds;
        public float LoopDurationSeconds => loopDurationSeconds;
        public bool IsPlaying => isPlaying;

        public void Configure(
            Transform car, Transform person, float worldWidth, float worldDepth)
        {
            carActor = car;
            personActor = person;
            float safeWidth = Mathf.Max(4f, worldWidth);
            float safeDepth = Mathf.Max(4f, worldDepth);
            carRadiusX = Mathf.Max(1.5f, safeWidth * 0.30f);
            carRadiusZ = Mathf.Max(1.5f, safeDepth * 0.28f);
            personPathCentre = new Vector3(safeWidth * 0.20f, 0f, 0f);
            personPathRadius = Mathf.Max(
                0.75f, Mathf.Min(safeWidth * 0.075f, safeDepth * 0.13f));
            playbackTimeSeconds = 0f;
            EvaluateCurrentPose();
        }

        private void Start()
        {
            isPlaying = playOnStart;
            EvaluateCurrentPose();
        }

        private void Update()
        {
            if (!isPlaying || loopDurationSeconds <= 0f)
                return;
            playbackTimeSeconds += Time.deltaTime * playbackSpeed;
            if (playbackTimeSeconds >= loopDurationSeconds)
            {
                if (loop)
                {
                    playbackTimeSeconds %= loopDurationSeconds;
                }
                else
                {
                    playbackTimeSeconds = loopDurationSeconds;
                    isPlaying = false;
                }
            }
            EvaluateCurrentPose();
        }

        [ContextMenu("Play")]
        public void Play()
        {
            isPlaying = true;
        }

        [ContextMenu("Pause")]
        public void Pause()
        {
            isPlaying = false;
        }

        [ContextMenu("Reset Playback")]
        public void ResetPlayback()
        {
            playbackTimeSeconds = 0f;
            EvaluateCurrentPose();
        }

        public void Seek(float timeSeconds)
        {
            if (loopDurationSeconds <= 0f)
                playbackTimeSeconds = 0f;
            else if (loop)
                playbackTimeSeconds = Mathf.Repeat(timeSeconds, loopDurationSeconds);
            else
                playbackTimeSeconds = Mathf.Clamp(timeSeconds, 0f, loopDurationSeconds);
            EvaluateCurrentPose();
        }

        private void EvaluateCurrentPose()
        {
            if (loopDurationSeconds <= 0f)
                return;
            float normalized = Mathf.Clamp01(
                playbackTimeSeconds / loopDurationSeconds);
            float angle = normalized * Mathf.PI * 2f;

            Vector3 carPosition = new Vector3(
                carRadiusX * Mathf.Cos(angle),
                0f,
                carRadiusZ * Mathf.Sin(angle));
            Vector3 carDirection = new Vector3(
                -carRadiusX * Mathf.Sin(angle),
                0f,
                carRadiusZ * Mathf.Cos(angle));
            SetActorPose(carActor, carPosition, carDirection);

            // Keep an integer number of path cycles per loop so the actor never
            // teleports when playback wraps from the last frame to the first.
            float personAngle = angle;
            Vector3 personPosition = personPathCentre + new Vector3(
                personPathRadius * Mathf.Sin(personAngle),
                0f,
                personPathRadius * 0.55f * Mathf.Sin(personAngle * 2f));
            Vector3 personDirection = new Vector3(
                personPathRadius * Mathf.Cos(personAngle),
                0f,
                personPathRadius * 1.1f * Mathf.Cos(personAngle * 2f));
            SetActorPose(personActor, personPosition, personDirection);
        }

        private static void SetActorPose(
            Transform actor, Vector3 localPosition, Vector3 localDirection)
        {
            if (actor == null)
                return;
            actor.localPosition = localPosition;
            if (localDirection.sqrMagnitude > 1e-6f)
                actor.localRotation = Quaternion.LookRotation(
                    localDirection.normalized, Vector3.up);
        }
    }
}
