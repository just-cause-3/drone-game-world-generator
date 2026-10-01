using UnityEngine;

namespace DroneWorld.Stage3
{
    public enum Stage3ActorKind
    {
        Car,
        Person,
    }

    /// <summary>
    /// Identifies a placeholder actor visual. Its root pivot is at ground level and
    /// local +Z is the actor's forward direction.
    /// </summary>
    public sealed class Stage3SampleActorVisual : MonoBehaviour
    {
        [SerializeField] private Stage3ActorKind kind;
        [SerializeField] private Vector3 nominalDimensions;

        public Stage3ActorKind Kind => kind;
        public Vector3 NominalDimensions => nominalDimensions;

        public void Configure(Stage3ActorKind actorKind, Vector3 dimensions)
        {
            kind = actorKind;
            nominalDimensions = dimensions;
        }
    }
}
