using System;
using System.IO;
using DroneWorld.Stage4;
using UnityEditor;
using UnityEngine;

namespace DroneWorld.Stage4.Editor
{
    [CustomEditor(typeof(Stage4SemanticReview))]
    internal sealed class Stage4SemanticReviewEditor : UnityEditor.Editor
    {
        private string acceptanceNote =
            "Reviewed semantic labels against the reference ground.";

        public override void OnInspectorGUI()
        {
            Stage4SemanticReview review = (Stage4SemanticReview)target;
            Stage4SemanticAsset asset = review.SemanticAsset;
            EditorGUILayout.HelpBox(
                "This scene is a semantic review overlay only. It contains no " +
                "generated buildings, roads, terrain, or measured heights.",
                MessageType.Info);
            if (asset == null)
            {
                EditorGUILayout.HelpBox(
                    "The review component has no Stage 4A asset.", MessageType.Error);
                return;
            }

            EditorGUILayout.LabelField("Proposal", EditorStyles.boldLabel);
            EditorGUILayout.LabelField("Provider", asset.ProviderImplementation);
            EditorGUILayout.LabelField("Training domain", asset.ProviderDomain);
            EditorGUILayout.LabelField(
                "Known valid area", $"{asset.KnownValidAreaPercent:F2}%");
            EditorGUILayout.LabelField(
                "Proposed building area", $"{asset.BuildingAreaPercent:F2}%");
            if (asset.ProviderDomain.Contains("street_view"))
            {
                EditorGUILayout.HelpBox(
                    "Domain mismatch: this bootstrap model was trained for street-view " +
                    "imagery, not aerial imagery. Inspect every region; do not accept " +
                    "the sample output merely because its structural gate passed.",
                    MessageType.Warning);
            }
            else if (asset.ProviderDomain.Contains("openearthmap"))
            {
                EditorGUILayout.HelpBox(
                    "Aerial-domain OpenEarthMap model. This is a substantially better " +
                    "proposal source, but different resolution, geography, shadows or " +
                    "synthetic footage can still produce incorrect boundaries.",
                    MessageType.Info);
            }

            EditorGUILayout.Space(6f);
            if (GUILayout.Button(
                    review.OverlayRenderer != null && review.OverlayRenderer.enabled
                        ? "Hide Semantic Overlay" : "Show Semantic Overlay"))
            {
                bool visible = review.OverlayRenderer == null ||
                               !review.OverlayRenderer.enabled;
                review.SetOverlayVisible(visible);
                EditorUtility.SetDirty(review);
            }
            if (GUILayout.Button("Reveal Full-resolution Overlay"))
            {
                string path = Path.Combine(
                    asset.SourceBundlePath, "semantic_overlay.png");
                if (File.Exists(path))
                    EditorUtility.RevealInFinder(path);
                else
                    EditorUtility.DisplayDialog(
                        "Overlay missing", $"File not found:\n{path}", "OK");
            }

            EditorGUILayout.Space(8f);
            EditorGUILayout.LabelField("Geometry acceptance", EditorStyles.boldLabel);
            EditorGUILayout.HelpBox(
                asset.AcceptedForGeometry
                    ? "ACCEPTED for the next geometry stage. Approval is bound to the " +
                      "exact report and label-image hashes."
                    : "BLOCKED. Review/correct the labels, rerun/import Stage 4A if " +
                      "needed, then explicitly accept this exact proposal.",
                asset.AcceptedForGeometry ? MessageType.Info : MessageType.Warning);
            acceptanceNote = EditorGUILayout.TextField(
                "Review note", acceptanceNote);

            if (!asset.AcceptedForGeometry)
            {
                if (GUILayout.Button(
                        "Accept Exact Semantic Map for Stage 4B",
                        GUILayout.Height(30f)) &&
                    EditorUtility.DisplayDialog(
                        "Accept exact semantic map?",
                        "This records your review of the exact semantic label bytes. " +
                        "It does not certify metric heights and does not create geometry.",
                        "Accept", "Cancel"))
                    WriteAcceptance(asset, true, acceptanceNote);
            }
            else if (GUILayout.Button("Revoke Geometry Acceptance"))
            {
                WriteAcceptance(asset, false, "Operator revoked semantic acceptance.");
            }
        }

        private static void WriteAcceptance(
            Stage4SemanticAsset asset, bool accepted, string note)
        {
            try
            {
                string reportPath = Path.Combine(
                    asset.SourceBundlePath, "report.json");
                string labelsPath = Path.Combine(
                    asset.SourceBundlePath, "semantic_labels.png");
                if (!File.Exists(reportPath) || !File.Exists(labelsPath))
                    throw new FileNotFoundException(
                        "The external Stage 4A report or label image is missing.");
                string reportHash = Stage4BundleImporter.ComputeSha256(reportPath);
                string labelsHash = Stage4BundleImporter.ComputeSha256(labelsPath);
                if (!string.Equals(reportHash, asset.SourceReportSha256,
                                   StringComparison.OrdinalIgnoreCase) ||
                    !string.Equals(labelsHash, asset.SourceLabelsSha256,
                                   StringComparison.OrdinalIgnoreCase))
                    throw new InvalidDataException(
                        "The external Stage 4A artifacts changed after import. " +
                        "Import the proposal again before accepting it.");

                var record = new Stage4AcceptanceData
                {
                    accepted = accepted,
                    stage4_report_sha256 = reportHash,
                    semantic_labels_sha256 = labelsHash,
                    accepted_utc = DateTime.UtcNow.ToString("O"),
                    note = string.IsNullOrWhiteSpace(note)
                        ? (accepted
                            ? "Operator reviewed and accepted the exact semantic proposal."
                            : "Operator revoked semantic acceptance.")
                        : note.Trim(),
                };
                string path = Path.Combine(
                    asset.SourceBundlePath, "semantic_acceptance.json");
                File.WriteAllText(path, JsonUtility.ToJson(record, true) + "\n");
                Stage4BundleImporter.RefreshAcceptanceState(asset);
                EditorUtility.SetDirty(asset);
                AssetDatabase.SaveAssets();
                Debug.Log(
                    accepted
                        ? $"Accepted exact Stage 4A semantic map: {path}"
                        : $"Revoked Stage 4A semantic acceptance: {path}");
            }
            catch (Exception exception)
            {
                Debug.LogException(exception);
                EditorUtility.DisplayDialog(
                    "Could not update acceptance", exception.Message, "OK");
            }
        }
    }
}
