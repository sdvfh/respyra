import pandas as pd
from scikitplot.helpers import binary_ks_curve
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from utils import path

df = pd.read_parquet(path["data"] / "full_with_suggestions_and_scores.snappy.parquet")

original_targets = [
    "Anxious_Mood",
    "Autonomic_symptoms",
    "Cardiovascular_symptoms",
    "Catatonic_behavior",
    "Decreased_energy_tiredness_fatigue",
    "Depressed_Mood",
    "Gastrointestinal_symptoms",
    "Genitourinary_symptoms",
    "Hyperactivity_agitation",
    "Impulsivity",
    "Inattention",
    "Indecisiveness",
    "Respiratory_symptoms",
    "Suicidal_ideas",
    "Worthlessness_and_guilty",
    "avoidance_of_stimuli",
    "compensatory_behaviors_to_prevent_weight_gain",
    "compulsions",
    "diminished_emotional_expression",
    "do_things_easily_get_painful_consequences",
    "drastical_shift_in_mood_and_energy",
    "fear_about_social_situations",
    "fear_of_gaining_weight",
    "fears_of_being_negatively_evaluated",
    "flight_of_ideas",
    "intrusion_symptoms",
    "loss_of_interest_or_motivation",
    "more_talktive",
    "obsession",
    "panic_fear",
    "pessimism",
    "poor_memory",
    "sleep_disturbance",
    "somatic_muscle",
    "somatic_symptoms_others",
    "somatic_symptoms_sensory",
    "weight_and_appetite_change",
    "Anger_Irritability",
]

created_targets = [
    "anxious mood",
    "autonomic symptoms",
    "cardiovascular symptoms",
    "catatonic behavior",
    "decreased energy tiredness fatigue",
    "depressed mood",
    "gastrointestinal symptoms",
    "genitourinary symptoms",
    "hyperactivity agitation",
    "impulsivity",
    "inattention",
    "indecisiveness",
    "respiratory symptoms",
    "suicidal ideas",
    "worthlessness and guilty",
    "avoidance of stimuli",
    "compensatory behaviors to prevent weight gain",
    "compulsions",
    "diminished emotional expression",
    "do things easily get painful consequences",
    "drastical shift in mood and energy",
    "fear about social situations",
    "fear of gaining weight",
    "fears of being negatively evaluated",
    "flight of ideas",
    "intrusion symptoms",
    "loss of interest or motivation",
    "more talktive",
    "obsession",
    "panic fear",
    "pessimism",
    "poor memory",
    "sleep disturbance",
    "somatic muscle",
    "somatic symptoms others",
    "somatic symptoms sensory",
    "weight and appetite change",
    "anger irritability",
]

scores_types = ["completion", "sentence"]

df[original_targets] = df[original_targets].replace({-1: 0})
threshold = 0.5

created_targets_completion = [
    f"score_completion_{target}" for target in created_targets
]
created_targets_sentence = [f"score_sentence_{target}" for target in created_targets]
created_targets_ensemble = [f"score_ensemble_{target}" for target in created_targets]

for ensemble, completion, sentence in zip(
    created_targets_ensemble,
    created_targets_completion,
    created_targets_sentence,
    strict=True,
):
    df[ensemble] = df[[completion, sentence]].mean(axis=1)

other_feats = ["sarcasm", "ambiguity"]

for other_feat in other_feats:
    df[f"score_ensemble_{other_feat}"] = df[
        [f"score_completion_{other_feat}", f"score_sentence_{other_feat}"]
    ].mean(axis=1)

test = df[df["type"] == "test"]

f1_scores = {
    "completion": f1_score(
        test[original_targets], test[created_targets_completion] >= 0.5, average="micro"
    ),
    "sentence": f1_score(
        test[original_targets], test[created_targets_sentence] >= 0.5, average="micro"
    ),
    "ensemble": f1_score(
        test[original_targets], test[created_targets_ensemble] >= 0.5, average="micro"
    ),
}

f1_scores_individual = {
    "completion": f1_score(
        test[original_targets], test[created_targets_completion] >= 0.5, average=None
    ),
    "sentence": f1_score(
        test[original_targets], test[created_targets_sentence] >= 0.5, average="micro"
    ),
    "ensemble": f1_score(
        test[original_targets], test[created_targets_ensemble] >= 0.5, average="micro"
    ),
}

auc_scores = {
    "completion": roc_auc_score(
        test[original_targets], test[created_targets_completion], average="micro"
    ),
    "sentence": roc_auc_score(
        test[original_targets], test[created_targets_sentence], average="micro"
    ),
    "ensemble": roc_auc_score(
        test[original_targets], test[created_targets_ensemble], average="micro"
    ),
}
print("f1_score", f1_scores)
print("auc_score", auc_scores)

for other_feat, columns in [
    ["completion", created_targets_completion],
    ["sentence", created_targets_sentence],
    ["ensemble", created_targets_ensemble],
]:
    for column in columns:
        test[column] -= (
            test[f"score_{other_feat}_sarcasm"] * 0.25
            + test[f"score_{other_feat}_ambiguity"] * 0.25
        )

f1_scores = {
    "completion": f1_score(
        test[original_targets], test[created_targets_completion] >= 0.5, average="micro"
    ),
    "sentence": f1_score(
        test[original_targets], test[created_targets_sentence] >= 0.5, average="micro"
    ),
    "ensemble": f1_score(
        test[original_targets], test[created_targets_ensemble] >= 0.5, average="micro"
    ),
}

auc_scores = {
    "completion": roc_auc_score(
        test[original_targets], test[created_targets_completion], average="micro"
    ),
    "sentence": roc_auc_score(
        test[original_targets], test[created_targets_sentence], average="micro"
    ),
    "ensemble": roc_auc_score(
        test[original_targets], test[created_targets_ensemble], average="micro"
    ),
}
print("f1_score", f1_scores)
print("auc_score", auc_scores)

scores = {"completion": {}, "sentence": {}}
df_final = []
for score_type in scores_types:
    for original_target, created_target in zip(
        original_targets, created_targets, strict=True
    ):
        y_true = df[original_target]
        y_pred = df[f"score_{score_type}_{created_target}"]
        y_pred_round = df[f"score_{score_type}_{created_target}_round"]

        line = pd.Series(
            {
                "accuracy": balanced_accuracy_score(y_true, y_pred_round),
                "precision": precision_score(y_true, y_pred_round),
                "recall": recall_score(y_true, y_pred_round),
                "f1": f1_score(y_true, y_pred_round),
                "kappa": cohen_kappa_score(y_true, y_pred_round),
                "roc_auc": roc_auc_score(y_true, y_pred),
                "score_type": score_type,
            },
            name=original_target,
        )

        # Compute KS curve for the current stage
        (
            _,
            _,
            _,
            line["ks"],
            line["threshold"],
            _,
        ) = binary_ks_curve(y_true, y_pred)
        df_final.append(line)

df_final = pd.DataFrame(df_final)
df_final = df_final[df_final["score_type"] == "sentence"]

# df_final.to_parquet(path["data"] / "scores_metrics.snappy.parquet")
