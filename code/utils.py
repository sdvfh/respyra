import os
from pathlib import Path

openai_api_key = os.getenv("OPENAI_API_KEY")

path = {"root": Path(__file__).parent.parent.absolute()}
path["data"] = path["root"] / "data"
path["psysym"] = path["data"] / "PsySym"
path["output"] = path["root"] / "output"
path["figures"] = path["output"] / "figures"

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
