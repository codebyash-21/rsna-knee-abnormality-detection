"""
Report-to-label extraction prompts.

Only 58 of 4,407 training studies carry expert labels. The other 4,349 have a
radiology report and nothing else, in one of nine languages. So the reports are
the labels -- for every competitor -- and this file is where most of the
competition's signal comes from.

Why not keyword matching
------------------------
A radiologist describes the anatomy whether or not it is damaged: "the ACL is
intact", "menisci are normal". 'normal' appears in 47% of reports, 'no' in 43%,
'intact' in 24%. The word ACL appears in nearly every report. Presence carries
almost no signal; polarity carries all of it.

Why two prompts
---------------
They fail in opposite directions, measured on the 58 gold studies:

                  categorical   graded
    ACL              0.835       0.619
    MCL              0.905       0.669
    Medial OA        0.850       0.940
    PF OA            0.611       0.770
    Effusion         0.648       0.801

A ligament is torn or it is not, so grading blurs a binary judgement.
Osteoarthritis genuinely comes in degrees, so forcing yes/no discards
information. Blending with per-label weights beat either alone.

Why the model was run locally
-----------------------------
Competition rule 2.4.b forbids transmitting competition data to any party not
participating. Sending reports to a commercial LLM API is arguably such a
transmission. An open-weights model executing on Kaggle's own hardware keeps
the data on the platform; rule 2.6 explicitly permits publicly available
pretrained models.

Model size was the bottleneck
-----------------------------
Measured on the 58 gold studies before committing GPU time:

    Qwen2.5-7B  greedy               0.8382
    Qwen2.5-7B  self-consistency x3  0.8431   (+0.005)
    Qwen2.5-14B greedy               0.8712   (+0.033)   <- chosen
    Qwen2.5-14B self-consistency x3  0.8785   (+0.041, but +21 GPU-h)

The full 14B re-extraction scored 0.8775 against expert labels, up from 0.838.
Retraining on those labels was worth +0.032 on the leaderboard -- the single
largest gain in the project.
"""

MODEL = "Qwen/Qwen2.5-14B-Instruct"

SYSTEM_CATEGORICAL = """You are a musculoskeletal radiologist extracting structured findings from knee MRI reports.
The report may be in ANY language. Read it in its original language.

For each finding, decide: PRESENT, ABSENT (negated/normal/intact), or NOT_MENTIONED.

- acl: anterior cruciate ligament tear or injury
- mcl: medial collateral ligament tear or injury
- medial_meniscus: medial meniscus tear
- lateral_meniscus: lateral meniscus tear
- medial_oa: MEDIAL tibiofemoral osteoarthritis / chondral loss
- lateral_oa: LATERAL tibiofemoral osteoarthritis / chondral loss
- pf_oa: patellofemoral osteoarthritis / chondromalacia
- effusion: joint effusion
- synovitis: synovial thickening or inflammation
- bakers: Baker's / popliteal cyst
- contusion: bone contusion, bruise, marrow oedema from trauma
- fracture: fracture of any bone

RULES:
1. Negation decides. "ACL intact", "menisci normal", "no tear" => absent.
2. Degenerative signal that is NOT a tear => absent for meniscus/ligament.
3. Distinguish MEDIAL from LATERAL carefully.
4. Never guess: if not referred to, say not_mentioned.

Reply ONLY JSON:
{"acl":"present|absent|not_mentioned","mcl":"...","medial_meniscus":"...","lateral_meniscus":"...","medial_oa":"...","lateral_oa":"...","pf_oa":"...","effusion":"...","synovitis":"...","bakers":"...","contusion":"...","fracture":"..."}"""


SYSTEM_GRADED = """You are a musculoskeletal radiologist reading knee MRI reports and estimating how likely each of twelve findings is to be truly present.

The report may be in ANY language. Read it in its original language.

For EACH finding give an integer 0-100: the likelihood an expert reviewing the actual images would mark it present.

  0-10    explicitly negated ("ACL intact", "menisci normal", "no effusion")
  11-30   not mentioned at all
  31-55   equivocal, borderline, trace, minimal, degenerative change short of a tear
  56-80   clearly described but mild, or described indirectly
  81-100  explicitly and unambiguously present

THE TWELVE FINDINGS (match on MEANING, not exact wording):
acl              ACL tear, rupture, discontinuity
mcl              MCL tear, rupture, high-grade sprain
medial_meniscus  medial meniscus TEAR (degeneration alone is NOT a tear -> low)
lateral_meniscus lateral meniscus TEAR (same rule)
medial_oa        MEDIAL compartment: osteoarthritis, chondral loss/thinning, joint space
                 narrowing, subchondral sclerosis, osteophytes
lateral_oa       LATERAL compartment: same features
pf_oa            PATELLOFEMORAL (patella/trochlea): same, chondromalacia patellae,
                 retropatellar cartilage thinning
effusion         joint effusion, intra-articular fluid (trace/small counts, ~50-65)
synovitis        synovial thickening, enhancement, inflammation, plica synovitis
bakers           Baker's cyst, popliteal cyst, gastrocnemio-semimembranosus bursa cyst
contusion        bone contusion, bruise, marrow oedema from trauma
fracture         fracture, cortical break, avulsion, insufficiency/stress fracture

RULES:
- Negation is decisive: "intact", "normal", "unremarkable", "no evidence of" => under 10.
- "Tricompartmental osteoarthritis" raises ALL THREE OA findings.
- Meniscal degeneration without a tear line => 20-40.
- If never referred to, give 15-25. Never exactly 0 for silence.

Reply ONLY JSON of twelve integers:
{"acl":0,"mcl":0,"medial_meniscus":0,"lateral_meniscus":0,"medial_oa":0,"lateral_oa":0,"pf_oa":0,"effusion":0,"synovitis":0,"bakers":0,"contusion":0,"fracture":0}"""


KEYS = ["acl", "mcl", "medial_meniscus", "lateral_meniscus", "medial_oa",
        "lateral_oa", "pf_oa", "effusion", "synovitis", "bakers",
        "contusion", "fracture"]

# Map the categorical answer onto the same 0-100 scale as the graded one.
CATEGORICAL_TO_SCORE = {"present": 90.0, "not_mentioned": 20.0, "absent": 5.0}

# Weight on the CATEGORICAL pass, per label. Fitted on the 58 gold studies:
# ligaments lean categorical, osteoarthritis and effusion lean graded.
BLEND_WEIGHTS = {
    "acl": 0.80, "mcl": 0.80,
    "medial_meniscus": 0.35, "lateral_meniscus": 0.35,
    "medial_oa": 0.20, "lateral_oa": 0.20, "pf_oa": 0.20,
    "effusion": 0.20, "synovitis": 0.35,
    "bakers": 0.50, "contusion": 0.50, "fracture": 0.50,
}


def blend(cat_value, num_value, key):
    """Combine one categorical and one graded prediction into a 0-100 score.

    Train on these as SOFT targets divided by 100, never thresholded. A report
    the extractor scored 55 is genuinely ambiguous and should produce a weaker
    gradient than one scored 95. Since the metric is AUC -- pure ranking --
    nothing downstream needs hard labels.
    """
    w = BLEND_WEIGHTS[key]
    c = CATEGORICAL_TO_SCORE.get(str(cat_value).lower(), 20.0)
    try:
        n = min(max(float(num_value), 0), 100)
    except (TypeError, ValueError):
        n = 20.0
    return w * c + (1 - w) * n
