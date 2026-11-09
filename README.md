# RSNA Knee Abnormality Detection 2026

Detecting twelve clinically important abnormalities on knee MRI, where **only 1.3% of
training studies have expert labels**.

**Final: 0.898 macro AUC** on the private leaderboard (0.919 public), from a
10-model ensemble trained entirely on Kaggle's free tier.

| | |
|---|---|
| Competition | [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) |
| Private LB | **0.898** macro AUC |
| Public LB | 0.919 |
| Approach | LLM-extracted labels → geometry-aware DICOM cache → 2.5D CNN + GRU + attention |
| Compute | Kaggle free tier, ~140 GPU-hours |
| Entry | Solo |

---

## The problem

Twelve binary findings per study — ACL, MCL, medial and lateral meniscus, three
compartments of osteoarthritis, effusion, synovitis, Baker's cyst, contusion, fracture.
Scored on **macro-averaged AUC**, so a rare fracture counts as much as a common effusion.

Two things make this dataset unusual.

**Only 58 of 4,407 training studies carry expert labels.** The other 4,349 have a free-text
radiology report, written in one of nine languages, and nothing else. The reports are
therefore the labels — for every competitor. This is a weakly supervised learning problem
wearing an imaging costume.

**There are no reports at test time.** `test.csv` contains study identifiers only. The
competition is advertised as multimodal, but at inference the model sees pixels and nothing
else; reports exist purely to manufacture training labels. Any architecture that consumes
text at prediction time cannot be submitted.

---

## Approach

```
PHASE 1   4,407 reports  ------------->  pseudo-labels
          Qwen2.5-14B, run locally on Kaggle          [notebooks 02, 03, 27-29]

PHASE 2   24,371 DICOM series  ------->  fixed-shape pixel cache
          CPU only, zero GPU cost                      [notebooks 04, 12, 13]

PHASE 3   cache + pseudo-labels  ----->  models  ---->  submission.csv
          2.5D CNN + GRU + attention pooling           [notebooks 30-32]
```

### Phase 1 — reports into labels

Keyword matching cannot work: a radiologist describes the anatomy whether or not it is
damaged. `normal` appears in 47% of reports, `no` in 43%, `intact` in 24%. Presence of the
word "ACL" carries almost no signal; **polarity carries all of it**.

Two prompts are used because they fail in opposite directions — a categorical
present/absent/not-mentioned prompt wins on ligaments (a tear either exists or it doesn't),
a graded 0–100 prompt wins on osteoarthritis (which genuinely comes in degrees). Blending
with per-label weights beat either alone.

The model runs **locally on Kaggle**, not through an API. Competition rule 2.4.b forbids
transmitting competition data to non-participants; rule 2.6 explicitly permits publicly
available pretrained models. Open weights on Kaggle hardware satisfies both.

Final extraction quality: **0.8775 macro AUC** against the 58 expert labels, up from 0.838
with a 7B model.

### Phase 2 — the pixel cache

Decoding DICOMs inside a training loop leaves the GPU idle while the CPU decompresses
JPEG2000. Everything is decoded **once**, on CPU, into a fixed-shape uint8 array — which
costs no GPU quota at all, decisive on a 30-hour weekly budget.

Four things the decoder gets right ([`src/decode.py`](src/decode.py)), each of which would
silently degrade the model if wrong:

- **Geometric slice ordering.** Filenames and `InstanceNumber` are not reliably anatomical.
  The slice normal comes from the cross product of the `ImageOrientationPatient` cosines,
  and slices sort by the projection of `ImagePositionPatient` onto it.
- **Per-series percentile normalisation.** MRI has no absolute intensity scale; stored
  `WindowCenter` ranges from 148 to 1660 in this dataset. Clip to the 1st–99th percentile
  of each series instead.
- **Metadata-driven series selection.** Three per study — sagittal, coronal, axial —
  preferring fluid-sensitive then fat-suppressed sequences.
- **Laterality canonicalisation.** All knees mirrored to a right-knee convention, because
  four labels are compartment-specific and medial/lateral swap between legs. This is also
  why **horizontal-flip augmentation is disabled** — standard practice almost everywhere,
  and silently wrong here.

### Phase 3 — the model

```
INPUT   one study: 3 series x 32 slices x 224 x 224

  per slice   -> 2.5D: slices [j-1, j, j+1] as RGB channels
                   |
                 CNN encoder, shared across all 96 slices
                   |
  per series  -> bidirectional GRU over the slice axis
                   |  attention-pooled -> one vector per series
  per study   -> attention pool over sagittal / coronal / axial
                   |
OUTPUT            12 logits, BCE against soft targets
```

2.5D rather than 3D convolutions: a tear spans several slices so through-plane context
matters, but full 3D costs ~4× more and cannot use ImageNet weights, which matter enormously
at 4,407 studies. Trained on **soft targets** (the raw 0–100 extraction scores / 100, never
thresholded) — since AUC measures ranking, nothing in the pipeline needs hard labels.

Final submission: 5× EfficientNet-B0 + 5× ResNet34d at 224px, equally weighted, TTA ×3.

---

## Results

| # | Change | Public LB |
|---|---|---|
| 1 | `sample_submission` unchanged — a plumbing test | 0.500 |
| 2 | Single model, 160px, EfficientNet-B0 | 0.826 |
| 3 | **+ 5-fold ensemble** | 0.856 |
| 4 | **+ 224px, 32 slices, laterality bug fixed** | 0.878 |
| 5 | + blend of 160px and 224px families | 0.875 ✗ |
| 6 | **+ ResNet34d as a second architecture** | 0.882 |
| 7 | **+ test-time augmentation ×3** | 0.886 |
| 8 | **+ models retrained on 14B-extracted labels** | 0.918 |
| 9 | + ResNet also on 14B labels, rebalanced blend | **0.919** |

What actually moved it, in order of size:

| Change | Gain |
|---|---|
| Better labels (7B → 14B extraction) | **+0.032** |
| 5-fold ensembling | +0.030 |
| Resolution 160 → 224px | +0.022 |
| Second architecture | +0.004 |
| Test-time augmentation | +0.004 |

The unglamorous levers won. More models, more pixels, better targets.

---

## What didn't work

Roughly half the GPU time went into ideas that produced nothing. Each one closed off a
direction, which is worth as much as a gain.

**Per-label loss weighting by extraction reliability.** Weighting the loss by how accurately
each label extracted from text. Both val and gold moved less than 0.003 — inside noise.

**Tissue bounding-box cropping.** The premise was that cropping away background would raise
effective resolution. Measured gain: **1.03×**. Knee MRIs are already framed tightly on the
joint, with only ~7% background to remove.

**Cross-resolution blending.** Offline search on 4,216 out-of-fold predictions promised
+0.007; the leaderboard returned **−0.003**. The cause is instructive: out-of-fold
predictions are scored against *pseudo-labels*, the leaderboard against *radiologists*. A
blend that better reproduces your own extractor looks like progress and isn't. After this,
offline gains under +0.01 were treated as unproven.

**ConvNeXt-nano as a second backbone.** Collapsed to gold 0.55 after five GPU-hours.
Pretrained weights had loaded fine; the cause was the learning rate. ConvNeXt uses LayerNorm
and is normally fine-tuned near 1e-4, while the OneCycle schedule peaked at 3e-4 and
destroyed the pretrained features in epoch one. Every subsequent training notebook gained a
**one-epoch sanity gate** that aborts below 0.65 validation AUC — it would have caught this
in 20 minutes.

**288px resolution, twice.**

| Backbone | 224px singles | 288px singles |
|---|---|---|
| EfficientNet-B0 | 0.815 | 0.805 |
| ResNet34d | 0.850 | 0.832 |

The first failure was explained by EfficientNet's compound scaling — B0's native input is
exactly 224. That hypothesis predicted ResNet would be fine, since it has no prescribed
input size. It wasn't. Because **both** architectures peak at 224, the ceiling is the
**data**, not the model: at 224px the network already resolves everything these images
contain.

---

## The bug worth reading about

`Lateral Meniscus` scored **0.577** on the gold holdout while every other label sat between
0.77 and 0.95.

Four hypotheses were tested ([notebook 11](notebooks/11-diagnose-lateral-meniscus.ipynb)):
small-sample noise, failed extraction, model confusion between compartments, and incorrect
laterality. The cause was the last one, and it was a single line:

```python
lat = next((m[4] for m in metas if m[4]), "")   # empty when the tag is absent
if str(lat).upper().startswith("L"):            # False when empty -> NO FLIP
    vol = np.flip(vol, axis=lr_axis)
```

The `Laterality` DICOM tag is present in only **52.5%** of studies. For the rest, no
mirroring happened — leaving roughly **1,000 studies, 24% of the dataset, with medial and
lateral reversed**. The symptom was a systematic deficit on *both* lateral structures,
measured out-of-fold on 4,216 studies:

| | medial | lateral | gap |
|---|---|---|---|
| Meniscus | 0.766 | 0.668 | 0.098 |
| Osteoarthritis | 0.837 | 0.745 | 0.092 |

The fix recovers laterality from geometry. DICOM uses LPS coordinates where +x is the
patient's left, so the sign of `ImagePositionPatient[0]` should indicate the knee. Rather
than assume this survived de-identification, it was **validated against the 2,203 studies
that do carry a tag**: left knees averaged +31.6, right knees −125.6. AUC 0.982, accuracy
**97.5%**. Applied to the untagged half, laterality became known for 100% of studies.

Honest footnote: fixing this did not produce the expected gain. The gaps narrowed only
slightly, and Lateral Meniscus stayed weak until better extraction fixed it instead
(0.761 → 0.884). The bug was real and worth correcting; it was not what held that label back.

---

## Repository layout

```
├── README.md
├── REPORT.pdf              13-page project record: dataset, every notebook,
│                           every artefact, and what changed at each step
├── notebooks/              35 notebooks, numbered in pipeline order
├── src/
│   ├── decode.py           DICOM decoder: slice ordering, normalisation, laterality
│   ├── model.py            2.5D encoder + GRU + attention pooling, TTA
│   └── prompts.py          the two extraction prompts and the blend
└── requirements.txt
```

Notebooks are numbered chronologically; each exists because the previous one raised a
question. Files suffixed `-NULL` or `-FAILED` are the experiments that didn't work — kept
deliberately, since they are half the story.

| Step | Question it answered |
|---|---|
| 01 | How many expert labels exist, and is laterality recoverable? |
| 02 → 03 | Can reports become labels? Then do it for all 4,407. |
| 04 → 06 | Decode the images, train one model, get a real score. |
| 07 | Does cropping to the joint buy resolution? (No.) |
| 08 → 10 | Does ensembling help? (Yes, +0.030.) |
| 11 → 13 | Why is Lateral Meniscus at 0.577? |
| 14 → 17 | Does 224px beat 160px? (Yes, +0.022.) |
| 18 → 19 | Does blending resolutions help? (No.) |
| 20 → 23 | Does a second architecture help? (Marginally.) |
| 24 → 25 | Does 288px beat 224px? (No.) |
| 26 | Does test-time augmentation help? (Yes.) |
| 27 → 31 | Can better extraction lift the ceiling? (Yes — the largest gain.) |
| 32 → 33 | Upgrade the second architecture; retest 288px. |

---

## Running this

The competition data is **not** included — rule 2.4.b prohibits redistributing it. Download
it from the [competition page](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
after accepting the rules.

```bash
pip install -r requirements.txt
```

Then run the notebooks in order on Kaggle, attaching each stage's output as the next
stage's input:

1. **`03-extract-all-7b`** / **`28-extract-all-14b`** → `pseudo_labels.csv` *(GPU, ~10h)*
2. **`13a`/`13b-cache224`** → the 224px pixel cache *(CPU, ~45min, 21.3 GB)*
3. **`30-retrain-b0-14b-labels`**, **`32-retrain-resnet-14b-labels`** → 10 checkpoints *(GPU, ~22h)*
4. **`31-inference-final`** → `submission.csv` *(GPU, ~6.5h)*

Two Kaggle-specific notes. A committed run may not exceed **12 hours** and a notebook's
output may not exceed **20 GB**, which is why long training runs are split across commits
and the 224px cache spans two notebooks. And Kaggle attaches only a notebook's *most recent*
version as input, so checkpoints split across commits must be consolidated into a Kaggle
Dataset before inference can see them all.

---

## Notes

- Competition data is governed by the [RSNA MIRA licence](http://rsna.org/mira-license)
  and the Kaggle competition rules. None of it is redistributed here.
- Code is MIT licensed. See [LICENSE](LICENSE).
