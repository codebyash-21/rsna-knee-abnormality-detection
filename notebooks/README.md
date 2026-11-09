# Notebooks

Numbered in pipeline order. Each exists because the previous one raised a question.
Outputs are cleared — the competition data cannot be redistributed (rule 2.4.b), and
several notebooks print report text or study identifiers when run.

Suffixes: `-NULL` = the experiment produced no gain, `-FAILED` = the run broke,
`-superseded` = replaced by a later version after a bug was found.

| File | Compute | Purpose |
|---|---|---|
| `01-eda` | CPU 1h | Five questions that set the strategy. Found 58 gold labels, laterality recoverable, institution stripped. |
| `02-extract-pilot` | GPU 25m | Can reports become labels? Categorical prompt on 58 studies → 0.770. |
| `02b-extract-graded-prompt` | GPU 15m | Graded 0–100 prompt → 0.794. Showed the two prompts fail oppositely. |
| `03-extract-all-7b` | GPU 4h | Both prompts over 4,407 reports, blended → `pseudo_labels.csv`, 0.838. |
| `04-cache-160px` | CPU 31m | Decode every DICOM once → `cache.npy`, 8.1 GB. |
| `05-train-160px-fold0` | GPU 50m | First model. Gold 0.803. |
| `06-inference-single` | GPU 15m | First real submission: **0.826**. |
| `07-cache-crop-NULL` | CPU 35m | Tissue bounding-box crop. Only 1.03× gain — scans already tightly framed. |
| `08-train-160px-folds` | GPU 3.5h | Folds 1–4. Ensemble gold 0.825. |
| `09a/09b-cache224-superseded` | CPU 45m | 224px cache, replaced by 13a/13b after the laterality bug surfaced. |
| `10-inference-5fold` | GPU 30m | **0.856** |
| `11-diagnose-lateral-meniscus` | CPU 5m | Four hypotheses for 0.577. Found the laterality bug. |
| `12-recover-laterality` | CPU 8m | Geometric recovery, validated at 97.5% → `laterality.csv`. |
| `13a/13b-cache224` | CPU 45m | 224px, 32 slices, laterality fixed. **Used by every final model.** |
| `14-train-224px-fold0` | GPU 4h | Gold 0.812. |
| `16-train-224px-folds` | GPU 9h | Folds 1–4. Ensemble gold 0.835. |
| `17-inference-224px` | GPU 70m | **0.878** |
| `18-blend-search-offline` | CPU 1m | Offline blend weights on out-of-fold data. Predicted +0.007. |
| `19-inference-blend-NULL` | GPU 100m | Cross-resolution blend. **0.875** — a regression. |
| `20-train-convnext-FAILED` | GPU 5h | Collapsed to 0.55 on a learning-rate mismatch. |
| `22-train-resnet34d` | GPU 10h | Second architecture, with a sanity gate added. |
| `23-inference-two-backbones` | GPU 140m | **0.882** |
| `24a/24b/24c-cache288` | CPU 60m | 288px cache, split by plane. 35.1 GB. |
| `25-train-288px-NULL` | GPU 10.5h | Worse than 224px (0.805 vs 0.815). |
| `26-inference-tta` | GPU 395m | **0.886** |
| `27-extract-pilot-14b` | GPU 50m | Four extraction configs on 58 studies. 14B greedy won. |
| `28-extract-all-14b` | GPU 10h | Re-extracted all 4,407 → 0.8775. |
| `30-retrain-b0-14b-labels` | GPU 11h | Singles 0.848, up from 0.815. |
| `31-inference-final` | GPU 390m | **0.918 → 0.919.** Both final submissions. |
| `32-retrain-resnet-14b-labels` | GPU 11h | Singles 0.850, up from 0.799. |
| `33-train-resnet-288px-NULL` | GPU 10h | Also worse at 288px — so the data is the ceiling, not the architecture. |

Four short steps were run as inline cells rather than saved notebooks: the submission
smoke test (00), the 160/224 paired comparison (15), the pretrained-weight check (21),
and the 14B blend (29). Their logic is described in `REPORT.pdf`.
