# Architecture

```
INPUT   one study: 3 series (sagittal, coronal, axial) x 32 slices x 224 x 224

        ┌─────────────────────────────────────────────────────────────┐
        │  per slice j:  stack [j-1, j, j+1] as RGB channels   (2.5D) │
        └─────────────────────────────────────────────────────────────┘
                                   │  96 three-channel images
                                   ▼
        ┌─────────────────────────────────────────────────────────────┐
        │  CNN encoder  (EfficientNet-B0 or ResNet34d, shared weights) │
        │  ImageNet-pretrained                                        │
        └─────────────────────────────────────────────────────────────┘
                                   │  (B*S, T, D)
                                   ▼
        ┌─────────────────────────────────────────────────────────────┐
        │  bidirectional GRU over the 32-slice axis                   │
        │  attention pool  ->  one vector per series                  │
        └─────────────────────────────────────────────────────────────┘
                                   │  (B, 3, 2H)
                                   ▼
        ┌─────────────────────────────────────────────────────────────┐
        │  attention pool over sagittal / coronal / axial             │
        │  masked where a plane is missing                            │
        └─────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
OUTPUT  12 logits  ->  BCEWithLogitsLoss against soft targets

ENSEMBLE  5x EfficientNet-B0 + 5x ResNet34d, equal weight, TTA x3
```

## Why each piece

**2.5D, not 3D.** A meniscal tear spans several slices, so through-plane context matters.
Full 3D convolutions cost roughly 4x more and cannot use ImageNet-pretrained weights,
which matter enormously at 4,407 studies. Three adjacent slices as colour channels buys
most of the context at 2D price. Slices are stacked only within a series — crossing
series would blend different acquisitions.

**Bidirectional GRU.** The 32 slices form an ordered sequence through the joint, and each
slice should be informed by both neighbours. This is why geometric slice ordering matters:
shuffled input makes the sequence meaningless.

**Attention pooling, twice.** Averaging 32 slice vectors would dilute a tear visible on
three of them. The same mechanism runs across planes, because different findings live in
different sequences — effusion and Baker's cysts on fluid-sensitive images, meniscal tears
on sagittal, MCL on coronal.

**Small backbones.** EfficientNet-B0 is 5.3M parameters, chosen deliberately: with 4,407
studies, larger backbones overfit. Measured, not assumed — ConvNeXt-nano (15.6M) collapsed
to 0.55, and B0 at 288px underperformed the same model at 224px.

## Training

| | |
|---|---|
| Loss | `BCEWithLogitsLoss` on soft targets, per-label weights from extraction reliability |
| Optimiser | AdamW, weight decay 1e-4 |
| Schedule | OneCycleLR, peak 3e-4, 10% warmup |
| Epochs | 8 — checkpoints consistently peaked at epochs 4–6, confirming saturation |
| Batch | 3 studies, 2-step gradient accumulation |
| Precision | Mixed, gradient clipping at norm 1.0 |
| Augmentation | Affine jitter (±12°, 0.9–1.1 scale, ±6% translate), intensity gain/bias. **No flips.** |
| Validation | `MultilabelStratifiedKFold`, 5 folds, the 58 gold studies excluded from training |

No horizontal flips: laterality is canonicalised to a right-knee orientation, so mirroring
would swap medial and lateral and corrupt four of the twelve labels.
