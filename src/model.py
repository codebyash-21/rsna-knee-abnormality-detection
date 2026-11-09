"""
Model for RSNA Knee Abnormality Detection.

    INPUT   one study: 3 series x 32 slices x 224 x 224

      per slice   -> 2.5D: slices [j-1, j, j+1] as RGB channels
                       |
                     CNN encoder, shared across all 96 slices
                       |
      per series  -> bidirectional GRU over the slice axis
                       |  attention-pooled -> one vector per series
      per study   -> attention pool over sagittal / coronal / axial
                       |  masked for missing planes
    OUTPUT            12 logits

Design notes
------------
2.5D rather than 3D convolutions. A meniscal tear spans several slices, so
through-plane context matters, but full 3D costs roughly 4x more and cannot use
ImageNet-pretrained weights -- which matter enormously at 4,407 studies.
Stacking three adjacent slices as colour channels buys most of the context at
2D price. Slices are stacked only within a series; crossing series would blend
different acquisitions.

Bidirectional GRU. The 32 slices form an ordered sequence through the joint.
This is why geometric slice ordering matters: shuffled input makes the sequence
meaningless.

Attention pooling, twice. Averaging 32 slice vectors would dilute a tear visible
on three of them. The same mechanism then runs across planes, because different
findings live in different sequences -- effusion on fluid-sensitive images,
meniscal tears on sagittal, MCL on coronal.

Backbone size. EfficientNet-B0 is 5.3M parameters, chosen deliberately: with
4,407 studies, larger backbones overfit. This was measured rather than assumed
-- ConvNeXt-nano (15.6M) collapsed to 0.55, and B0 at 288px underperformed the
same model at 224px.
"""

import torch
import torch.nn as nn
import timm


class AttnPool(nn.Module):
    """Learned weighted average over a sequence, with optional masking."""

    def __init__(self, dim):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.Tanh(), nn.Linear(dim // 2, 1)
        )

    def forward(self, x, mask=None):            # x: (B, T, D)
        a = self.score(x).squeeze(-1)           # (B, T)
        if mask is not None:
            a = a.masked_fill(mask < 0.5, -1e4)
        return (x * a.softmax(-1).unsqueeze(-1)).sum(1)


class KneeNet(nn.Module):
    """2.5D encoder + slice GRU + series attention.

    Parameters
    ----------
    backbone : timm model name. 'efficientnet_b0' and 'resnet34d' were used in
        the final ensemble. ConvNeXt variants need a lower learning rate than
        the 3e-4 OneCycle peak used here -- see the note in train_fold().
    stack_in_forward : if True the input is (B, S, T, H, W) and the 2.5D channel
        stack is built on the GPU, which moves a third of the data through the
        DataLoader. If False the input arrives pre-stacked as (B, S, T, 3, H, W).
    """

    def __init__(self, backbone="efficientnet_b0", n_labels=12, hidden=256,
                 pretrained=True, stack_in_forward=True):
        super().__init__()
        self.stack_in_forward = stack_in_forward
        self.encoder = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0, in_chans=3
        )
        d = self.encoder.num_features
        self.gru = nn.GRU(d, hidden, batch_first=True, bidirectional=True)
        self.slice_pool = AttnPool(hidden * 2)
        self.series_pool = AttnPool(hidden * 2)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(hidden * 2, n_labels))

    def forward(self, v, mask):
        B, S, T = v.shape[0], v.shape[1], v.shape[2]

        if self.stack_in_forward:
            H, W = v.shape[3], v.shape[4]
            lo = torch.cat([v[:, :, :1], v[:, :, :-1]], dim=2)
            hi = torch.cat([v[:, :, 1:], v[:, :, -1:]], dim=2)
            x = torch.stack([lo, v, hi], dim=3).reshape(B * S * T, 3, H, W)
        else:
            H, W = v.shape[4], v.shape[5]
            x = v.reshape(B * S * T, 3, H, W)

        f = self.encoder(x).reshape(B * S, T, -1)   # (B*S, T, D)
        f, _ = self.gru(f)                          # (B*S, T, 2H)
        f = self.slice_pool(f).reshape(B, S, -1)    # (B, S, 2H)
        return self.head(self.series_pool(f, mask))


def tta_affine(v, angle_deg, scale):
    """Rotate and scale every slice about its centre, on GPU.

    Used for test-time augmentation. At 124,800 slices per view, doing this on
    CPU would cost more than the forward passes.

    No horizontal flips: laterality is canonicalised to a right-knee
    orientation, so mirroring would swap medial and lateral and corrupt four of
    the twelve labels.
    """
    import math
    import torch.nn.functional as F

    if angle_deg == 0.0 and scale == 1.0:
        return v
    B, S, T, H, W = v.shape
    x = v.reshape(B * S * T, 1, H, W)
    a = math.radians(angle_deg)
    cos, sin = math.cos(a) / scale, math.sin(a) / scale
    theta = torch.tensor(
        [[cos, -sin, 0.0], [sin, cos, 0.0]], dtype=x.dtype, device=x.device
    ).unsqueeze(0).expand(x.size(0), -1, -1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    out = F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")
    return out.reshape(B, S, T, H, W)
