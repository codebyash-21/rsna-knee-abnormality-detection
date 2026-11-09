"""
DICOM decoding for RSNA Knee Abnormality Detection.

Four things here are load-bearing. Each would silently degrade the model if
wrong -- no exception, no visible symptom in the loss curve.

1. Geometric slice ordering. Filenames and InstanceNumber are not reliably in
   anatomical order. The slice normal is the cross product of the two
   ImageOrientationPatient direction cosines; slices sort by the projection of
   ImagePositionPatient onto it.

2. Per-series percentile normalisation. MRI has no absolute intensity scale the
   way CT has Hounsfield units. Stored WindowCenter ranged from 148 to 1660 in
   this dataset, BitsStored is 12 or 16, and PixelRepresentation mixes signed
   and unsigned. Percentiles are indifferent to all of it.

3. Canonical orientation. Volume axes are flipped so each points consistently
   in patient space.

4. Laterality. Four of the twelve labels are compartment-specific, and medial
   and lateral swap sides between a left and a right knee, so every knee is
   mirrored to a right-knee convention.

   The Laterality DICOM tag is present in only ~52.5% of studies. Earlier code
   silently skipped mirroring when it was absent, which left roughly 1,000
   studies (24% of the dataset) with medial and lateral reversed. The fallback
   below derives laterality from the x-coordinate of ImagePositionPatient:
   DICOM uses LPS coordinates where +x is the patient's left. Validated against
   the 2,203 studies that do carry a tag -- 97.5% accurate, AUC 0.982.

   Subtlety: on a SAGITTAL image the medial-lateral direction is the
   through-slice axis, not the in-plane horizontal. Mirroring therefore means
   reversing slice order, not flipping the picture. The code finds which volume
   axis tracks patient left-right from the direction cosines rather than
   assuming.
"""

import os
import numpy as np
import cv2
import pydicom

# Threshold and sign fitted on the 2,203 tagged studies (see notebook 12).
LAT_THRESHOLD = -37.5
LAT_SIGN = 1

PLANES = ["Sagittal", "Coronal", "Axial"]


def select_series(group, planes=PLANES):
    """Pick one series per anatomical plane from a study's series metadata.

    Prefers fluid-sensitive then fat-suppressed sequences: those show effusion,
    synovitis, contusion and Baker's cysts most clearly. Returns a list the same
    length as `planes`, with None where a plane is absent.
    """
    out = []
    for plane in planes:
        sub = group[group["Anatomical_Plane"] == plane]
        if len(sub) == 0:
            out.append(None)
            continue
        sub = sub.sort_values(["Fluid_Sensitive", "Fat_Suppression"], ascending=False)
        out.append(sub.iloc[0]["SeriesInstanceUID"])
    return out


def _read_headers(dirpath):
    """Read geometry and laterality from every slice header in a series."""
    metas = []
    for fname in os.listdir(dirpath):
        try:
            ds = pydicom.dcmread(os.path.join(dirpath, fname), stop_before_pixels=True)
        except Exception:
            continue
        iop = getattr(ds, "ImageOrientationPatient", None)
        ipp = getattr(ds, "ImagePositionPatient", None)
        metas.append((
            fname,
            np.array(iop, float) if iop else None,
            np.array(ipp, float) if ipp else None,
            getattr(ds, "InstanceNumber", 0),
            str(getattr(ds, "Laterality", "") or ""),
        ))
    return metas


def resolve_laterality(metas, hint=None):
    """Return 'L' or 'R'. Tag first, then geometry. Never returns empty."""
    if hint in ("L", "R"):
        return hint
    tag = next((m[4] for m in metas if m[4]), "")
    if tag:
        return tag.strip().upper()[0]
    xs = [float(m[2][0]) for m in metas if m[2] is not None]
    if xs:
        return "L" if (np.mean(xs) - LAT_THRESHOLD) * LAT_SIGN > 0 else "R"
    return "R"


def load_series(dirpath, px=224, n_slices=32, laterality_hint=None):
    """Decode one MRI series into a (n_slices, px, px) uint8 volume.

    Returns None if the directory is empty or no header could be read.
    """
    metas = _read_headers(dirpath)
    if not metas:
        return None

    # --- 1. geometric slice ordering -------------------------------------
    iops = [m[1] for m in metas if m[1] is not None]
    if iops:
        row_cos, col_cos = iops[0][:3], iops[0][3:]
        normal = np.cross(row_cos, col_cos)
        positions = [
            float(np.dot(m[2], normal)) if m[2] is not None else float(m[3])
            for m in metas
        ]
    else:  # no geometry at all -- fall back to InstanceNumber
        row_cos = np.array([1.0, 0, 0])
        col_cos = np.array([0, 1.0, 0])
        normal = np.array([0, 0, 1.0])
        positions = [float(m[3]) for m in metas]

    metas = [metas[i] for i in np.argsort(positions)]
    take = np.linspace(0, len(metas) - 1, n_slices).round().astype(int)

    # --- decode pixels ----------------------------------------------------
    slices = []
    for i in take:
        try:
            ds = pydicom.dcmread(os.path.join(dirpath, metas[i][0]))
            arr = ds.pixel_array.astype(np.float32)
            if hasattr(ds, "RescaleSlope"):
                arr = arr * float(ds.RescaleSlope) + float(getattr(ds, "RescaleIntercept", 0))
        except Exception:
            arr = np.zeros((px, px), np.float32)
        slices.append(cv2.resize(arr, (px, px), interpolation=cv2.INTER_AREA))
    vol = np.stack(slices)

    # --- 2. per-series percentile normalisation ---------------------------
    lo, hi = np.percentile(vol, [1, 99])
    if hi <= lo:
        hi = lo + 1
    vol = np.clip((vol - lo) / (hi - lo), 0, 1)

    # --- 3. canonical orientation -----------------------------------------
    axes = [normal, col_cos, row_cos]  # volume axes Z, Y, X
    for ax, vec in enumerate(axes):
        dominant = int(np.argmax(np.abs(vec)))
        if vec[dominant] < 0:
            vol = np.flip(vol, axis=ax)
        axes[ax] = vec * (1 if vec[dominant] >= 0 else -1)

    # --- 4. mirror left knees onto the right-knee convention --------------
    if resolve_laterality(metas, laterality_hint) == "L":
        lr_axis = int(np.argmax([abs(v[0]) for v in axes]))  # which axis tracks patient L-R
        vol = np.flip(vol, axis=lr_axis)

    return (vol * 255).astype(np.uint8)
