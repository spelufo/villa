"""Unlabeled observations attach to any fitted winding, in working scan coordinates."""
import hashlib
import json
import math

import numpy as np
import torch
import torch.nn.functional as F


def load_front_points(path, z_begin, z_end):
    """Read once on the host; the fitter uploads the selected points once per build."""
    import zarr

    root = zarr.open_group(str(path), mode='r')
    metadata = dict(root.attrs)
    if (metadata.get('artifact_type') != 'spiral_front_points'
            or metadata.get('format_version') != 1
            or metadata.get('coordinate_order') != 'zyx'
            or metadata.get('coordinate_units') != 'working_voxels'):
        raise ValueError('unsupported front point format or coordinates')
    points = np.asarray(root['position_zyx'][:], dtype=np.float32)
    shape = np.asarray(metadata.get('shape_zyx', []), dtype=np.float64)
    voxel = metadata.get('working_voxel_size_um', 0)
    origin = np.asarray(metadata.get('origin_mm_xyz', []), dtype=np.float64)
    if (points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all()
            or shape.shape != (3,) or not np.isfinite(shape).all()
            or not (shape > 0).all() or not (shape == np.floor(shape)).all()
            or not np.isfinite(voxel) or voxel <= 0
            or origin.shape != (3,) or not np.isfinite(origin).all()
            or not (points >= 0).all() or not (points <= shape - 1).all()):
        raise ValueError('invalid front point geometry or coordinate metadata')
    # Identity excludes location/provenance: moving an identical artifact is safe.
    geometry = {key: metadata[key] for key in (
        'format_version', 'coordinate_order', 'coordinate_units',
        'working_voxel_size_um', 'origin_mm_xyz', 'shape_zyx')}
    digest = hashlib.sha256(json.dumps(geometry, sort_keys=True).encode())
    digest.update(np.ascontiguousarray(points, dtype='<f4').tobytes())
    selected = points[(points[:, 0] >= z_begin) & (points[:, 0] < z_end)]
    return np.ascontiguousarray(selected), digest.hexdigest()


def get_front_attachment_loss(transform, dr_per_winding, points, num_points,
                              winding_domain, beta=1.0):
    """Radial candidate attachment, not an exact closest-point surface distance.

    Correspondences are detached as in patch DT; gradients flow through inv.
    Missing observations create no reverse attachment or sheet-count penalty.
    """
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError('front attachment beta must be positive')
    inner, outer = winding_domain
    if len(points) == 0 or num_points <= 0 or outer < inner:
        return dr_per_winding.sum() * 0
    observed = points[torch.randint(len(points), (num_points,), device=points.device)]
    with torch.no_grad():
        spiral = transform(observed)
        theta = torch.atan2(spiral[:, 1], spiral[:, 2]).remainder(2 * torch.pi)
        radius = torch.linalg.vector_norm(spiral[:, 1:], dim=-1)
        phase = radius / dr_per_winding.detach() - theta / (2 * torch.pi)
        lower = phase.floor()
        windings = torch.stack((lower, lower + 1), dim=-1).clamp(inner, outer)
        radii = (windings + theta[:, None] / (2 * torch.pi)) * dr_per_winding.detach()
        candidates = torch.stack((spiral[:, 0, None].expand_as(radii),
                                  theta.sin()[:, None] * radii,
                                  theta.cos()[:, None] * radii), dim=-1)
    mapped = transform.inv(candidates.reshape(-1, 3)).reshape(-1, 2, 3)
    distances = torch.linalg.vector_norm(mapped - observed[:, None], dim=-1)
    choice = distances.detach().argmin(dim=-1, keepdim=True)
    distance = distances.gather(1, choice).squeeze(1)
    return F.smooth_l1_loss(distance, torch.zeros_like(distance), beta=beta)
