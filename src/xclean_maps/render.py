# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Necessity Robotics, India
"""Rasteriser interface used by the trainer, eval, export previews and the live view.

render(params, T_cw, K, W, H, sh_degree, depth) -> (colors [H, W, 3(+1)], alphas [H, W, 1], info)
  colors are premultiplied by alpha (no background); with depth=True the last channel is the EXPECTED depth (ED).

gsplat_renderer(): gsplat 1.4.0 rasterization (GPU), antialiased, packed=False (BUILDER 5.1, 5.8).
reference_renderer(): a small differentiable torch rasteriser with the same maths (EWA projection, eps2d dilation with
the antialiasing compensation, SH degree <= 1, front-to-back compositing, alpha >= 1/255, ED) for CPU unit tests of the
trainer loop on tiny synthetic scenes. It is never used for a map.
"""
from __future__ import annotations

import torch

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199


def opencv_to_gsplat_K(K):
    """Review F5: K is in the OpenCV convention (pixel (i, j) centred at image coordinates (j, i): cv2.remap, the RAFT
    cache, masks, reprojection); gsplat 1.4.0 samples pixel (i, j) at (j + 0.5, i + 0.5) (rasterize_to_pixels_fwd.cu).
    Passing K with cx + 0.5, cy + 0.5 puts a point that projects to OpenCV (u, v) on the centre of pixel (v, u)."""
    K2 = K.clone() if isinstance(K, torch.Tensor) else torch.as_tensor(K, dtype=torch.float32)
    K2 = K2.clone()
    K2[..., 0, 2] = K2[..., 0, 2] + 0.5
    K2[..., 1, 2] = K2[..., 1, 2] + 0.5
    return K2


def activated(params):
    return dict(means=params["means"], quats=torch.nn.functional.normalize(params["quats"], dim=-1),
                scales=torch.exp(params["scales"]), opacities=torch.sigmoid(params["opacities"].reshape(-1)),
                sh=torch.cat([params["sh0"], params["shN"]], 1))


def gsplat_renderer(rasterize_mode="antialiased", packed=False, near_plane=0.05, far_plane=1000.0, eps2d=0.3):
    from gsplat import rasterization

    def render(params, T_cw, K, W, H, sh_degree, depth=True, camera_model="pinhole"):
        a = activated(params)
        Kg = opencv_to_gsplat_K(K)
        colors, alphas, info = rasterization(
            a["means"], a["quats"], a["scales"], a["opacities"], a["sh"], T_cw[None], Kg[None], int(W), int(H),
            near_plane=near_plane, far_plane=far_plane, eps2d=eps2d, sh_degree=int(sh_degree), packed=packed,
            absgrad=False, rasterize_mode=rasterize_mode, render_mode="RGB+ED" if depth else "RGB",
            camera_model=camera_model)
        return colors[0], alphas[0], info
    render.kind = "gsplat-%s" % rasterize_mode
    return render


def _sh_colors(sh, means, campos, deg):
    c = SH_C0 * sh[:, 0, :]
    if deg >= 1:
        d = torch.nn.functional.normalize(means - campos[None], dim=-1)
        x, y, z = d[:, 0:1], d[:, 1:2], d[:, 2:3]
        c = c + SH_C1 * (-y * sh[:, 1, :] + z * sh[:, 2, :] - x * sh[:, 3, :])
    return torch.clamp_min(c + 0.5, 0.0)


def reference_renderer(antialiased=True, near_plane=0.05, far_plane=1000.0, eps2d=0.3):
    def render(params, T_cw, K, W, H, sh_degree, depth=True, camera_model="pinhole"):
        assert camera_model == "pinhole"
        a = activated(params)
        R, t = T_cw[:3, :3], T_cw[:3, 3]
        mc = a["means"] @ R.T + t
        z = mc[:, 2]
        keep = (z > near_plane) & (z < far_plane)
        idx = torch.nonzero(keep, as_tuple=True)[0]
        dev = a["means"].device
        Cn = 3
        if idx.numel() == 0:
            col = torch.zeros((H, W, Cn + (1 if depth else 0)), device=dev)
            return col, torch.zeros((H, W, 1), device=dev), dict(radii=torch.zeros(0))
        mc, z = mc[idx], z[idx]
        Kg = opencv_to_gsplat_K(K)                      # same pixel convention as gsplat (centres at +0.5)
        fx, fy, cx, cy = Kg[0, 0], Kg[1, 1], Kg[0, 2], Kg[1, 2]
        x, y = mc[:, 0], mc[:, 1]
        u = fx * x / z + cx
        v = fy * y / z + cy
        Rq = _quat_to_R(a["quats"][idx])
        S = a["scales"][idx]
        M = Rq * S[:, None, :]
        cov3 = M @ M.transpose(1, 2)
        covc = R[None] @ cov3 @ R.T[None]
        J = torch.zeros((len(idx), 2, 3), device=dev)
        J[:, 0, 0] = fx / z
        J[:, 0, 2] = -fx * x / (z * z)
        J[:, 1, 1] = fy / z
        J[:, 1, 2] = -fy * y / (z * z)
        cov2 = J @ covc @ J.transpose(1, 2)
        det0 = cov2[:, 0, 0] * cov2[:, 1, 1] - cov2[:, 0, 1] * cov2[:, 1, 0]
        cov2 = cov2 + eps2d * torch.eye(2, device=dev)[None]
        det = cov2[:, 0, 0] * cov2[:, 1, 1] - cov2[:, 0, 1] * cov2[:, 1, 0]
        op = a["opacities"][idx]
        if antialiased:
            op = op * torch.sqrt(torch.clamp(det0 / det, min=0.0))
        inv = torch.stack([torch.stack([cov2[:, 1, 1], -cov2[:, 0, 1]], -1),
                           torch.stack([-cov2[:, 1, 0], cov2[:, 0, 0]], -1)], -2) / det[:, None, None]
        campos = -R.T @ t
        col = _sh_colors(a["sh"][idx], a["means"][idx], campos, sh_degree)
        order = torch.argsort(z.detach())
        u, v, inv, op, col, zz = u[order], v[order], inv[order], op[order], col[order], z[order]
        jj, ii = torch.meshgrid(torch.arange(W, device=dev, dtype=torch.float32) + 0.5,
                                torch.arange(H, device=dev, dtype=torch.float32) + 0.5, indexing="xy")
        px = torch.stack([jj.reshape(-1), ii.reshape(-1)], -1)                    # [P, 2]
        d = px[:, None, :] - torch.stack([u, v], -1)[None]                         # [P, N, 2]
        sig = 0.5 * (inv[None, :, 0, 0] * d[..., 0] ** 2 + inv[None, :, 1, 1] * d[..., 1] ** 2
                     + (inv[None, :, 0, 1] + inv[None, :, 1, 0]) * d[..., 0] * d[..., 1])
        al = torch.clamp(op[None] * torch.exp(-sig), max=0.99)
        al = torch.where(al >= 1.0 / 255.0, al, torch.zeros_like(al))
        T = torch.cumprod(torch.cat([torch.ones_like(al[:, :1]), 1 - al[:, :-1]], 1), 1)
        # gsplat's early termination: a gaussian whose compositing would drop T to <= 1e-4 and everything behind it are
        # not composited at all (exactly zero gradient for gaussians behind an opaque surface)
        live = (T * (1 - al)).detach() > 1e-4
        w = T * al * live                                                           # [P, N]
        rgb = w @ col                                                               # [P, 3]
        alpha = w.sum(1, keepdim=True)
        out = [rgb]
        if depth:
            ed = (w @ zz[:, None]) / alpha.clamp_min(1e-10)
            out.append(ed)
        colors = torch.cat(out, -1).reshape(H, W, -1)
        radii = torch.zeros(a["means"].shape[0], dtype=torch.int32, device=dev)
        radii[idx] = 1
        return colors, alpha.reshape(H, W, 1), dict(radii=radii)
    render.kind = "reference-cpu"
    return render


def _quat_to_R(q):
    q = torch.nn.functional.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(q.shape[:-1] + (3, 3))


def look_at(eye, target, up):
    """T_cw for an OpenCV camera (x right, y down, z forward) at eye looking at target ([FIN] look_at)."""
    import numpy as np
    f = target - eye
    f = f / np.linalg.norm(f)
    r = np.cross(f, up)
    if np.linalg.norm(r) < 1e-9:
        r = np.cross(f, np.array([1.0, 0, 0]) if abs(f[0]) < 0.9 else np.array([0, 1.0, 0]))
    r = r / np.linalg.norm(r)
    d = np.cross(f, r)
    Rcw = np.stack([r, d, f], 0)
    T = np.eye(4)
    T[:3, :3] = Rcw
    T[:3, 3] = -Rcw @ eye
    return T
