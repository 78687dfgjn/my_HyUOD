# Ultralytics YOLO, AGPL-3.0 license
"""Transmission-Guided Edge-Gaussian Refinement for HyUOD.

Experimental HyUOD adaptation using Scharr/Gaussian mechanisms, not the
original LEGNet EGA implementation. Inputs are transmission-P3 and B-P3.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import Conv


class TGER(nn.Module):
    """Refine B-P3 with transmission-guided detail and a zero-init residual."""

    def __init__(self, t_channels, p3_channels, partial_ratio=0.5):
        super().__init__()
        if not 0.0 < partial_ratio <= 1.0:
            raise ValueError("partial_ratio must be in (0, 1].")
        hidden = int(round(p3_channels * partial_ratio / 8.0) * 8)
        hidden = max(8, min(hidden, p3_channels))
        self.t_channels, self.p3_channels, self.hidden = t_channels, p3_channels, hidden
        self.reduce = Conv(p3_channels, hidden, k=1, s=1)
        g1 = torch.tensor([1., 4., 6., 4., 1.], dtype=torch.float32)
        gaussian = torch.outer(g1, g1)
        gaussian = gaussian / gaussian.sum()
        self.register_buffer("gaussian_kernel", gaussian.view(1, 1, 5, 5).repeat(hidden, 1, 1, 1), persistent=False)
        sx = torch.tensor([[-3., 0., 3.], [-10., 0., 10.], [-3., 0., 3.]], dtype=torch.float32) / 16.
        sy = torch.tensor([[-3., -10., -3.], [0., 0., 0.], [3., 10., 3.]], dtype=torch.float32) / 16.
        self.register_buffer("scharr_x", sx.view(1, 1, 3, 3).repeat(hidden, 1, 1, 1), persistent=False)
        self.register_buffer("scharr_y", sy.view(1, 1, 3, 3).repeat(hidden, 1, 1, 1), persistent=False)
        self.t_gate = nn.Conv2d(t_channels, hidden, 1, bias=True)
        self.s_gate = nn.Conv2d(hidden, hidden, 1, bias=True)
        for gate in (self.t_gate, self.s_gate):
            nn.init.zeros_(gate.weight)
            nn.init.zeros_(gate.bias)
        self.detail_fuse = Conv(hidden * 2, hidden, k=1, s=1)
        self.refine = Conv(hidden, hidden, k=3, s=1, g=hidden)
        self.out_proj = Conv(hidden, p3_channels, k=1, s=1, act=False)
        self.alpha = nn.Parameter(torch.zeros(1, p3_channels, 1, 1))

    @staticmethod
    def _match_kernel(kernel, x):
        return kernel.to(device=x.device, dtype=x.dtype)

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise ValueError("TGER expects [transmission_P3, B_P3]")
        t_p3, base_p3 = x
        if t_p3.ndim != 4 or base_p3.ndim != 4:
            raise ValueError("TGER inputs must be BCHW tensors.")
        if t_p3.shape[1] != self.t_channels:
            raise ValueError(f"TGER transmission channels mismatch: {t_p3.shape[1]} != {self.t_channels}")
        if base_p3.shape[1] != self.p3_channels:
            raise ValueError(f"TGER P3 channels mismatch: {base_p3.shape[1]} != {self.p3_channels}")
        if t_p3.shape[-2:] != base_p3.shape[-2:]:
            t_p3 = F.interpolate(t_p3, size=base_p3.shape[-2:], mode="bilinear", align_corners=False)
        feat = self.reduce(base_p3)
        smooth = F.conv2d(feat, self._match_kernel(self.gaussian_kernel, feat), padding=2, groups=self.hidden)
        high = feat - smooth
        gx = F.conv2d(feat, self._match_kernel(self.scharr_x, feat), padding=1, groups=self.hidden)
        gy = F.conv2d(feat, self._match_kernel(self.scharr_y, feat), padding=1, groups=self.hidden)
        edge = torch.sqrt(gx.float().square() + gy.float().square() + 1e-6).to(feat.dtype)
        detail = self.refine(self.detail_fuse(torch.cat((edge, high), dim=1)))
        gate = 1.0 + torch.tanh(self.t_gate(t_p3) + self.s_gate(feat))
        detail = self.out_proj(detail * gate)
        return base_p3 + torch.tanh(self.alpha) * detail
