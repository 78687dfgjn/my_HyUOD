# Ultralytics YOLO, AGPL-3.0 license
"""Physics-Guided Anti-Aliased Edge Refinement for HyUOD.

Experimental PGAER-v1 supplied by the user; not an official MSAD/MSEA module.
Inputs: MFDA P2 detail, P3 transmission, and B/FreqFusion P3 semantics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv


class PGAER(nn.Module):
    """Refine B-P3 with partial-width P2 detail and physical/semantic gates."""

    def __init__(self, p2_channels, t_channels, p3_channels, partial_ratio=0.5):
        super().__init__()
        if not 0.0 < partial_ratio <= 1.0:
            raise ValueError("partial_ratio must be in (0, 1].")

        hidden = int(round(p2_channels * partial_ratio / 8.0) * 8)
        hidden = max(8, min(hidden, p2_channels))
        self.p2_channels = p2_channels
        self.t_channels = t_channels
        self.p3_channels = p3_channels
        self.hidden = hidden

        k3 = torch.tensor([1.0, 2.0, 1.0])
        k3 = torch.outer(k3, k3)
        k3 = k3 / k3.sum()
        k5 = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0])
        k5 = torch.outer(k5, k5)
        k5 = k5 / k5.sum()
        self.register_buffer(
            "blur3", k3.view(1, 1, 3, 3).repeat(hidden, 1, 1, 1), persistent=False
        )
        self.register_buffer(
            "blur5", k5.view(1, 1, 5, 5).repeat(hidden, 1, 1, 1), persistent=False
        )

        scale_mid = max(8, hidden // 4)
        self.scale_gate = nn.Sequential(
            nn.Conv2d(hidden, scale_mid, 1, bias=True),
            nn.SiLU(),
            nn.Conv2d(scale_mid, hidden * 2, 1, bias=True),
        )
        nn.init.zeros_(self.scale_gate[-1].weight)
        nn.init.zeros_(self.scale_gate[-1].bias)

        self.edge_conv = Conv(hidden, hidden, k=3, s=1, g=hidden)
        self.t_gate = nn.Conv2d(t_channels, hidden, kernel_size=1, bias=True)
        self.semantic_gate = nn.Conv2d(p3_channels, hidden, kernel_size=1, bias=True)
        nn.init.zeros_(self.t_gate.weight)
        nn.init.zeros_(self.t_gate.bias)
        nn.init.zeros_(self.semantic_gate.weight)
        nn.init.zeros_(self.semantic_gate.bias)
        self.out_proj = Conv(hidden, p3_channels, k=1, s=1, act=False)
        self.alpha = nn.Parameter(torch.zeros(1, p3_channels, 1, 1))

    @staticmethod
    def _channel_shuffle_2(x):
        """Shuffle channels before selecting the partial-width detail branch."""
        b, c, h, w = x.shape
        if c % 2:
            return x
        x = x.view(b, 2, c // 2, h, w)
        x = x.transpose(1, 2).contiguous()
        return x.view(b, c, h, w)

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 3:
            raise ValueError("PGAER expects [P2_detail, transmission_P3, B_P3]")
        p2, t_p3, base_p3 = x
        p2 = self._channel_shuffle_2(p2)
        detail_src = p2[:, : self.hidden]

        aa3 = F.conv2d(detail_src, self.blur3, stride=2, padding=1, groups=self.hidden)
        aa5 = F.conv2d(detail_src, self.blur5, stride=2, padding=2, groups=self.hidden)
        scale_logits = self.scale_gate(F.adaptive_avg_pool2d(detail_src, 1))
        b = detail_src.shape[0]
        scale_weight = scale_logits.view(b, 2, self.hidden, 1, 1).softmax(dim=1)
        aa = scale_weight[:, 0] * aa3 + scale_weight[:, 1] * aa5

        raw = detail_src[:, :, ::2, ::2]
        if raw.shape[-2:] != aa.shape[-2:]:
            raw = F.interpolate(raw, size=aa.shape[-2:], mode="nearest")
        edge = self.edge_conv(raw - aa)

        target_hw = aa.shape[-2:]
        if t_p3.shape[-2:] != target_hw:
            t_p3 = F.interpolate(t_p3, size=target_hw, mode="bilinear", align_corners=False)
        if base_p3.shape[-2:] != target_hw:
            raise RuntimeError(
                "PGAER: B-P3 spatial size does not match "
                f"anti-aliased P2 size: {base_p3.shape[-2:]} vs {target_hw}"
            )
        gate = torch.sigmoid(self.t_gate(t_p3) + self.semantic_gate(base_p3))
        detail = self.out_proj(aa + gate * edge)
        return base_p3 + torch.tanh(self.alpha) * detail
