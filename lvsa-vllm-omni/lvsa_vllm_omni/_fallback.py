"""Shared helper to log loud warnings when LVSA falls back to dense attention.

Warnings are emitted ONCE per distinct (origin, reason, seq_len) triplet to avoid
spamming logs (every transformer block × every denoising step would otherwise
produce the same message thousands of times).

Controlled by env var ``LVSA_WARN_FALLBACK`` (default: on). Set to 0/false to
silence all fallback warnings.
"""
import os
from typing import Optional, Set

_FALLBACK_WARN = os.environ.get("LVSA_WARN_FALLBACK", "1").lower() in ("1", "true", "yes")
_seen_warnings: Set[str] = set()


def warn_fallback(
    origin: str,
    reason: str,
    seq_len: int = -1,
    extra: Optional[dict] = None,
    action: str = "falling back to DENSE attention",
) -> None:
    """Emit a one-time warning when LVSA can't run sparse on this call.

    ``action`` describes what happens INSTEAD of the LVSA sparse path. The
    default is a dense fallback, but not every site goes dense: the Wan hook
    hands off to ``self.attn`` (= the LVSA *backend*), which under
    sequence-parallel RE-ENGAGES LVSA on the framework-gathered full grid.
    Pass ``action`` so the log states the real target rather than always
    claiming "DENSE".

    Parameters
    ----------
    origin : str
        Short identifier of the fallback site, e.g. ``"forward_cuda"``,
        ``"ring_processor"``, ``"hunyuan_hook"``, ``"wan_hook"``.
    reason : str
        Short identifier of WHY we fell back, e.g. ``"geometry_detect"``,
        ``"no_t_lat"``, ``"no_encoder"``.
    seq_len : int
        Sequence length at the fallback site (helps diagnose detection issues).
    extra : dict, optional
        Additional context keys (e.g. ``{"T_lat": 33, "P_tried": 1560}``).
    action : str
        Human-readable description of what runs instead of LVSA on this call
        (``"falling back to DENSE attention"`` by default, or a delegate-to
        message for sites that hand off to another forward).
    """
    if not _FALLBACK_WARN:
        return
    key = f"{origin}:{reason}:{seq_len}"
    if key in _seen_warnings:
        return
    _seen_warnings.add(key)
    extra_str = ""
    if extra:
        extra_str = " " + ", ".join(f"{k}={v}" for k, v in extra.items())
    print(
        f"[LVSA-FALLBACK] origin={origin} reason={reason} seq_len={seq_len}{extra_str} "
        f"-- {action} "
        f"(set LVSA_WARN_FALLBACK=0 to silence)",
        flush=True,
    )


def reset_warnings() -> None:
    """Clear the dedup cache — used by tests."""
    _seen_warnings.clear()
