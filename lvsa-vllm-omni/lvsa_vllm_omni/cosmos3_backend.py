"""Cosmos 3.0 LVSA attention-backend seam-swap (the go-forward path).

Installs LVSA on Cosmos3CrossAttention's resolved attention backend
(`seam.attn.attention`) at construction — engages sparse single-GPU AND under
Ulysses SP. Gated by LVSA_COSMOS3_BACKEND=1 (see register.py).
"""


def _swap_seam_to_lvsa(seam) -> None:
    """Replace a Cosmos3CrossAttention gen seam's resolved backend impl
    (`seam.attn.attention`) with a cosmos-gen-marked LVSAAttentionImpl matching
    the seam's head geometry. Pure (operates on an already-built seam) so it is
    CPU-unit-testable on a fake seam."""
    from .attention_impl import LVSAAttentionImpl
    # Use the PER-RANK (local) head counts — the seam's own forward slices with
    # num_heads_local/num_kv_heads_local (transformer_cosmos3.py), and under TP the
    # global counts would be wrong. (LVSAAttentionImpl.forward derives all shapes
    # from tensors and never reads these, so this is a stored-geometry correctness
    # fix, not a runtime shape change — but it should still be right.)
    impl = LVSAAttentionImpl(
        num_heads=seam.num_heads_local,
        head_size=seam.head_dim,
        softmax_scale=1.0 / (seam.head_dim ** 0.5),
        num_kv_heads=seam.num_kv_heads_local,
        cosmos_gen=True,
    )
    seam.attn.attention = impl
    # Deliberately do NOT reassign ``seam.attn.attn_impl_cls``: the framework layer
    # reads it only once, when it builds ``.attention`` at __init__ (already done
    # before this swap). Nothing reads it afterwards, so setting it has no effect —
    # and it is a footgun: any future re-instantiation from the class would rebuild
    # LVSAAttentionImpl WITHOUT ``cosmos_gen=True`` and silently degrade Cosmos to dense.


def install_cosmos3_lvsa_backend(
    total_latent_frames: int | None = None,
    num_patches: int | None = None,
) -> None:
    """Route the Cosmos gen seam through the LVSA attention backend by swapping
    `Cosmos3CrossAttention.attn.attention` at construction. Unlike the forward-patch
    hook, this engages under Ulysses (the framework gathers the full grid before the
    backend). Targets only the gen seam (Cosmos3CrossAttention); the und causal seam
    is a different class and is untouched. Idempotent.

    ``total_latent_frames`` / ``num_patches`` are **diagnostic only** — they are
    echoed in the install log so it matches what the forward path will use, but the
    swap itself consumes neither. The authoritative geometry is resolved per-forward
    by ``LVSAAttentionImpl`` from ``LVSA_TOTAL_LATENT_FRAMES`` / ``LVSA_PATCHES_PER_FRAME``
    (or, failing that, detected from the sequence). Pass them for an accurate log."""
    from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3 import (
        Cosmos3CrossAttention,
    )
    if getattr(Cosmos3CrossAttention.__init__, "_lvsa_backend_installed", False):
        return
    _orig_init = Cosmos3CrossAttention.__init__

    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        _swap_seam_to_lvsa(self)

    _patched_init._lvsa_backend_installed = True
    Cosmos3CrossAttention.__init__ = _patched_init
    print(f"[LVSA] Cosmos3 backend seam installed (T_lat={total_latent_frames}, "
          f"num_patches={num_patches}) — gen seam → LVSA under all parallelisms")
