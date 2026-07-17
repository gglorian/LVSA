import torch.nn as nn
from lvsa_vllm_omni.cosmos3_backend import _swap_seam_to_lvsa
from lvsa_vllm_omni.attention_impl import LVSAAttentionImpl


class _FakeFrameworkAttn(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.attention = object()          # stand-in resolved backend impl
        self.attn_impl_cls = object
        self.head_dim = head_dim


class _FakeCrossSeam(nn.Module):
    """Mirrors the attrs _swap_seam_to_lvsa reads on Cosmos3CrossAttention: the
    PER-RANK (local) head counts, matching the seam's own forward under TP."""
    def __init__(self, heads_local=8, kv_heads_local=2, head_dim=16):
        super().__init__()
        self.num_heads_local = heads_local; self.num_kv_heads_local = kv_heads_local
        self.head_dim = head_dim
        self.attn = _FakeFrameworkAttn(head_dim)


def test_swap_installs_marked_lvsa_impl_with_local_seam_dims():
    seam = _FakeCrossSeam(heads_local=8, kv_heads_local=2, head_dim=16)
    _swap_seam_to_lvsa(seam)
    impl = seam.attn.attention
    assert isinstance(impl, LVSAAttentionImpl)
    assert impl.cosmos_gen is True
    assert impl.num_heads == 8 and impl.num_kv_heads == 2 and impl.head_size == 16
    assert abs(impl.softmax_scale - 1.0 / (16 ** 0.5)) < 1e-9
    # attn_impl_cls is intentionally left untouched (see _swap_seam_to_lvsa): the
    # installed instance is what matters, and rebinding the class is a dead footgun.
    assert seam.attn.attn_impl_cls is object
