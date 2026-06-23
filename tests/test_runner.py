import argparse
import pytest
from examples._runner import add_common_args, add_lvsa_args


def test_common_args_present():
    p = argparse.ArgumentParser()
    add_common_args(p)
    add_lvsa_args(p)
    got = {a.option_strings[0] for a in p._actions if a.option_strings}
    for flag in [
        "--model",
        "--prompt",
        "--num-frames",
        "--height",
        "--width",
        "--steps",
        "--guidance",
        "--seed",
        "--fps",
        "--output-dir",
        "--output-name",
        "--lvsa",
        "--flashinfer",
        "--auto-keyframes",
        "--rotate-keyframes",
        "--sparsity-scale",
        "--window-size",
        "--n-first-frames",
        "--show-mask",
        "--show-mask-compact",
    ]:
        assert flag in got, f"missing {flag}"


def test_common_defaults_match_baseline():
    import json
    import pathlib

    base = json.loads(
        (pathlib.Path(__file__).parent / "_example_flags_baseline.json").read_text()
    )
    p = argparse.ArgumentParser()
    add_common_args(p)
    add_lvsa_args(p)
    defaults = {a.option_strings[0]: a.default for a in p._actions if a.option_strings}
    assert defaults["--seed"] == 16
    assert defaults["--guidance"] == base["wan"]["--guidance"]["default"]


def test_resolve_distributed_single(monkeypatch):
    from examples._runner import resolve_distributed
    for v in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        monkeypatch.delenv(v, raising=False)
    ctx = resolve_distributed(init=False)   # init=False: no dist.init_process_group (CPU test)
    assert ctx.rank == 0 and ctx.world == 1 and ctx.distributed is False


def test_resolve_distributed_multi(monkeypatch):
    from examples._runner import resolve_distributed
    monkeypatch.setenv("RANK", "1"); monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "1")
    ctx = resolve_distributed(init=False)
    assert ctx.rank == 1 and ctx.world == 2 and ctx.distributed is True


def test_step_callback_rotates_and_logs(monkeypatch, capsys):
    from examples._runner import make_step_callback
    class FakeProc:
        def __init__(self): self.steps = []
        def set_step(self, i): self.steps.append(i)
        def print_attention_mask_compact(self): print("[mask]")
    args = argparse.Namespace(rotate_keyframes=True, show_mask_compact=None, profile=False)
    proc = FakeProc()
    monkeypatch.setenv("LVSA_STEP_TIME_LOG", "1")
    cb = make_step_callback(args, proc, rank=0)
    cb(None, 0, None, {}); cb(None, 1, None, {})     # two steps
    assert proc.steps == [0, 1]                       # rotation drove set_step
    out = capsys.readouterr().out
    assert "[LVSA-TIME] step=0" in out                # per-step log emitted from step 1


def test_step_callback_none_when_inactive():
    from examples._runner import make_step_callback
    import os
    os.environ.pop("LVSA_STEP_TIME_LOG", None); os.environ.pop("LVSA_MEM_LOG", None)
    args = argparse.Namespace(rotate_keyframes=False, show_mask_compact=None, profile=False)
    assert make_step_callback(args, lvsa_processor=None, rank=0) is None


def test_setup_cp_wires_both_experts():
    from examples._runner import setup_cp
    from types import SimpleNamespace
    calls = []
    def fake_setup(adapter, transformer, world): calls.append(id(transformer))
    pipe = SimpleNamespace(transformer=object(), transformer_2=object())
    setup_cp(adapter=object(), pipe=pipe, world=2, _setup=fake_setup)
    assert len(calls) == 2          # both transformer and transformer_2 wired

def test_setup_cp_single_transformer():
    from examples._runner import setup_cp
    from types import SimpleNamespace
    calls = []
    pipe = SimpleNamespace(transformer=object(), transformer_2=None)
    setup_cp(adapter=object(), pipe=pipe, world=2, _setup=lambda a, t, w: calls.append(id(t)))
    assert len(calls) == 1          # transformer_2 is None -> only one wiring

def test_build_output_path_descriptive():
    from examples._runner import build_output_path
    import argparse
    args = argparse.Namespace(
        output_name=None, output_dir="out", lvsa=True, flashinfer=True,
        auto_keyframes=True, rotate_keyframes=True, key_frame_interval=None,
        window_size=12, n_first_frames=4, balanced=False, model="/data/Wan2.1-1.3B",
        prompt="a cat runs", height=480, width=832, fps=16, num_frames=161,
        steps=40, guidance=5.0, seed=16)
    p = build_output_path(args, world=1, gen_duration=42.0, mem_mb=8421.0,
                          stem="wan_generate", ext="mp4")
    assert p == ("out/wan_generate_Wan2.1-1.3B_gpu1_480x832@16_frames161"
                 "_lvsa_w12_f4_kfiauto_rot_flashinfer_steps40_cfg5.0_seed16"
                 "_dur42s_mem8421MB_a_cat_runs.mp4")

def test_build_output_path_explicit_name_appends_ext():
    from examples._runner import build_output_path
    import argparse
    args = argparse.Namespace(output_name="myrun", output_dir="out")
    assert build_output_path(args, 1, 0.0, 0.0, stem="x", ext="pt") == "out/myrun.pt"

def test_build_output_path_fullatt_when_no_lvsa():
    from examples._runner import build_output_path
    import argparse
    args = argparse.Namespace(
        output_name=None, output_dir=".", lvsa=False, flashinfer=False,
        auto_keyframes=False, rotate_keyframes=False, key_frame_interval=None,
        window_size=12, n_first_frames=4, balanced=False, model="/m/Foo",
        prompt="x", height=480, width=832, fps=16, num_frames=81,
        steps=40, guidance=5.0, seed=16)
    p = build_output_path(args, world=2, gen_duration=1.0, mem_mb=2.0, stem="s", ext="mp4")
    assert "_fullatt_" in p and "_gpu2_" in p


def test_scheduler_hook_runs_actions_and_increments(monkeypatch, capsys):
    from examples._runner import install_scheduler_step_hook
    class FakeProc:
        def __init__(self): self.steps = []
        def set_step(self, i): self.steps.append(i)
        def print_attention_mask_compact(self): print("[mask]")
    class FakeSched:
        def __init__(self): self.calls = 0
        def step(self, *a, **k): self.calls += 1; return "R"
    class FakePipe:
        def __init__(self): self.scheduler = FakeSched()
    args = argparse.Namespace(rotate_keyframes=True, show_mask_compact=None, profile=False)
    proc = FakeProc(); pipe = FakePipe()
    monkeypatch.setenv("LVSA_STEP_TIME_LOG", "1")
    installed = install_scheduler_step_hook(pipe, args, proc, rank=0)
    assert installed is True
    r1 = pipe.scheduler.step(); r2 = pipe.scheduler.step()
    assert r1 == "R" and r2 == "R"            # original step result passes through
    assert proc.steps == [0, 1]               # rotation driven with incrementing index
    out = capsys.readouterr().out
    assert "[LVSA-TIME] step=0" in out         # gained LVSA-TIME (drift fix)

def test_scheduler_hook_noop_when_inactive(monkeypatch):
    from examples._runner import install_scheduler_step_hook
    import os
    os.environ.pop("LVSA_STEP_TIME_LOG", None); os.environ.pop("LVSA_MEM_LOG", None)
    class FakeSched:
        def __init__(self): self.calls = 0
        def step(self, *a, **k): self.calls += 1; return "R"
    class FakePipe:
        def __init__(self): self.scheduler = FakeSched()
    args = argparse.Namespace(rotate_keyframes=False, show_mask_compact=None, profile=False)
    pipe = FakePipe(); orig_name = pipe.scheduler.step.__func__.__name__
    installed = install_scheduler_step_hook(pipe, args, lvsa_processor=None, rank=0)
    assert installed is False
    assert pipe.scheduler.step.__func__.__name__ == orig_name  # not wrapped when nothing requested


import json, pathlib
import argparse as _argparse


def _norm_default(d):
    # Mirror tests/_gen_flags_baseline.py: SUPPRESS -> None, then stringify any
    # non-JSON-scalar default (e.g. cosmos' --output-dir Path) so it compares
    # against the baseline JSON (which stored it as a string).
    if d is _argparse.SUPPRESS:
        return None
    if d is None or isinstance(d, (bool, int, float, str)):
        return d
    return str(d)


def _parser_flags(build_parser):
    p = build_parser()
    return {a.option_strings[0]: {"type": getattr(a.type, "__name__", None),
            "default": _norm_default(a.default),
            "nargs": a.nargs,
            "choices": list(a.choices) if a.choices else None}
            for a in p._actions if a.option_strings}


def test_wan_flags_unchanged():
    base = json.loads((pathlib.Path(__file__).parent / "_example_flags_baseline.json").read_text())
    from examples import wan_generate
    assert _parser_flags(wan_generate.build_parser) == base["wan"]


def test_hunyuan_flags_unchanged():
    base = json.loads((pathlib.Path(__file__).parent / "_example_flags_baseline.json").read_text())
    from examples import hunyuan_generate
    assert _parser_flags(hunyuan_generate.build_parser) == base["hunyuan"]


def test_cogvideox_flags_unchanged():
    base = json.loads((pathlib.Path(__file__).parent / "_example_flags_baseline.json").read_text())
    from examples import cogvideox_generate
    assert _parser_flags(cogvideox_generate.build_parser) == base["cogvideox"]


def test_cosmos_flags_unchanged():
    # Cosmos is intentionally NOT rewired onto the shared helpers (it is the
    # outlier: single-GPU, required --output-name, Path output-dir, .video, no
    # per-step hook). This test guards its parser against accidental drift.
    # cosmos_generate imports Cosmos3OmniPipeline (diffusers main only); skip on
    # release diffusers (CI), where importing the module raises ImportError.
    pytest.importorskip("diffusers.models.transformers.transformer_cosmos3")
    base = json.loads((pathlib.Path(__file__).parent / "_example_flags_baseline.json").read_text())
    from examples import cosmos_generate
    assert _parser_flags(cosmos_generate.build_parser) == base["cosmos"]


def test_step_callback_show_mask_compact_step(capsys, monkeypatch):
    from examples._runner import make_step_callback
    monkeypatch.delenv("LVSA_STEP_TIME_LOG", raising=False)
    monkeypatch.delenv("LVSA_MEM_LOG", raising=False)

    class P:
        def __init__(self):
            self.steps = []

        def set_step(self, i):
            self.steps.append(i)

        def print_attention_mask_compact(self):
            print("[mask-printed]")

    args = argparse.Namespace(rotate_keyframes=False, show_mask_compact="step", profile=False)
    proc = P()
    cb = make_step_callback(args, proc, rank=0)
    assert cb is not None
    cb(None, 0, None, {})
    out = capsys.readouterr().out
    assert "[LVSA-WINDOW] step 0" in out and "[mask-printed]" in out


def test_step_callback_mem_log_branch(capsys, monkeypatch):
    from examples._runner import make_step_callback
    monkeypatch.setenv("LVSA_MEM_LOG", "1")
    monkeypatch.delenv("LVSA_STEP_TIME_LOG", raising=False)
    args = argparse.Namespace(rotate_keyframes=False, show_mask_compact=None, profile=False)
    cb = make_step_callback(args, lvsa_processor=None, rank=0)
    assert cb is not None              # mem-log alone activates the callback
    cb(None, 0, None, {})              # exercises the LVSA_MEM_LOG branch (memory_stats() may be None on CPU)


def test_auto_keyframes_default_on():
    import argparse
    from examples._runner import add_common_args, add_lvsa_args
    p = argparse.ArgumentParser(); add_common_args(p); add_lvsa_args(p)
    assert p.parse_args(["--model","m","--prompt","x"]).auto_keyframes is True   # default ON now
    assert p.parse_args(["--model","m","--prompt","x","--no-auto-keyframes"]).auto_keyframes is False

def test_expand_window_flag():
    import argparse
    from examples._runner import add_common_args, add_lvsa_args
    p = argparse.ArgumentParser(); add_common_args(p); add_lvsa_args(p)
    assert p.parse_args(["--model","m","--prompt","x"]).expand_window is True            # default expanded
    assert p.parse_args(["--model","m","--prompt","x","--no-expand-window"]).expand_window is False
