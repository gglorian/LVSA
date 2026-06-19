"""Snapshot each examples/*_generate.py argparse flag set (run BEFORE refactor).
Re-run only to deliberately re-baseline. Imports each script's parse_args by
monkeypatching sys.argv to ['--help'] is fragile; instead we build the parser."""
import argparse, importlib.util, json, sys
from pathlib import Path

SCRIPTS = ["wan", "hunyuan", "cogvideox", "cosmos"]
EX = Path(__file__).resolve().parents[1] / "examples"

def _jsonify(v):
    """Convert values that aren't JSON-serializable (e.g. PosixPath) to strings."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    return str(v)

def flags_of(parser: argparse.ArgumentParser) -> dict:
    out = {}
    for a in parser._actions:
        if not a.option_strings:
            continue
        default = a.default if a.default is not argparse.SUPPRESS else None
        out[a.option_strings[0]] = {
            "type": getattr(a.type, "__name__", None),
            "default": _jsonify(default),
            "nargs": a.nargs,
            "choices": list(a.choices) if a.choices else None,
        }
    return out

def main():
    baseline = {}
    for name in SCRIPTS:
        spec = importlib.util.spec_from_file_location(f"_ex_{name}", EX / f"{name}_generate.py")
        mod = importlib.util.module_from_spec(spec)
        sys.argv = [name]  # parse_args() must NOT run at import
        spec.loader.exec_module(mod)
        parser = mod.build_parser()
        baseline[name] = flags_of(parser)
    (Path(__file__).parent / "_example_flags_baseline.json").write_text(json.dumps(baseline, indent=2, sort_keys=True))
    print("wrote baseline for", list(baseline))

if __name__ == "__main__":
    main()
