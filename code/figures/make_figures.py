"""Regenerate every figure used by the LaTeX notes into latex/figures/.

Each module in this folder named chNN_*.py exposes main(outdir).
"""
import importlib, pathlib, sys

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent.parent / "latex" / "figures"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(HERE))
    mods = sorted(p.stem for p in HERE.glob("ch[0-9][0-9]_*.py"))
    for name in mods:
        print(f"-> {name}")
        importlib.import_module(name).main(OUT)
    print(f"{len(mods)} figure scripts done; output in {OUT}")


if __name__ == "__main__":
    main()
