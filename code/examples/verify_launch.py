"""Write a finite mapping study; this does not run GPU hardware."""
import argparse
import json
from pathlib import Path
from lumbre.passport import Launch, verify


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path("reports/passport.json"))
    args = ap.parse_args()
    good = [verify(Launch(m, n, tm, tn))
            for m, n in [(0, 7), (1, 1), (16, 16), (17, 23), (31, 65)]
            for tm, tn in [(4, 8), (8, 8), (16, 16)]]
    assert all(r["status"] == "LOGICAL_MAP_VALIDATED" for r in good)
    bad = verify(Launch(3, 5, 2, 4), lambda y, x, s: y * s.rows + x)
    assert bad["status"] == "FAILED"
    report = {"valid_cases": good, "deliberate_bug": bad,
              "scope": "CPU enumeration of index maps; not a GPU backend port"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("15 valid maps; wrong-stride bug rejected; no GPU execution")


if __name__ == "__main__":
    main()
