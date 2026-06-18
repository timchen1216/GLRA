import shutil
import argparse
from pathlib import Path

DETECTORS = ["SDP", "DPM", "FRCNN"]
# 可自行調整要處理的 sequence
TARGET_SEQS = [
    "MOT17-01",
    "MOT17-03",
    "MOT17-06",
    "MOT17-07",
    "MOT17-08",
    "MOT17-12",
    "MOT17-14",
]


def duplicate_mot17_results(input_dir: str):
    input_path = Path(input_dir)

    if not input_path.exists():
        print(f"❌ Directory not found: {input_path}")
        return

    output_path = input_path.parent / "mot17_test"
    output_path.mkdir(exist_ok=True)

    for seq in TARGET_SEQS:
        src = input_path / f"{seq}.txt"
        if not src.exists():
            print(f"⚠️  Source not found, skipping: {src.name}")
            continue
        for det in DETECTORS:
            dst = output_path / f"{seq}-{det}.txt"
            shutil.copy2(src, dst)
            print(f"  ✅ {src.name}  →  {dst.name}")
        print()

    print(f"Done. Output: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Duplicate MOT17 result .txt files with SDP/DPM/FRCNN suffixes"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./yolox_mix17/yolox_mix17_det/track_results_test",
        help="Directory containing MOT17-XX.txt result files",
    )
    args = parser.parse_args()
    duplicate_mot17_results(args.input_dir)
