"""
visualize_mot17.py
==================
把 tracker 輸出的 MOT17 格式 .txt 結果畫在原始圖片上，逐幀輸出 PNG。

MOT17 結果格式（每行）：
  frame_id, track_id, x, y, w, h, conf, -1, -1, -1

使用範例：
  python3 tools/visualize_mot17.py \
      --seq_dir  /home/caig/data/MOT17/train/MOT17-02-FRCNN \
      --result   ./yolox_mix17/yolox_mix17_det/track_results/MOT17-02-FRCNN.txt \
      --out_dir  ./yolox_mix17/yolox_mix17_det/track_results/visualize/MOT17-02-FRCNN 

參數：
  --seq_dir   MOT17 sequence 根目錄（內含 img1/）
  --result    tracker 輸出的 .txt 檔
  --out_dir   輸出 PNG 存放目錄（預設：<seq_name>_vis/）
  --start     起始幀（預設：1）
  --end       結束幀（預設：全部）
  --thickness bbox 線寬（預設：2）
  --show_id   是否顯示 track ID（預設：True）
"""

import argparse
import configparser
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def id_to_color(track_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(seed=track_id * 6364136223846793005 & 0xFFFFFFFF)
    h = rng.integers(0, 180)
    s = rng.integers(160, 256)
    v = rng.integers(180, 256)
    hsv = np.uint8([[[h, s, v]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def read_seqinfo(seq_dir: Path) -> dict:
    ini = seq_dir / "seqinfo.ini"
    info = {"imExt": ".jpg"}
    if ini.exists():
        cfg = configparser.ConfigParser()
        cfg.read(ini)
        s = cfg["Sequence"]
        info["imExt"] = s.get("imExt", ".jpg")
    return info


def load_results(result_path: Path) -> dict[int, list]:
    data = defaultdict(list)
    with open(result_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue
            fid = int(float(parts[0]))
            tid = int(float(parts[1]))
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            data[fid].append((tid, x, y, w, h))
    return data


def main():
    parser = argparse.ArgumentParser(description="MOT17 result visualizer → PNG frames")
    parser.add_argument("--seq_dir", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--thickness", type=int, default=2)
    parser.add_argument(
        "--show_id", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()

    seq_dir = Path(args.seq_dir)
    img_dir = seq_dir / "img1"
    result_path = Path(args.result)

    if not img_dir.exists():
        sys.exit(f"[ERROR] img1 not found: {img_dir}")
    if not result_path.exists():
        sys.exit(f"[ERROR] Result file not found: {result_path}")

    out_dir = Path(args.out_dir) if args.out_dir else Path(seq_dir.name + "_vis")
    out_dir.mkdir(parents=True, exist_ok=True)

    info = read_seqinfo(seq_dir)
    exts = {info["imExt"], ".jpg", ".jpeg", ".png"}
    img_files = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in exts)
    if not img_files:
        sys.exit(f"[ERROR] No images found in {img_dir}")

    frame_end = args.end if args.end else len(img_files)
    img_files = img_files[args.start - 1 : frame_end]

    detections = load_results(result_path)
    print(f"[INFO] {len(img_files)} frames → {out_dir}")

    first = cv2.imread(str(img_files[0]))
    if first is None:
        sys.exit(f"[ERROR] Cannot read: {img_files[0]}")
    W = first.shape[1]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.4, W / 1920 * 0.8)
    font_thick = max(1, args.thickness - 1)

    for idx, img_path in enumerate(img_files):
        frame_id = args.start + idx
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Cannot read {img_path}, skipping")
            continue

        for tid, x, y, w, h in detections.get(frame_id, []):
            color = id_to_color(tid)
            x1, y1 = int(round(x)), int(round(y))
            x2, y2 = int(round(x + w)), int(round(y + h))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, args.thickness)
            if args.show_id:
                label = str(tid)
                (tw, th), baseline = cv2.getTextSize(
                    label, font, font_scale, font_thick
                )
                cv2.rectangle(
                    img, (x1, y1 - th - baseline - 2), (x1 + tw + 2, y1), color, -1
                )
                cv2.putText(
                    img,
                    label,
                    (x1 + 1, y1 - baseline - 1),
                    font,
                    font_scale,
                    (255, 255, 255),
                    font_thick,
                    cv2.LINE_AA,
                )

        # 幀號
        cv2.putText(
            img,
            f"Frame {frame_id}",
            (8, 24),
            font,
            font_scale * 1.1,
            (255, 255, 255),
            font_thick + 1,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            f"Frame {frame_id}",
            (8, 24),
            font,
            font_scale * 1.1,
            (0, 0, 0),
            font_thick,
            cv2.LINE_AA,
        )

        # 用和原始圖片一樣的 6 位數命名：000001.png
        out_name = f"{frame_id:06d}.png"
        cv2.imwrite(str(out_dir / out_name), img)

        if (idx + 1) % 100 == 0 or idx == 0:
            print(f"  {idx + 1}/{len(img_files)} frames done")

    print(f"[DONE] Saved {len(img_files)} PNGs → {out_dir.resolve()}")


if __name__ == "__main__":
    main()
