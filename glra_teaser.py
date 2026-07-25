r"""
glra_teaser.py  ──  GLRA 接回成功案例「論文第一頁 teaser 圖」產生器
================================================================================
這支程式不是 runtime overlay（那是 visualize.py 的工作），而是專門產生一張
publication-quality 的多格 filmstrip 圖，用來放在論文第一頁，講清楚一個故事：

    某條軌跡因遮擋 / 漏檢而 Lost  →  GPR 在空窗期外推預測位置（虛線）
    →  在某一幀用 low-score detection 重新接回（GLRA recovery）
    →  ID 維持不變。

並可選擇加一條「baseline 對照列」：同一段，沒有 GLRA 時 track 被換新 ID
（ID switch）。上下兩列對照，就是最有說服力的 teaser。

特色：
  • 純 matplotlib 出圖，輸出 PDF（向量）+ 高 DPI PNG，直接進 LaTeX \includegraphics
  • 字型用 serif（搭配 CVPR 雙欄），所有標註與你的論文用語一致
  • 配色考慮黑白列印仍可辨識（實線 vs 虛線 + 形狀區分，不只靠顏色）
  • 只吃標準 MOT 格式結果檔（frame,id,x,y,w,h,...）＋ 序列影像，可重現

--------------------------------------------------------------------------------
用法（單列，只展示 Ours 的接回過程）：
    python glra_teaser.py \
        --seq_dir   /path/to/MOT17-04/img1 \
        --ours_txt  /path/to/ours/MOT17-04.txt \
        --track_id  17 \
        --frames    300 312 324 336 348 \
        --out       fig1_glra_teaser

用法（雙列對照，加上 baseline 的 ID switch）：
    python glra_teaser.py \
        --seq_dir       /path/to/MOT17-04/img1 \
        --ours_txt      /path/to/ours/MOT17-04.txt \
        --base_txt      /path/to/baseline/MOT17-04.txt \
        --track_id      17 \
        --base_id_pre   17 \
        --base_id_post  93 \
        --frames        300 312 324 336 348 \
        --lost_frames   312 324 336 \
        --recover_frame 348 \
        --out           fig1_glra_teaser

說明：
    --track_id      Ours 結果中要追蹤的那條 ID（全程同一個）
    --frames        要顯示的幀號（建議 4~6 格）
    --lost_frames   這些幀該 track 在 Ours 是「Lost / GPR 預測」狀態（畫橘色虛線框）
    --recover_frame GLRA 成功接回的那一幀（畫 GPR 橘虛線 + low-score det 紅虛線）
    --base_id_pre   baseline 在接回前用的 ID
    --base_id_post  baseline 接回後「換成」的新 ID（ID switch 的證據）
================================================================================
"""

import argparse
import glob
import os
import sys

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D

# ── 配色（黑白列印友善：靠線型 + 標籤輔助，不只靠顏色）────────────────────
C_ACTIVE = "#2ca02c"  # 綠：track 正常 active
C_LOST = "#ff7f0e"  # 橘：GPR 預測（lost 空窗期）
C_DET = "#d62728"  # 紅：matched low-score detection
C_SWITCH = "#9467bd"  # 紫：baseline 換掉的新 ID
C_ARROW = "#ff7f0e"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,  # TrueType，避免 LaTeX 內嵌字型問題
        "ps.fonttype": 42,
    }
)


# ── 讀取 MOT 結果 ────────────────────────────────────────────────────────
def load_mot(txt_path):
    """回傳 dict: {frame_id: {track_id: (x, y, w, h, score)}}"""
    data = {}
    if not os.path.isfile(txt_path):
        raise FileNotFoundError(f"找不到結果檔: {txt_path}")
    with open(txt_path) as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 6:
                continue
            fid, tid = int(float(p[0])), int(float(p[1]))
            x, y, w, h = float(p[2]), float(p[3]), float(p[4]), float(p[5])
            score = float(p[6]) if len(p) > 6 and p[6] not in ("", "-1") else 1.0
            data.setdefault(fid, {})[tid] = (x, y, w, h, score)
    return data


def find_frame_image(seq_dir, frame_id):
    """支援 000300.jpg / 0300.jpg / 300.jpg / .png 等命名。"""
    for ext in ("jpg", "png"):
        for n in (6, 5, 4, 3, 0):
            name = f"{frame_id:0{n}d}.{ext}" if n else f"{frame_id}.{ext}"
            cand = os.path.join(seq_dir, name)
            if os.path.isfile(cand):
                return cand
    # 後備：抓排序後第 frame_id 張
    imgs = sorted(
        glob.glob(os.path.join(seq_dir, "*.jpg"))
        + glob.glob(os.path.join(seq_dir, "*.png"))
    )
    if 1 <= frame_id <= len(imgs):
        return imgs[frame_id - 1]
    raise FileNotFoundError(f"找不到第 {frame_id} 幀影像於 {seq_dir}")


def crop_around(img, box, pad_ratio=0.9, min_size=120):
    """以 box 為中心裁一塊（含 padding）方便看清楚目標。回傳 (crop, ox, oy)。"""
    H, W = img.shape[:2]
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    half_w = max(w * (1 + pad_ratio), min_size) / 2
    half_h = max(h * (1 + pad_ratio), min_size) / 2
    # 維持固定長寬比，讓每格大小一致
    half_w = half_h = max(half_w, half_h)
    x0 = int(max(0, cx - half_w))
    y0 = int(max(0, cy - half_h))
    x1 = int(min(W, cx + half_w))
    y1 = int(min(H, cy + half_h))
    return img[y0:y1, x0:x1], x0, y0


def draw_box(ax, box, ox, oy, color, dashed=False, lw=2.0, label=None, label_loc="top"):
    x, y, w, h = box
    rx, ry = x - ox, y - oy
    style = (0, (4, 3)) if dashed else "solid"
    rect = Rectangle(
        (rx, ry),
        w,
        h,
        fill=False,
        edgecolor=color,
        linewidth=lw,
        linestyle=style,
        joinstyle="round",
    )
    ax.add_patch(rect)
    if label:
        ly = ry - 4 if label_loc == "top" else ry + h + 11
        va = "bottom" if label_loc == "top" else "top"
        ax.text(
            rx,
            ly,
            label,
            color="white",
            fontsize=7,
            va=va,
            ha="left",
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.18", fc=color, ec="none"),
        )


def panel(
    ax,
    seq_dir,
    frame_id,
    box,
    status,
    track_id,
    gpr_box=None,
    det_box=None,
    det_score=None,
):
    """
    status ∈ {"active", "lost", "recover"}
      active  : 綠色實線框 + #id
      lost    : 橘色虛線框（GPR 預測）+ "GPR pred"
      recover : 橘色虛線(GPR 預測) + 紅色虛線(low-score det)，不畫綠色實線、不畫箭頭
    """
    img_path = find_frame_image(seq_dir, frame_id)
    img = plt.imread(img_path)
    if img.dtype != np.uint8 and img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)

    anchor = box if box is not None else (gpr_box or det_box)
    crop, ox, oy = crop_around(img, anchor)
    ax.imshow(crop)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    if status == "active":
        draw_box(ax, box, ox, oy, C_ACTIVE, dashed=False, lw=2.2, label=f"#{track_id}")
    elif status == "lost":
        b = gpr_box if gpr_box is not None else box
        draw_box(ax, b, ox, oy, C_LOST, dashed=True, lw=2.0, label=f"#{track_id} GPR")
    elif status == "recover":
        # GPR 預測框（橘虛線）── 接回那一刻 GLRA 用來配對的預測位置
        if gpr_box is not None:
            draw_box(
                ax,
                gpr_box,
                ox,
                oy,
                C_LOST,
                dashed=True,
                lw=2.0,
                label=f"#{track_id} GPR",
            )
        # low-score detection 框（紅虛線）── 被接回的低分檢測
        if det_box is not None:
            lbl = "low-score det"
            if det_score is not None:
                lbl += f" ({det_score:.2f})"
            draw_box(
                ax,
                det_box,
                ox,
                oy,
                C_DET,
                dashed=True,
                lw=1.8,
                label=lbl,
                label_loc="bottom",
            )

    ax.set_title(f"frame {frame_id}", fontsize=8, pad=2)


def baseline_panel(ax, seq_dir, frame_id, box, track_id, is_switch):
    img_path = find_frame_image(seq_dir, frame_id)
    img = plt.imread(img_path)
    if img.dtype != np.uint8 and img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    crop, ox, oy = crop_around(img, box)
    ax.imshow(crop)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    color = C_SWITCH if is_switch else C_ACTIVE
    lbl = f"#{track_id}" + ("  (ID switch!)" if is_switch else "")
    draw_box(ax, box, ox, oy, color, dashed=False, lw=2.2, label=lbl)


def build_figure(args):
    ours = load_mot(args.ours_txt)
    base = load_mot(args.base_txt) if args.base_txt else None

    frames = args.frames
    n = len(frames)
    two_rows = base is not None
    nrow = 2 if two_rows else 1

    # 圖寬以雙欄半頁為基準（~3.3in/欄 → 雙欄 ~7in），高度依列數
    fig_w = min(7.0, 1.55 * n)
    fig_h = 2.0 * nrow + 0.55
    fig, axes = plt.subplots(nrow, n, figsize=(fig_w, fig_h), squeeze=False)

    lost_set = set(args.lost_frames or [])
    recover_f = args.recover_frame

    # ── 第一列：Ours（GLRA） ──
    for j, fid in enumerate(frames):
        ax = axes[0][j]
        rec = ours.get(fid, {}).get(args.track_id)
        if rec is None and fid in lost_set:
            # lost 空窗期：Ours 結果檔通常不輸出 lost track，用 GPR 預測示意
            gpr_box = _interp_box(ours, args.track_id, fid, frames)
            panel(ax, args.seq_dir, fid, None, "lost", args.track_id, gpr_box=gpr_box)
            continue
        if rec is None:
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            ax.text(
                0.5,
                0.5,
                f"frame {fid}\n(no track)",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=8,
            )
            continue
        box = rec[:4]
        if fid == recover_f:
            det_box = (
                _interp_box(ours, args.track_id, fid, frames, prefer="next") or box
            )
            gpr_box = _interp_box(ours, args.track_id, fid, frames, prefer="prev")
            panel(
                ax,
                args.seq_dir,
                fid,
                box,
                "recover",
                args.track_id,
                gpr_box=gpr_box,
                det_box=box,
                det_score=rec[4],
            )
        elif fid in lost_set:
            panel(ax, args.seq_dir, fid, None, "lost", args.track_id, gpr_box=box)
        else:
            panel(ax, args.seq_dir, fid, box, "active", args.track_id)

    # ── 第二列：Baseline（無 GLRA，發生 ID switch）──
    if two_rows:
        for j, fid in enumerate(frames):
            ax = axes[1][j]
            switched = recover_f is not None and fid >= recover_f
            tid = args.base_id_post if switched else args.base_id_pre
            rec = base.get(fid, {}).get(tid)
            if rec is None:
                # 試另一個 ID
                alt = args.base_id_pre if switched else args.base_id_post
                rec = base.get(fid, {}).get(alt)
                tid = alt
            if rec is None:
                ax.set_xticks([])
                ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)
                ax.text(
                    0.5,
                    0.5,
                    f"frame {fid}\n(lost)",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=8,
                )
                continue
            baseline_panel(
                ax,
                args.seq_dir,
                fid,
                rec[:4],
                tid,
                is_switch=(tid == args.base_id_post and switched),
            )

    # ── 列標題（左側）──
    axes[0][0].set_ylabel(
        "Ours (+GLRA)", fontsize=9, weight="bold", rotation=90, labelpad=8
    )
    axes[0][0].yaxis.set_label_coords(-0.04, 0.5)
    if two_rows:
        axes[1][0].set_ylabel(
            "Baseline", fontsize=9, weight="bold", rotation=90, labelpad=8
        )
        axes[1][0].yaxis.set_label_coords(-0.04, 0.5)

    # ── 圖例 ──
    handles = [
        Line2D([0], [0], color=C_ACTIVE, lw=2.4, label="Tracked (ID kept)"),
        Line2D(
            [0],
            [0],
            color=C_LOST,
            lw=2.0,
            ls=(0, (4, 3)),
            label="GPR prediction (lost)",
        ),
        Line2D(
            [0], [0], color=C_DET, lw=1.8, ls=(0, (4, 3)), label="Low-score detection"
        ),
    ]
    if two_rows:
        handles.append(
            Line2D([0], [0], color=C_SWITCH, lw=2.2, label="New ID (ID switch)")
        )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        fontsize=7.5,
        bbox_to_anchor=(0.5, -0.01),
    )

    fig.tight_layout(rect=[0.02, 0.06, 1, 1])
    fig.subplots_adjust(wspace=0.06, hspace=0.18)

    pdf = args.out + ".pdf"
    png = args.out + ".png"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"[DONE] 已輸出:\n  {pdf}  (向量，放 LaTeX 用這個)\n  {png}  (300 DPI 預覽)")


def _interp_box(data, tid, fid, frames, prefer="both"):
    """空窗期沒有框時，用前/後最近的已知框做線性內插，模擬 GPR 外推位置。"""
    known = sorted(f for f in data if tid in data[f])
    prev = max((f for f in known if f < fid), default=None)
    nxt = min((f for f in known if f > fid), default=None)
    if prefer == "prev" and prev is not None:
        return list(data[prev][tid][:4])
    if prefer == "next" and nxt is not None:
        return list(data[nxt][tid][:4])
    if prev is not None and nxt is not None:
        a = data[prev][tid][:4]
        b = data[nxt][tid][:4]
        t = (fid - prev) / (nxt - prev)
        return [a[i] + t * (b[i] - a[i]) for i in range(4)]
    if prev is not None:
        return list(data[prev][tid][:4])
    if nxt is not None:
        return list(data[nxt][tid][:4])
    return None


def parse_args():
    p = argparse.ArgumentParser(
        description="GLRA recovery teaser figure for paper page 1"
    )
    p.add_argument("--seq_dir", required=True, help="序列影像資料夾 (img1)")
    p.add_argument("--ours_txt", required=True, help="Ours(+GLRA) 的 MOT 結果檔")
    p.add_argument(
        "--base_txt", default=None, help="baseline 結果檔（可選，畫第二列對照）"
    )
    p.add_argument("--track_id", type=int, required=True, help="Ours 中要追蹤的 ID")
    p.add_argument(
        "--frames",
        type=int,
        nargs="+",
        required=True,
        help="要顯示的幀號，例如 300 312 324 336 348",
    )
    p.add_argument(
        "--lost_frames",
        type=int,
        nargs="*",
        default=[],
        help="這些幀為 lost/GPR 預測狀態（畫橘虛線框）",
    )
    p.add_argument("--recover_frame", type=int, default=None, help="GLRA 接回的那一幀")
    p.add_argument("--base_id_pre", type=int, default=None, help="baseline 接回前的 ID")
    p.add_argument(
        "--base_id_post",
        type=int,
        default=None,
        help="baseline 接回後換掉的新 ID（ID switch）",
    )
    p.add_argument("--out", default="fig1_glra_teaser", help="輸出檔名（不含副檔名）")
    return p.parse_args()


if __name__ == "__main__":
    build_figure(parse_args())
