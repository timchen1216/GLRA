r"""
glra_eval_recoveries.py  ──  把 GLRA diag CSV 跟 GT 比對,篩出「真正救對」的案例
================================================================================
背景:
  sparse_tracker.py 在 `--glra_diag` 開啟時,每次「成功接回(committed recovery)」
  會往一個 CSV (glra_diag.csv) append 一列,內含:
      seq, frame, track_id, lost_duration, pred_*, det_*, det_score, cost, ...
  這就是完整的 GLRA recovery event log。本程式做兩件事:

  (1) 用 GT 標註每個 event 是否「救對」:
        • det_gt_id   : 接回那一幀,跟 matched detection 最高 IoU 的 GT id
        • prior_gt_id : track 進入 lost 之前,它對應的 GT id
        • correct     : prior_gt_id == det_gt_id 且 IoU 都過門檻 → 身分維持正確
  (2) 輸出排序後的候選清單 (best recoveries),欄位包含畫 teaser 需要的所有資訊,
      可直接挑一筆餵給 glra_teaser.py。

位移(displacement)分析 ── 用來區分「真外推」vs「原地複製」:
  • displacement : prior(lost前最後位置)中心 → det(接回位置)中心的 pixel 距離
  • disp_norm    : displacement / prior_box_height(用框高正規化,消除遠近偏差)
  靜止目標短暫漏檢的接回,disp_norm 趨近 0(GPR 形同 copy-paste 上一幀);
  移動目標被遮擋後在他處重現的接回,disp_norm 大 → 才能展示 GPR 外推價值。
  用 `--sort_by displacement` 可優先撈出後者作為 teaser。

「track 進入 lost 之前對應的 GT id」怎麼來?
  我們需要 tracker 的「結果檔 (ours .txt)」:在 track 還沒 lost 前,它在某些幀
  有輸出框。我們取 recovery frame 之前、該 track_id 最後一次出現的那一幀的框,
  跟 GT 比對得 prior_gt_id。(只用 diag CSV 不夠,因為 lost 前的軌跡在 GT 端的
  身分要靠框去配。)

--------------------------------------------------------------------------------
用法:
    python glra_eval_recoveries.py \
        --diag_csv   ./glra_diag.csv \
        --gt_root    /path/to/MOT17/train \
        --res_dir    /path/to/track_results \
        --iou_thr    0.5 \
        --out_csv    glra_recoveries_labeled.csv \
        [--sort_by displacement]      # lost(預設) | displacement | score
        [--teaser_cmds 5]             # 印幾筆 teaser 指令範本
        [--seq MOT17-04-FRCNN]        # 只看單一序列(可選)

目錄假設(MOTChallenge 標準):
    gt_root/<seq>/gt/gt.txt
    res_dir/<seq>.txt                 # tracker 輸出(ours,+GLRA)

輸出:
    glra_recoveries_labeled.csv  ── 每個 event 一列,加上 det_gt_id / prior_gt_id /
                                    det_iou / displacement / disp_norm / correct 等欄位
    並在 stdout 印出「最佳 N 筆正確接回」與對應的 glra_teaser.py 指令範本。
================================================================================
"""

import argparse
import csv
import os
from collections import defaultdict


# ── IoU ──────────────────────────────────────────────────────────────────
def iou_xywh(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def center(box_xywh):
    x, y, w, h = box_xywh
    return (x + w / 2.0, y + h / 2.0)


# ── 讀取 ─────────────────────────────────────────────────────────────────
def load_gt(gt_path):
    """MOT GT: frame,id,x,y,w,h,conf,class,vis  → {frame: [(id,(x,y,w,h),vis), ...]}
    只保留 person class (1或7) 且 conf!=0 的條目(MOTChallenge 慣例)。"""
    gt = defaultdict(list)
    with open(gt_path) as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 6:
                continue
            fid, tid = int(float(p[0])), int(float(p[1]))
            x, y, w, h = map(float, p[2:6])
            conf = float(p[6]) if len(p) > 6 else 1.0
            cls = int(float(p[7])) if len(p) > 7 else 1
            vis = float(p[8]) if len(p) > 8 else 1.0
            if conf == 0:
                continue
            if len(p) > 7 and cls not in (1, 2, 7):  # pedestrian-ish
                continue
            gt[fid].append((tid, (x, y, w, h), vis))
    return gt


def load_res(res_path):
    """tracker 結果: {frame: {track_id: (x,y,w,h,score)}}"""
    res = defaultdict(dict)
    if not os.path.isfile(res_path):
        return res
    with open(res_path) as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 6:
                continue
            fid, tid = int(float(p[0])), int(float(p[1]))
            x, y, w, h = map(float, p[2:6])
            s = float(p[6]) if len(p) > 6 and p[6] not in ("", "-1") else 1.0
            res[fid][tid] = (x, y, w, h, s)
    return res


def best_gt_match(box_xywh, gt_frame, iou_thr):
    """回傳 (gt_id, iou) 為跟 box 最高 IoU 的 GT;若無達門檻回 (None, best_iou)。"""
    best_id, best_iou = None, 0.0
    for gid, gbox, _vis in gt_frame:
        v = iou_xywh(box_xywh, gbox)
        if v > best_iou:
            best_iou, best_id = v, gid
    if best_iou >= iou_thr:
        return best_id, best_iou
    return None, best_iou


# ── 主流程 ───────────────────────────────────────────────────────────────
def evaluate(args):
    # 讀 diag CSV
    events = []
    with open(args.diag_csv) as f:
        for row in csv.DictReader(f):
            events.append(row)
    if not events:
        print("[WARN] diag CSV 是空的,沒有任何 GLRA recovery event。")
        return

    if args.seq:
        events = [e for e in events if e["seq"] == args.seq]

    # 依序列分組,各序列只讀一次 GT / res
    by_seq = defaultdict(list)
    for e in events:
        by_seq[e["seq"]].append(e)

    out_rows = []
    for seq, evs in sorted(by_seq.items()):
        gt_path = os.path.join(args.gt_root, seq, "gt", "gt.txt")
        res_path = os.path.join(args.res_dir, f"{seq}.txt")
        if not os.path.isfile(gt_path):
            print(f"[WARN] 找不到 GT: {gt_path},跳過序列 {seq}")
            continue
        gt = load_gt(gt_path)
        res = load_res(res_path)

        for e in evs:
            frame = int(float(e["frame"]))
            tid = int(float(e["track_id"]))
            lost_dur = int(float(e.get("lost_duration", 0)))
            det_box = (
                float(e["det_cx"]) - float(e["det_w"]) / 2,
                float(e["det_cy"]) - float(e["det_h"]) / 2,
                float(e["det_w"]),
                float(e["det_h"]),
            )
            # (1) 接回幀:matched det vs GT
            det_gt_id, det_iou = best_gt_match(det_box, gt.get(frame, []), args.iou_thr)

            # (2) lost 之前:該 track_id 在結果檔最後一次出現的框 vs GT
            prior_gt_id, prior_iou, prior_frame = None, 0.0, None
            prior_box = None
            for pf in range(frame - 1, max(0, frame - lost_dur - 60), -1):
                if pf in res and tid in res[pf]:
                    prior_box = res[pf][tid][:4]
                    prior_gt_id, prior_iou = best_gt_match(
                        prior_box, gt.get(pf, []), args.iou_thr
                    )
                    prior_frame = pf
                    break

            correct = (
                det_gt_id is not None
                and prior_gt_id is not None
                and det_gt_id == prior_gt_id
            )
            recovered_real = det_gt_id is not None

            # (3) 位移分析:prior 中心 → det 中心
            displacement = -1.0
            disp_norm = -1.0
            if prior_box is not None:
                pcx, pcy = center(prior_box)
                dcx, dcy = center(det_box)
                displacement = ((dcx - pcx) ** 2 + (dcy - pcy) ** 2) ** 0.5
                prior_h = max(prior_box[3], 1.0)
                disp_norm = displacement / prior_h

            out_rows.append(
                {
                    "seq": seq,
                    "frame": frame,
                    "track_id": tid,
                    "lost_duration": lost_dur,
                    "det_score": float(e.get("det_score", 0.0)),
                    "cost": float(e.get("cost", 0.0)),
                    "gpr_sigma_px": float(e.get("gpr_sigma_px", 0.0)),
                    "det_iou": round(det_iou, 3),
                    "det_gt_id": det_gt_id if det_gt_id is not None else -1,
                    "prior_frame": prior_frame if prior_frame is not None else -1,
                    "prior_iou": round(prior_iou, 3),
                    "prior_gt_id": prior_gt_id if prior_gt_id is not None else -1,
                    "displacement": round(displacement, 1),
                    "disp_norm": round(disp_norm, 3),
                    "correct": int(correct),
                    "recovered_real": int(recovered_real),
                }
            )

    if not out_rows:
        print("[WARN] 沒有可比對的 event。")
        return

    # 寫出完整標註
    fields = list(out_rows[0].keys())
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    # 統計
    n = len(out_rows)
    n_correct = sum(r["correct"] for r in out_rows)
    n_real = sum(r["recovered_real"] for r in out_rows)
    print(f"\n[統計] 共 {n} 個 GLRA recovery events")
    print(f"  身分維持正確 (correct)      : {n_correct}  ({100*n_correct/n:.1f}%)")
    print(f"  接回到真實目標 (recovered_real): {n_real}  ({100*n_real/n:.1f}%)")
    print(f"  → 完整標註已寫到: {args.out_csv}")

    # 位移分布(只看 correct=1 且 disp 可計算的)──幫你判斷「真外推 vs 原地複製」
    movers = [r for r in out_rows if r["correct"] == 1 and r["disp_norm"] >= 0]
    if movers:
        dn = sorted(r["disp_norm"] for r in movers)
        n_static = sum(1 for v in dn if v < 0.3)  # <0.3倍框高 ≈ 幾乎沒動
        n_move = sum(1 for v in dn if v >= 0.5)  # >=0.5倍框高 ≈ 明顯移動
        med = dn[len(dn) // 2]
        print(f"\n[位移分布 — correct=1, 共 {len(movers)} 筆]")
        print(f"  disp_norm 中位數: {med:.2f} (倍框高)")
        print(
            f"  幾乎沒動 (disp_norm<0.3): {n_static}  "
            f"({100*n_static/len(movers):.0f}%)  ← GPR 形同複製上一幀"
        )
        print(
            f"  明顯移動 (disp_norm>=0.5): {n_move}  "
            f"({100*n_move/len(movers):.0f}%)  ← 能展示 GPR 外推價值"
        )

    # ── 排序準則 ──
    cands = [r for r in out_rows if r["correct"] == 1]
    if args.sort_by == "displacement":
        # 優先「移動明顯且接對」:disp_norm 大 → det_iou 高 → lost 久
        # disp_norm<0 (prior 配不到) 自動沉底
        cands = [r for r in cands if r["disp_norm"] >= 0]
        cands.sort(
            key=lambda r: (r["disp_norm"], r["det_iou"], r["lost_duration"]),
            reverse=True,
        )
        sort_desc = "位移大(disp_norm) -> det_iou -> lost"
    elif args.sort_by == "score":
        # 優先 low-score recovery:det_score 低 → det_iou 高 → lost 久
        cands.sort(
            key=lambda r: (-r["det_score"], r["det_iou"], r["lost_duration"]),
            reverse=True,
        )
        sort_desc = "低分接回(det_score低) -> det_iou -> lost"
    else:  # "lost"(原行為)
        cands.sort(
            key=lambda r: (r["lost_duration"], r["det_iou"], -r["det_score"]),
            reverse=True,
        )
        sort_desc = "lost久 -> det_iou -> 低分"

    topn = cands[: args.top]
    print(f"\n[最佳 {len(topn)} 筆正確接回案例 — 排序依據: {sort_desc}]")
    print(
        f"{'seq':<18}{'recover_f':>10}{'tid':>6}{'lost':>6}"
        f"{'disp_n':>8}{'det_iou':>9}{'score':>7}{'gt_id':>7}"
    )
    for r in topn:
        print(
            f"{r['seq']:<18}{r['frame']:>10}{r['track_id']:>6}"
            f"{r['lost_duration']:>6}{r['disp_norm']:>8.2f}{r['det_iou']:>9.2f}"
            f"{r['det_score']:>7.2f}{r['det_gt_id']:>7}"
        )

    # 為前 N 筆印出 teaser 指令範本
    n_cmds = min(args.teaser_cmds, len(cands))
    if n_cmds > 0:
        print(f"\n[teaser 指令範本 — 前 {n_cmds} 筆最佳案例]")
        for rank, b in enumerate(cands[:n_cmds], 1):
            rf = b["frame"]
            ld = b["lost_duration"]
            # 平均挑 5 格:lost 前一幀、空窗期幾幀、接回幀
            f0 = max(1, rf - ld - 3)
            gap = max(1, ld // 2)
            frames = sorted(set([f0, rf - ld, rf - gap, rf - 1, rf]))
            lost_frames = [x for x in frames if (rf - ld) <= x < rf]
            print(
                f"\n# [{rank}] {b['seq']}  tid={b['track_id']}  "
                f"recover_f={rf}  lost={ld}  disp_norm={b['disp_norm']:.2f}  "
                f"det_iou={b['det_iou']:.2f}  score={b['det_score']:.2f}  "
                f"gt_id={b['det_gt_id']}"
            )
            print(f"python glra_teaser.py \\")
            print(f"  --seq_dir   {os.path.join(args.gt_root, b['seq'], 'img1')} \\")
            print(f"  --ours_txt  {os.path.join(args.res_dir, b['seq'] + '.txt')} \\")
            print(f"  --track_id  {b['track_id']} \\")
            print(f"  --frames    {' '.join(map(str, frames))} \\")
            print(f"  --lost_frames {' '.join(map(str, lost_frames))} \\")
            print(f"  --recover_frame {rf} \\")
            print(f"  --out       fig1_glra_{b['seq']}_{b['track_id']}")


def parse_args():
    p = argparse.ArgumentParser(description="Label GLRA recovery events against GT")
    p.add_argument(
        "--diag_csv", required=True, help="sparse_tracker 產生的 glra_diag.csv"
    )
    p.add_argument(
        "--gt_root", required=True, help="GT 根目錄,內含 <seq>/gt/gt.txt 與 <seq>/img1"
    )
    p.add_argument("--res_dir", required=True, help="tracker 結果目錄,內含 <seq>.txt")
    p.add_argument("--iou_thr", type=float, default=0.5, help="判定配對的 IoU 門檻")
    p.add_argument("--out_csv", default="glra_recoveries_labeled.csv")
    p.add_argument("--seq", default=None, help="只處理單一序列(可選)")
    p.add_argument("--top", type=int, default=10, help="印出前 N 筆最佳案例(摘要表)")
    p.add_argument(
        "--teaser_cmds",
        type=int,
        default=5,
        help="印出前 N 筆的 glra_teaser.py 指令範本",
    )
    p.add_argument(
        "--sort_by",
        default="lost",
        choices=["lost", "displacement", "score"],
        help="最佳案例排序依據:lost(預設) | displacement(撈移動案例) | score(撈低分接回)",
    )
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())


"""
python glra_eval_recoveries.py \
  --diag_csv ./glra_diag.csv \
  --gt_root  /home/caig/data/MOT17/train \
  --res_dir  /home/caig/repo/SparseTrack/yolox_mix17_ablation/yolox_mix17_ablation_det/track_results \
  --out_csv  glra_recoveries_labeled.csv \
  --sort_by  displacement \
  --teaser_cmds 5

python glra_eval_recoveries.py \
  --diag_csv ./glra_diag.csv \
  --gt_root  /home/caig/data/MOT20/train \
  --res_dir  /home/caig/repo/SparseTrack/yolox_mix20_ablation/yolox_mix20_ablation_det/track_results \
  --out_csv  glra_recoveries_labeled.csv \
  --sort_by  displacement \
  --teaser_cmds 5

"""
