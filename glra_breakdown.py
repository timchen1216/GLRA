r"""
glra_breakdown.py  ──  拆解 GLRA recovery 的「不正確」案例(互斥分類修正版)
================================================================================
舊版的問題:類別 A(prior配不到)與 B(det接空氣)條件可同時成立,導致重複計數、
「其他」出現負數。本版改成「每個 wrong event 只歸一類」的互斥分類:

  correct : det_gt_id!=-1 且 prior_gt_id!=-1 且兩者相等  → 身分維持正確
  ── 以下為 correct=0,依優先序互斥歸類 ──
  B0  det接到空氣 : det_gt_id == -1
        → 接回的 detection 配不到任何 GT。可能是真誤檢,
          也可能是「GT 因嚴重遮擋未標註 / vis 過低被濾掉」的無辜案例。
  A0  prior配不到 : det_gt_id != -1 且 prior_gt_id == -1
        → det 接到真人,但 track 進 lost 前的框配不到 GT(多半進遮擋時框歪)。
          這類多半其實救對了,是比對腳本的限制,不是 GLRA 的錯。
  C0  接錯人      : det_gt_id != -1 且 prior_gt_id != -1 且兩者不等
        → GPR 把 A 的軌跡接到 B 身上。真 identity swap,最該修。

重要提醒:
  「det接到空氣(B0)」≠「GLRA 製造 FP」。MOT GT 會濾掉 conf=0 / 嚴重遮擋的條目,
  而 GLRA 救的正是被遮擋的目標 → 很可能接到「真人但 GT 沒標」。
  用 --keep_low_vis(保留低vis GT)+ --iou_thr 0.3 重跑,若 B0 大降,
  就證實 B0 多為 GT 標註限制造成的假誤檢,而非真 FP。
  最終 GLRA 是否有淨貢獻,以 TrackEval 的 IDF1/IDsw/FP/FN 開關對照為準。

用法:
    # 直接用既有 labeled csv 做互斥分類(快)
    python glra_breakdown.py --labeled glra_recoveries_labeled.csv

    # 重新用寬鬆條件比對(需要 diag/gt/res,較慢但能驗證 B0 真假)
    python glra_breakdown.py \
        --diag_csv ./glra_diag.csv \
        --gt_root  /path/to/MOT20/train \
        --res_dir  /path/to/track_results \
        --iou_thr  0.3 --keep_low_vis
================================================================================
"""

import argparse
import csv
import os
from collections import Counter, defaultdict


def iou_xywh(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def classify(rows):
    """互斥分類。rows 需含 correct/det_gt_id/prior_gt_id。回傳 dict of lists。"""

    def gi(r, k):
        return int(float(r[k]))

    out = {"correct": [], "B0": [], "A0": [], "C0": []}
    for r in rows:
        if gi(r, "correct") == 1:
            out["correct"].append(r)
        elif gi(r, "det_gt_id") == -1:
            out["B0"].append(r)
        elif gi(r, "prior_gt_id") == -1:
            out["A0"].append(r)
        else:
            out["C0"].append(r)  # det、prior 都有但不等
    return out


def report(rows, tag=""):
    n = len(rows)
    if n == 0:
        print("空。")
        return
    g = classify(rows)
    nc, nB, nA, nC = (len(g["correct"]), len(g["B0"]), len(g["A0"]), len(g["C0"]))
    assert nc + nB + nA + nC == n, "互斥分類總數不符(不該發生)"

    print(f"\n===== Breakdown {tag} (共 {n} events) =====")
    print(f"  correct 救對            : {nc:5d}  ({100*nc/n:5.1f}%)")
    print(
        f"  B0 det接到空氣          : {nB:5d}  ({100*nB/n:5.1f}%)  "
        f"<- 可能真誤檢,也可能GT沒標的遮擋目標"
    )
    print(
        f"  A0 prior配不到(det是真人): {nA:5d}  ({100*nA/n:5.1f}%)  "
        f"<- 多半其實救對(腳本低估)"
    )
    print(f"  C0 接錯人(真IDsw)        : {nC:5d}  ({100*nC/n:5.1f}%)  " f"<- 真錯,該壓")

    lo = 100 * nc / n
    hi = 100 * (nc + nA) / n
    print(f"\n  校正後成功率區間: [{lo:.1f}% , {hi:.1f}%]  (上界把 A0 也算對)")
    print(
        f"  真 identity swap (C0): {nC} ({100*nC/n:.1f}%)  <- 這個才是真接錯,通常很小"
    )

    if g["C0"]:
        cseq = Counter(r["seq"] for r in g["C0"])
        print(
            f"  接錯人(C0)集中序列: "
            + ", ".join(f"{s}x{c}" for s, c in cseq.most_common())
        )
    return g


def from_labeled(args):
    rows = list(csv.DictReader(open(args.labeled)))
    g = report(rows, tag="(from labeled csv)")
    if g and g["A0"]:
        print(f"\n[A0 案例 - det接到真人但prior配不到,值得再檢視]")
        print(
            f"{'seq':<18}{'recover_f':>10}{'tid':>6}{'lost':>6}{'det_iou':>9}{'det_gt':>8}"
        )
        A = sorted(
            g["A0"],
            key=lambda r: (int(float(r["lost_duration"])), float(r["det_iou"])),
            reverse=True,
        )
        for r in A[:15]:
            print(
                f"{r['seq']:<18}{int(float(r['frame'])):>10}"
                f"{int(float(r['track_id'])):>6}{int(float(r['lost_duration'])):>6}"
                f"{float(r['det_iou']):>9.2f}{int(float(r['det_gt_id'])):>8}"
            )


def load_gt(gt_path, keep_low_vis, vis_min):
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
            # conf=0 永遠排除:這是 MOT 的「忽略區域 / distractor」,
            # id 系統獨立且不一致,保留會污染身分比對(導致 correct 全0)。
            if conf == 0:
                continue
            # 只保留行人類別
            if len(p) > 7 and cls not in (1, 2, 7):
                continue
            # keep_low_vis 的真正意義:放寬 visibility 下限,
            # 把「被遮擋但仍是正常行人(conf=1)」的低 vis 條目納入比對。
            # 不開時用 vis_min;開時 vis 下限降為 0(全收 conf=1 行人)。
            eff_vis_min = 0.0 if keep_low_vis else vis_min
            if vis < eff_vis_min:
                continue
            gt[fid].append((tid, (x, y, w, h)))
    return gt


def load_res(res_path):
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
            res[fid][tid] = (x, y, w, h)
    return res


def best_gt(box, gtf, thr):
    bid, biou = None, 0.0
    for gid, gb in gtf:
        v = iou_xywh(box, gb)
        if v > biou:
            biou, bid = v, gid
    return (bid, biou) if biou >= thr else (None, biou)


def from_diag(args):
    events = list(csv.DictReader(open(args.diag_csv)))
    by_seq = defaultdict(list)
    for e in events:
        by_seq[e["seq"]].append(e)

    rows = []
    for seq, evs in sorted(by_seq.items()):
        gt_path = os.path.join(args.gt_root, seq, "gt", "gt.txt")
        res_path = os.path.join(args.res_dir, f"{seq}.txt")
        if not os.path.isfile(gt_path):
            print(f"[WARN] 找不到 GT: {gt_path}")
            continue
        gt = load_gt(gt_path, args.keep_low_vis, args.vis_min)
        res = load_res(res_path)
        for e in evs:
            frame = int(float(e["frame"]))
            tid = int(float(e["track_id"]))
            lost = int(float(e.get("lost_duration", 0)))
            det_box = (
                float(e["det_cx"]) - float(e["det_w"]) / 2,
                float(e["det_cy"]) - float(e["det_h"]) / 2,
                float(e["det_w"]),
                float(e["det_h"]),
            )
            dgid, diou = best_gt(det_box, gt.get(frame, []), args.iou_thr)
            pgid = None
            for pf in range(frame - 1, max(0, frame - lost - 60), -1):
                if pf in res and tid in res[pf]:
                    pgid, _ = best_gt(res[pf][tid][:4], gt.get(pf, []), args.iou_thr)
                    break
            correct = dgid is not None and pgid is not None and dgid == pgid
            rows.append(
                {
                    "seq": seq,
                    "frame": frame,
                    "track_id": tid,
                    "lost_duration": lost,
                    "det_iou": round(diou, 3),
                    "det_gt_id": dgid if dgid is not None else -1,
                    "prior_gt_id": pgid if pgid is not None else -1,
                    "correct": int(correct),
                }
            )

    tag = f"(寬鬆比對 iou={args.iou_thr}, keep_low_vis={args.keep_low_vis}, vis_min={args.vis_min})"
    report(rows, tag=tag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", default=None, help="既有 labeled csv(模式一)")
    ap.add_argument("--diag_csv", default=None, help="diag csv(模式二:寬鬆重比對)")
    ap.add_argument("--gt_root", default=None)
    ap.add_argument("--res_dir", default=None)
    ap.add_argument("--iou_thr", type=float, default=0.5)
    ap.add_argument(
        "--keep_low_vis",
        action="store_true",
        help="保留 conf=0 / 遮擋 GT 條目(測 B0 是否為被遮擋的真目標)",
    )
    ap.add_argument(
        "--vis_min", type=float, default=0.0, help="GT visibility 下限(預設0=全保留)"
    )
    args = ap.parse_args()

    if args.diag_csv:
        if not (args.gt_root and args.res_dir):
            ap.error("模式二需要 --gt_root 與 --res_dir")
        from_diag(args)
    elif args.labeled:
        from_labeled(args)
    else:
        ap.error("請提供 --labeled(模式一) 或 --diag_csv(模式二)")


if __name__ == "__main__":
    main()
