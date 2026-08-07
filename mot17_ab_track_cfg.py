from detectron2.config import LazyCall as L
from omegaconf import OmegaConf
from .datasets.builder import build_test_loader
from .models.model_utils import get_model

# if do ablation please invalidate the specific thresh settings from 124 - 138 in evaluators.py

# build dataloader
dataloader = OmegaConf.create()
dataloader.test = L(build_test_loader)(
    test_size=(800, 1440), infer_batch=1  # for tracking process frame by frame
)

# build model
model = L(get_model)(
    model_type="yolox",
    depth=1.33,
    width=1.25,
    num_classes=1,
    confthre=0.01,
    nmsthre=0.7,
)

# build train cfg
train = dict(
    output_dir="./yolox_mix17_ablation",
    init_checkpoint="/home/caig/pretrain/bytetrack_ablation.pth.tar",
    # model ema
    model_ema=dict(
        enabled=False,
        use_ema_weights_for_eval_only=False,
        decay=0.9998,
        device="cuda",
        after_backward=False,
    ),
    device="cuda",
)

# build tracker
track = dict(
    experiment_name="yolox_mix17_ablation_det",
    # tracking settings
    track_thresh=0.6,
    track_buffer=30,
    match_thresh=0.85,
    min_box_area=100,
    down_scale=4,
    depth_levels=1,
    depth_levels_low=8,
    confirm_thresh=0.7,
    # ── Association metric ablation ───────────────────────────────────────
    # iou_type 一次決定所有關聯階段（DCM high/low、unconfirmed、GLRA）用的
    # 度量，跑 ablation 只改這一行:
    #   "iou"  = SparseTrack baseline
    #   "giou" = 外接框面積懲罰 (Rezatofighi et al., CVPR 2019)
    #   "diou" = 中心距離懲罰（本論文採用）
    #   "ciou" = DIoU + 長寬比一致性 (Zheng et al., AAAI 2020)
    iou_type="diou",
    use_diou=False,  # legacy 開關，iou_type 有設定時會被忽略；留著讓舊 run 可重現
    # is fuse scores
    use_gmc_history=False,
    use_glra=True,  # 開關
    gpr_min_lost=1,  # GPR 最多能恢復幾幀丟失的軌跡
    gpr_max_lost=30,  # GPR 最多能恢復幾幀丟失的軌跡
    gpr_min_obs=5,  # GPR 最少需要幾筆觀測
    gpr_history_len=60,  # 歷史長度
    glra_thresh=0.1,  # 配對 cost threshold（1 - DIoU）
    glra_sigma_cap=20,  # GLRA sigma 超過多少 px 就不配對了（設 None 可還原舊行為做 ablation）
    glra_confirm=False,  # 是否要等 GLRA 配對成功才正式把 track 加回 tracked_stracks（設 False 可還原舊行為做 ablation）
    glra_confirm_thresh=0.55,  # 一致性檢查門檻(1 - DIoU)
    glra_confirm_grace=1,  # 未匹配的寬限幀數;診斷後若 unmatched 為主因改成 1
    glra_frag_gate=False,  # 碎片排除
    glra_frag_contain=0.7,  # 包含度門檻
    glra_height_gate=False,  # 高度一致性
    glra_height_tol=0.3,  # 高度容忍比例
    # GLRA adaptive threshold
    glra_adaptive=False,  # 設 False 可還原舊行為做 ablation
    glra_sigma_scale=30.0,  # σ 超過 30px → 拿到完整 thresh_range 加分
    glra_thresh_range=0.25,  # 最多放寬 0.25
    glra_max_thresh=0.85,  # 絕對上限
    glra_diag=False,
    glra_diag_path="./glra_diag.csv",  # 跨序列共用一個檔,append
    glra_dump=False,
    glra_dump_dir="./glra_cases",
    mot20=False,
    # trackers
    byte=False,
    deep=True,
    bot=False,
    sort=False,
    ocsort=False,
    # detector model settings
    fp16=True,
    fuse=True,
    # val json
    val_ann="val_half.json",
    # is public dets using
    is_public=False,
)
