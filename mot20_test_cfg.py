from detectron2.config import LazyCall as L
from omegaconf import OmegaConf
from .datasets.builder import build_test_loader
from .models.model_utils import get_model

# build dataloader
dataloader = OmegaConf.create()
dataloader.test = L(build_test_loader)(
    test_size=(896, 1600),  # (736, 1920)
    infer_batch=1,  # for tracking process frame by frame
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
    output_dir="./yolox_mix20",
    init_checkpoint="/home/caig/pretrain/bytetrack_x_mot20.tar",
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

# build tracker. For mot20, dancetrack -- unenabled GMC: 368 - 373 in sparse_tracker.py
track = dict(
    experiment_name="yolox_mix20_det",
    folder_name="track_results_test",
    track_thresh=0.6,
    track_buffer=60,
    match_thresh=0.6,
    min_box_area=100,
    down_scale=4,
    depth_levels=1,
    depth_levels_low=8,
    confirm_thresh=0.7,
    use_diou=True,  # is diou matching
    use_gmc_history=False,
    use_glra=True,  # 開關
    gpr_min_lost=1,
    gpr_max_lost=30,  # 10→8，密集場景長 lost 配對品質掉很快
    gpr_min_obs=15,  # 保留
    gpr_history_len=60,  # 保留
    glra_thresh=0.2,  # 0.45→0.38，這是主要槓桿
    glra_sigma_cap=6,  # 10→6，MOT20 下作用有限但無害，可作保險絲；若想完全交給 cost 也可保留 10
    glra_confirm=False,  # 是否要等 GLRA 配對成功才正式把 track 加回 tracked_stracks（設 False 可還原舊行為做 ablation）
    glra_confirm_thresh=0.55,
    # GLRA adaptive threshold
    glra_adaptive=False,  # 設 False 可還原舊行為做 ablation
    glra_sigma_scale=80.0,  # 解析度大，σ 要到 80px 才算真的不確定
    glra_thresh_range=0.10,  # 最多放寬到 0.55
    glra_max_thresh=0.55,  # 與 0.45+0.10 對齊，不留額外空間
    # is fuse scores
    mot20=True,
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
    val_ann="test.json",
    # is public dets using
    is_public=False,
)
