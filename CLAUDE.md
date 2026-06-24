# SparseTrack + GLRA — Multi-Object Tracking

碩士論文專案。基於 SparseTrack MOT framework，提出兩項增強:
1. **DCM 中以 DIoU 取代 vanilla IoU** 做 Depth Cascade Matching 關聯。
2. **GLRA (GPR-based Lost Track Re-Association)** — 一個 online 的遺失軌跡重關聯模組,使用低分偵測 (low-score detections) 做額外的 re-association,處理 DTI 因果性上無法處理的情況。

論文標題固定為:**"Multi-Object Tracking with Center-Aware Association and Lost Track Recovery"**。

## 環境

- GPU: **RTX 4060 Ti**(實驗用機器,不是 5070 Ti — 5070 Ti 是 ComfyUI 那台,與本專案無關)。
- 資料集: MOT17 / MOT20 / DanceTrack。
- 評估指標: HOTA 為主,搭配 IDF1、MOTA。

## 常用指令

> 啟動前請確認用的是本專案的 conda env,不要污染 base。

```bash
# (依你的實際指令補上,例如)
CUDA_VISIBLE_DEVICES=0 python3 track.py  --num-gpus 1  --config-file mot17_track_cfg.py # 跑追蹤
bash tools/eval_hota.sh                        # 評估
```

執行任何訓練/推論前,先確認 `cudnn.benchmark` 的設定:`True` 兩個資料集都會略升,但會引入非決定性。報告數字時需註明。

## 程式碼結構(重點檔案)

- `glra_distance.py` — GLRA 的核心距離/門檻邏輯,含 tiered threshold policy (Method B)。
- `interpolation.py` — 離線插值;glob 結果需 filter 成檔名以 `MOT17`/`MOT20` 開頭(已修過的 bug,不要回退)。
- `glra_interp_only.py` — 只做 GPR 插值、不做 re-association,用來跟 DTI 比較。
- `glra_offline.py` — full re-association 但無 tracker context,**負結果**,保留作為 motivate online-only 設計的證據。

## 已定案的決策 — 不要擅自更動

- **GPR kernel**: `DotProduct(sigma_0=0, fixed) + RBF + WhiteKernel`,**`optimizer=None`**(超參數固定,不做最佳化)。離線插值最佳設定 `rbf_ls=30.0, white_noise=0.05, n_history=30`。
- **DIoU clipping**: 在 code 中用 `np.clip(1.0 - _dious, 0.0, 1.0)`。
- **GLRA 不與 DTI 疊用**: GLRA+DTI 一致無法超越 DTI-only(FP cost 仍在、FN 好處被吸收)。論文一律在 **no-DTI** 設定下比較。
- **`use_gmc_history`**: 會傷害效能,維持關閉(GPR 線性核已隱含吸收平滑 ego-motion,顯式 history warping 會累積誤差)。
- `gpr_obs` 存的是 **raw detection positions**,不是 KF-smoothed(已修過的 bug)。


## 已知結果(別重跑來「驗證」,除非我要求)

- MOT17 最佳 GLRA 設定: `glra_min_lost=1, gpr_max_lost=8, gpr_min_obs=5, gpr_history_len=30, thresh=0.45`,HOTA ≈ 78.09。
- GLRA 在 **MOT17 增益邊際、MOT20 為正增益**(希望 MOT17 未來有機會超過)。
- MOT20 密集場景: **DIoU-based cost 是最強的區辨特徵**,建議 `glra_thresh` 收緊至 0.38。
- DanceTrack: GLRA 可分類回收中有 22.4% identity-swap;單一特徵門檻無法乾淨分離(對抗性設計)。Tiered policy (P5): `lost=1→0.45`, `lost=2-3→0.35`, `lost≥4→disabled`。
- 離線插值: 最佳 HOTA 78.459 vs DTI 78.483 — 差距在評估噪聲內。

## 工作方式

- 多檔重構或改關聯邏輯前,先進 **Plan Mode** 列步驟、等我核准再動手。
- 改動牽涉到「已定案決策」清單中的任何項目,先停下來向我確認,不要自行判斷。
