import pandas as pd


def load(path):
    df = pd.read_csv(path)
    df = df[df["seq"] != "COMBINED"]  # 去掉總和列
    return df.set_index("seq")


off = load("tracker/glra_off/pedestrian_detailed.csv")
on = load("tracker/glra_on/pedestrian_detailed.csv")

cols = ["HOTA___AUC", "AssA___AUC", "DetA___AUC", "CLR_FP", "CLR_FN", "IDSW"]
diff = (on[cols] - off[cols]).round(3)
diff.columns = [c + "_Δ" for c in cols]

# 標記鏡頭型態
camera = {
    "MOT17-02": "static",
    "MOT17-04": "static",
    "MOT17-09": "static",
    "MOT17-05": "moving",
    "MOT17-10": "moving",
    "MOT17-11": "moving",
    "MOT17-13": "moving",
}
diff["camera"] = [
    camera.get(s.split("-FRCNN")[0].split("-SDP")[0].split("-DPM")[0], "?")
    for s in diff.index
]

print(diff.sort_values("camera"))
print("\n=== 依鏡頭型態彙總 ===")
print(diff.groupby("camera")[[c for c in diff.columns if c != "camera"]].sum().round(3))
