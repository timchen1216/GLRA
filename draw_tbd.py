import cv2


def color_for(tid):  # track_id → 固定 BGR 顏色
    return ((tid * 47) % 255, (tid * 97) % 255, (tid * 137) % 255)


def draw_boxes(img, boxes, mode):
    for b in boxes:
        x, y, w, h = map(int, b["tlwh"])
        if mode == "detect":
            color, label = (0, 220, 255), None  # 黃色 (BGR)，無 ID
        else:  # predict / update
            color, label = color_for(b["track_id"]), f"ID {b['track_id']}"
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        if label:
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (x, y - th - 6), (x + tw + 4, y), color, -1)
            cv2.putText(
                img,
                label,
                (x + 2, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
    return img


f = "MOT20-05/img1/000173.jpg"
cv2.imwrite("1_detect.png", draw_boxes(cv2.imread(f), detections_raw, "detect"))
cv2.imwrite("2_predict.png", draw_boxes(cv2.imread(f), kf_predictions, "predict"))
cv2.imwrite("4_update.png", draw_boxes(cv2.imread(f), updated_tracks, "update"))
