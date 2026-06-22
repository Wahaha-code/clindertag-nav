import os
from typing import Optional, Sequence

import cv2
import numpy as np


def _col_color(col_idx: int, total_cols: int):
    hsv = np.uint8([[[int(180.0 * col_idx / max(1, total_cols - 1)), 200, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_columns_with_ids(
    image_bgr: np.ndarray,
    corners_array: Sequence[Sequence],
    n_cols: int,
    out_path: Optional[str] = None,
):
    """Render CylinderTag corner annotations in the same style as the offline script."""
    if corners_array is None or len(corners_array) == 0:
        return None

    vis = image_bgr.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    ca = np.asarray(corners_array, dtype=object)

    col_arr = np.asarray(ca[:, 2]).astype(int)
    row_arr = np.asarray(ca[:, 3]).astype(int)
    kid_arr = np.asarray(ca[:, 4]).astype(int)

    for c in range(n_cols):
        col_color = _col_color(c, n_cols)

        idx_top = np.where((col_arr == c) & (row_arr == 0))[0]
        idx_bot = np.where((col_arr == c) & (row_arr == 1))[0]

        if idx_top.size == 4:
            order_top = idx_top[np.argsort(kid_arr[idx_top])]
            poly_top = ca[order_top][:, :2].astype(float).astype(int)
            cv2.polylines(vis, [poly_top.reshape(-1, 1, 2)], True, col_color, 2, cv2.LINE_AA)

        if idx_bot.size == 4:
            order_bot = idx_bot[np.argsort(kid_arr[idx_bot])]
            poly_bot = ca[order_bot][:, :2].astype(float).astype(int)
            cv2.polylines(vis, [poly_bot.reshape(-1, 1, 2)], True, col_color, 2, cv2.LINE_AA)

        idx_in_col = np.where(col_arr == c)[0]
        for idx in idx_in_col:
            x, y = float(ca[idx, 0]), float(ca[idx, 1])
            r = int(row_arr[idx])
            k = int(kid_arr[idx])
            cv2.circle(vis, (int(x), int(y)), 3, col_color, -1, cv2.LINE_AA)
            txt = f"{idx}|c{c} r{r} k{k}"
            cv2.putText(vis, txt, (int(x) + 3, int(y) - 3), font, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(vis, txt, (int(x) + 3, int(y) - 3), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        if idx_top.size == 4:
            tl = ca[idx_top[np.argmin(kid_arr[idx_top])], :2].astype(float).astype(int)
            cv2.putText(vis, f"Column {c}", (int(tl[0]) - 5, int(tl[1]) - 8), font, 0.55, col_color, 2, cv2.LINE_AA)

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        cv2.imwrite(out_path, vis)

    return vis
