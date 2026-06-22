import argparse
import math
import numpy as np
import cv2
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2

def _col_color(c, ncols):
    hsv = np.uint8([[[int(180.0*c/max(1,ncols-1)), 200, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0,0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])

def draw_columns_with_ids(image_bgr, corners_array, n_cols, out_path):
    """
    corners_array: 每行 [x_px, y_px, col, row, corner_id]，dtype 可能是 object
    n_cols: 总列数
    返回：cv2.imwrite 的布尔值
    """
    if corners_array is None or len(corners_array) == 0:
        print("[BYCOL] corners_array is empty")
        return False

    vis = image_bgr.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    ca = np.asarray(corners_array, dtype=object)

    # 先把 col、row、k 安全转成整型数组，避免 object 比较踩坑
    col_arr = np.asarray(ca[:,2]).astype(int)
    row_arr = np.asarray(ca[:,3]).astype(int)
    kid_arr = np.asarray(ca[:,4]).astype(int)

    for c in range(n_cols):
        col_color = _col_color(c, n_cols)

        # 该列的上/下条索引
        idx_top = np.where((col_arr==c) & (row_arr==0))[0]
        idx_bot = np.where((col_arr==c) & (row_arr==1))[0]

        # 画上条（按 corner_id 排序 TL,TR,BR,BL）
        if idx_top.size == 4:
            order_top = idx_top[np.argsort(kid_arr[idx_top])]
            poly_top = ca[order_top][:, :2].astype(float).astype(int)
            cv2.polylines(vis, [poly_top.reshape(-1,1,2)], True, col_color, 2, cv2.LINE_AA)

        # 画下条
        if idx_bot.size == 4:
            order_bot = idx_bot[np.argsort(kid_arr[idx_bot])]
            poly_bot = ca[order_bot][:, :2].astype(float).astype(int)
            cv2.polylines(vis, [poly_bot.reshape(-1,1,2)], True, col_color, 2, cv2.LINE_AA)

        # 标注每个角点：idx|c r k
        idx_in_col = np.where(col_arr==c)[0]
        for idx in idx_in_col:
            x, y = float(ca[idx,0]), float(ca[idx,1])
            r = int(row_arr[idx]); k = int(kid_arr[idx])
            cv2.circle(vis, (int(x),int(y)), 3, col_color, -1, cv2.LINE_AA)
            txt = f"{idx}|c{c} r{r} k{k}"
            cv2.putText(vis, txt, (int(x)+3, int(y)-3),
                        font, 0.45, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(vis, txt, (int(x)+3, int(y)-3),
                        font, 0.45, (255,255,255), 1, cv2.LINE_AA)

        # 在该列左上附近写 Column c
        if idx_top.size == 4:
            tl = ca[idx_top[np.argmin(kid_arr[idx_top])], :2].astype(float).astype(int)  # k=0 位置
            cv2.putText(vis, f"Column {c}", (int(tl[0])-5, int(tl[1])-8),
                        font, 0.55, col_color, 2, cv2.LINE_AA)

    # 确保目录存在
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    ok = cv2.imwrite(out_path, vis)
    print(f"[BYCOL] saved: {out_path}  ok={ok}")
    return ok

# ------------------ 基础工具 ------------------

def order_quad_corners(pts):
    """将近似四边形的4点排序为 TL, TR, BR, BL"""
    pts = np.array(pts, dtype=np.float32).reshape(-1, 2)
    ys = pts[:, 1]
    idx = np.argsort(ys)
    top2 = pts[idx[:2]]
    bot2 = pts[idx[2:]]
    tl, tr = top2[np.argsort(top2[:, 0])]
    bl, br = bot2[np.argsort(bot2[:, 0])]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def preprocess(gray, invert=False):
    """对灰度图进行对比度增强 + 自适应阈值 + 形态学闭操作"""
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    eq = clahe.apply(gray)
    blur = cv2.GaussianBlur(eq, (3,3), 0)
    th = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 5
    )
    if invert:
        th = cv2.bitwise_not(th)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
    return th

def find_quads(binary, min_area=400, max_area_ratio=0.25):
    """在二值图中检出四边形轮廓"""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = binary.shape[:2]
    max_area = W * H * max_area_ratio
    quads = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        peri = cv2.arcLength(cnt, True)
        eps = 0.02 * peri
        approx = cv2.approxPolyDP(cnt, eps, True)
        if len(approx) != 4:
            eps2 = 0.04 * peri
            approx2 = cv2.approxPolyDP(cnt, eps2, True)
            if len(approx2) == 4:
                approx = approx2
            else:
                continue
        quad = approx.reshape(-1, 2).astype(np.float32)
        x, y, w, h = cv2.boundingRect(approx)
        quads.append({
            "quad": order_quad_corners(quad),
            "bbox": (x, y, w, h),
            "area": float(area),
            "cx": float(x + w / 2.0),
            "cy": float(y + h / 2.0),
        })
    return quads

def group_quads_into_grid(quads, n_cols=12, n_rows=2):
    """
    将所有四边形按 x 聚类成 n_cols 列，再按 y 排序取每列的 n_rows 个。
    聚类标签按列中心从左到右重映射，确保 col=0..n_cols-1 为左→右。
    """
    quads_sorted = sorted(quads, key=lambda q: q["cx"])
    if len(quads_sorted) < n_cols * n_rows:
        return None
    xs = np.array([q["cx"] for q in quads_sorted], dtype=np.float32).reshape(-1,1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
    _, labels, centers = cv2.kmeans(xs, n_cols, None, criteria, 5, cv2.KMEANS_PP_CENTERS)

    centers = centers.flatten()
    order = np.argsort(centers)
    old2new = {int(old): int(new) for new, old in enumerate(order)}

    cols = [[] for _ in range(n_cols)]
    for q, lab in zip(quads_sorted, labels.flatten().tolist()):
        cols[old2new[lab]].append(q)

    grid = []
    for c in range(n_cols):
        col = sorted(cols[c], key=lambda q: q["cy"])
        if len(col) < n_rows:
            return None
        grid.append(col[:n_rows])
    return grid

def detect_corners(image_bgr, cols=12, rows=2, min_area=400, max_area_ratio=0.25):
    """
    返回 corners 数组（每行 [x_px, y_px, col, row, corner_id]）和可视化图
    corner_id: 0=TL,1=TR,2=BR,3=BL
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    for invert in (False, True):
        th = preprocess(gray, invert=invert)
        quads = find_quads(th, min_area=min_area, max_area_ratio=max_area_ratio)
        grid = group_quads_into_grid(quads, n_cols=cols, n_rows=rows)
        if grid is not None:
            corners = []
            vis = image_bgr.copy()
            font = cv2.FONT_HERSHEY_SIMPLEX
            idx = 0
            for col_idx, col in enumerate(grid):               # col_idx 已保证左→右
                for row_idx, q in enumerate(col):              # 0=上带, 1=下带
                    pts = q["quad"].reshape(-1,1,2).astype(np.int32)
                    cv2.polylines(vis, [pts], True, (0,255,0), 2, cv2.LINE_AA)
                    ordered = order_quad_corners(q["quad"])
                    for k,(x,y) in enumerate(ordered):
                        corners.append([float(x), float(y), int(col_idx), int(row_idx), int(k)])
                        cv2.circle(vis, (int(x),int(y)), 3, (255,0,0), -1, cv2.LINE_AA)
                        cv2.putText(vis, f"{idx}", (int(x)+3, int(y)-3), font, 0.45, (0,0,255), 1, cv2.LINE_AA)
                        idx += 1
            return np.array(corners, dtype=object), vis
    return None, None

def reorder_per_column_clockwise(corners_array, n_cols=12):
    """
    目标列内顺序（共 8 点）：
      上半条：TL, TR, BR, BL   -> 索引 [0,1,2,3]
      下半条：BR, BL, TL, TR   -> 索引 [2,3,0,1] （★从右下角开始顺时针）
    """
    ca = np.asarray(corners_array, dtype=object)
    out_rows = []
    for c in range(n_cols):
        top = ca[(ca[:,2] == c) & (ca[:,3] == 0)]
        bot = ca[(ca[:,2] == c) & (ca[:,3] == 1)]
        if top.shape[0] != 4 or bot.shape[0] != 4:
            # 有缺列就按原状拼上，方便你排错
            out_rows.extend(list(top)); out_rows.extend(list(bot))
            continue

        # 先把两条都排成 TL,TR,BR,BL
        top = top[np.argsort(top[:,4].astype(int))]
        bot = bot[np.argsort(bot[:,4].astype(int))]

        # 上半条保持 TL,TR,BR,BL
        top_seq = top
        # 下半条改成 BR,BL,TL,TR（从右下角开始顺时针）
        bot_seq = bot[[2,3,0,1]]

        col8 = np.vstack([top_seq, bot_seq])
        out_rows.extend(list(col8))
    return np.array(out_rows, dtype=object)

# ------------------ 关键：像素 -> 圆柱面 3D(mm) ------------------

def pixels_to_cylinder_xyz(
    pts_px, img_w_px, img_h_px, diameter_mm,
    print_width_mm=None, wrap_ratio=None,
    print_height_mm=None,            # ★ 新增：明确打印高度（轴向）
    theta0_deg=0.0, z_origin='center',
    invert_theta=False               # ★ 新增：横向镜像（贴反时 True）
):
    """
    将像素坐标映射到圆柱表面3D坐标（单位 mm）。
    - print_width_mm：贴纸在圆周方向的实际宽度（没绕一圈也OK）
    - print_height_mm：贴纸轴向实际高度（不传则按像素比例推算）
    - wrap_ratio = print_width_mm / (π*diameter_mm)
    - theta0_deg：贴纸左边缘在圆周上的起始角（0°沿X轴，逆时针为正）
    - invert_theta：若整张左右颠倒贴，置 True
    """
    R = diameter_mm / 2.0
    C = math.pi * diameter_mm

    if wrap_ratio is None:
        if print_width_mm is None:
            print_width_mm = C
        wrap_ratio = print_width_mm / C
    else:
        if print_width_mm is None:
            print_width_mm = wrap_ratio * C

    if print_height_mm is None:
        print_height_mm = print_width_mm * (img_h_px / img_w_px)

    theta0 = math.radians(theta0_deg)

    pts_px = np.asarray(pts_px, dtype=float)
    x_px = pts_px[:, 0]; y_px = pts_px[:, 1]

    # 像素 → 贴纸坐标(mm)
    x_mm = x_px / img_w_px * print_width_mm
    z_mm = (1.0 - y_px / img_h_px) * print_height_mm

    if z_origin == 'center':
        z_mm -= print_height_mm / 2.0
    elif z_origin == 'bottom':
        z_mm -= print_height_mm
    elif z_origin == 'top':
        pass
    else:
        raise ValueError("z_origin must be 'center'|'top'|'bottom'")

    t = x_mm / print_width_mm  # 0..1
    if invert_theta:
        t = 1.0 - t

    theta = theta0 + 2.0 * math.pi * wrap_ratio * t
    X = R * np.cos(theta)
    Y = R * np.sin(theta)
    Z = z_mm
    return np.stack([X, Y, Z], axis=1)

# ------------------ I/O & 可视化 ------------------

def save_corners_csv(path, corners_array):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x_px","y_px","col","row","corner_id"])
        for r in corners_array:
            w.writerow(list(r))

def save_xyz_csv(path, xyz_array):
    with open(path, "w") as f:
        for i, r in enumerate(xyz_array, start=0):
            f.write(f"{i} {r[0]:.6f} {r[1]:.6f} {r[2]:.6f}\n")

def render_vis_from_corners(image_bgr, corners_array, out_path):
    vis = image_bgr.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for idx, r in enumerate(corners_array):
        x, y = float(r[0]), float(r[1])
        cv2.circle(vis, (int(x), int(y)), 3, (255,0,0), -1, cv2.LINE_AA)
        cv2.putText(vis, f"{idx}", (int(x)+3, int(y)-3), font, 0.45, (0,0,255), 1, cv2.LINE_AA)
    cv2.imwrite(out_path, vis)

def plot_xyz_png(path, xyz_array, title="CylinderTag 3D points"):
    xs, ys, zs = xyz_array[:,0], xyz_array[:,1], xyz_array[:,2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs, marker='o')
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title(title)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

# ------------------ 主流程 ------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to CylinderTag image (BMP/PNG/JPG)")
    ap.add_argument("--out_prefix", default="result", help="Output prefix")

    ap.add_argument("--cols", type=int, default=12)
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--img_w_px", type=int, default=1440)
    ap.add_argument("--img_h_px", type=int, default=1200)

    ap.add_argument("--diameter_mm", type=float, required=True, help="Cylinder diameter (mm)")
    ap.add_argument("--print_width_mm", type=float, default=None, help="Sticker width along circumference (mm)")
    ap.add_argument("--wrap_ratio", type=float, default=None, help="Alternative: width / circumference")
    ap.add_argument("--print_height_mm", type=float, default=None, help="Sticker height along axis (mm)")

    ap.add_argument("--theta0_deg", type=float, default=0.0, help="Start angle for left edge (deg)")
    ap.add_argument("--invert_theta", action="store_true", help="Set if sticker is mirrored left-right")
    ap.add_argument("--z_origin", choices=["center","top","bottom"], default="center")

    ap.add_argument("--min_area", type=float, default=400)
    ap.add_argument("--max_area_ratio", type=float, default=0.25)
    args = ap.parse_args()

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Cannot read image: {args.image}")

    corners, _ = detect_corners(
        img, cols=args.cols, rows=args.rows,
        min_area=args.min_area, max_area_ratio=args.max_area_ratio
    )
    if corners is None:
        raise SystemExit("Corner detection failed. Try adjusting --min_area / --max_area_ratio.")

    # 统一角点顺序
    corners = reorder_per_column_clockwise(corners, n_cols=args.cols)

    # 保存像素角点 + 重排顺序编号图
    save_corners_csv(f"{args.out_prefix}_corners.csv", corners)
    render_vis_from_corners(img, corners, f"{args.out_prefix}_annotated.png")
    out_by_col = f"{args.out_prefix}_by_col.png"
    draw_columns_with_ids(img, corners, args.cols, out_by_col)

    # 像素 → mm（圆柱面）
    xy = corners[:, :2].astype(float)
    xyz = pixels_to_cylinder_xyz(
        xy,
        img_w_px=args.img_w_px, img_h_px=args.img_h_px,
        diameter_mm=args.diameter_mm,
        print_width_mm=args.print_width_mm,
        wrap_ratio=args.wrap_ratio,
        print_height_mm=args.print_height_mm,   # ★ 新增：支持显式高度
        theta0_deg=args.theta0_deg,
        z_origin=args.z_origin,
        invert_theta=args.invert_theta          # ★ 新增：支持镜像
    )

    # 存 3D + 预览
    save_xyz_csv(f"{args.out_prefix}_xyz.csv", xyz)
    plot_xyz_png(f"{args.out_prefix}_preview.png", xyz)

    print("[OK] Saved:")
    print(f"  {args.out_prefix}_corners.csv")
    print(f"  {args.out_prefix}_annotated.png")
    print(f"  {args.out_prefix}_xyz.csv")
    print(f"  {args.out_prefix}_preview.png")

if __name__ == "__main__":
    main()
