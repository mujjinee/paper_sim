# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image

DIR = r"기본코드_(2)_AR_MLR_baseline_proposed_3가지_profit_figure코드"

def hex_to_rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i+2], 16) for i in (0, 2, 4)])

def analyze(fname, left_range, right_range):
    img = Image.open(f"{DIR}/{fname}").convert("RGB")
    arr = np.array(img)
    H, W, _ = arr.shape

    dark = (arr.sum(axis=2) < 200)
    col_counts = dark.sum(axis=0)
    row_counts = dark.sum(axis=1)
    candidate_cols = np.where(col_counts > 0.5*H)[0]
    candidate_rows = np.where(row_counts > 0.3*W)[0]
    x_left = candidate_cols.min(); x_right = candidate_cols.max()
    y_top = candidate_rows.min(); y_bottom = candidate_rows.max()
    print(f"=== {fname} === frame box: x_left={x_left} x_right={x_right} y_top={y_top} y_bottom={y_bottom}")

    def hex_mask(hexcolor, tol=25):
        target = hex_to_rgb(hexcolor)
        diff = np.abs(arr.astype(int) - target.astype(int)).sum(axis=2)
        m = diff < tol
        # restrict to inside frame (exclude axis label text / legend)
        frame_mask = np.zeros_like(m)
        frame_mask[y_top+2:y_bottom-1, x_left+2:x_right-1] = True
        return m & frame_mask

    om = hex_mask("#F97316")
    bm = hex_mask("#3B82F6")

    oys, oxs = np.where(om)
    bys, bxs = np.where(bm)
    print("orange(inside frame) x range:", oxs.min(), oxs.max(), "count:", len(oxs))
    print("blue(inside frame) x range:", bxs.min(), bxs.max(), "count:", len(bxs))

    n_cat = 11
    x_min_o, x_max_o = oxs.min(), oxs.max()
    cat_x = [x_min_o + (x_max_o - x_min_o) * i / (n_cat - 1) for i in range(n_cat)]
    print("category x positions:", [round(c) for c in cat_x])

    def value_at_x(mask_ys, mask_xs, x_target, val_top, val_bottom, tol_x=5):
        sel = np.abs(mask_xs - x_target) <= tol_x
        if sel.sum() == 0:
            return None
        y_at = np.median(mask_ys[sel])
        frac = (y_at - y_top) / (y_bottom - y_top)
        val = val_top + frac * (val_bottom - val_top)
        return val, y_at, sel.sum()

    labels = ["AR","1/20","1/10","1/5","1/2","1/1","2/1","5/1","10/1","20/1","1/0"]
    print("\n--- nRMSE (orange, left axis) ---")
    for i, lbl in enumerate(labels):
        res = value_at_x(oys, oxs, cat_x[i], left_range[1], left_range[0])
        if res:
            val, y_at, n = res
            print(f"  {lbl:>5}: nRMSE ~ {val:6.2f}%  (y_px={y_at:.1f}, n={n})")

    print("--- Gap (blue, right axis) ---")
    for i, lbl in enumerate(labels):
        res = value_at_x(bys, bxs, cat_x[i], right_range[1], right_range[0])
        if res:
            val, y_at, n = res
            print(f"  {lbl:>5}: Gap ~ {val:6.2f}%  (y_px={y_at:.1f}, n={n})")

analyze("fig3_ar_original.png", left_range=(30,120), right_range=(0,25))
