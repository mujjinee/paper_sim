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

    # find near-black pixels (axis frame / spines), typically RGB close to (0,0,0) or dark gray
    dark = (arr.sum(axis=2) < 200)
    ys, xs = np.where(dark)
    # Find the plot frame: look for the largest horizontal/vertical runs
    # Left spine: a vertical line -> mode of x among dark pixels spanning large y range
    # We'll bucket by x and find x columns where dark pixel count (contiguous vertical) is large (>200 px)
    col_counts = dark.sum(axis=0)
    row_counts = dark.sum(axis=1)
    # candidate vertical spine columns: high col_counts
    candidate_cols = np.where(col_counts > 0.5*H)[0]
    candidate_rows = np.where(row_counts > 0.3*W)[0]
    print(fname, "candidate vertical spine x's:", candidate_cols.min() if len(candidate_cols) else None, candidate_cols.max() if len(candidate_cols) else None)
    print(fname, "candidate horizontal spine y's:", candidate_rows.min() if len(candidate_rows) else None, candidate_rows.max() if len(candidate_rows) else None)

    x_left = candidate_cols.min(); x_right = candidate_cols.max()
    y_top = candidate_rows.min(); y_bottom = candidate_rows.max()
    print("frame box: x_left=%d x_right=%d y_top=%d y_bottom=%d" % (x_left, x_right, y_top, y_bottom))

    def hex_mask(hexcolor, tol=25):
        target = hex_to_rgb(hexcolor)
        diff = np.abs(arr.astype(int) - target.astype(int)).sum(axis=2)
        return diff < tol

    om = hex_mask("#F97316")  # orange solid (우리 결과 nRMSE)
    bm = hex_mask("#3B82F6")  # blue solid (우리 결과 gap)

    oys, oxs = np.where(om)
    bys, bxs = np.where(bm)

    # 11 categories evenly spaced between x_left..x_right (approx, category centers may have padding)
    # Use actual marker x-extent instead (line spans exactly from cat0 to cat10 center)
    x_min_o, x_max_o = oxs.min(), oxs.max()
    n_cat = 11
    cat_x = [x_min_o + (x_max_o - x_min_o) * i / (n_cat - 1) for i in range(n_cat)]
    print("category x positions (from orange line extent):", [round(c) for c in cat_x])

    def value_at_x(mask_ys, mask_xs, x_target, y_top, y_bottom, val_top, val_bottom, tol_x=6):
        sel = np.abs(mask_xs - x_target) <= tol_x
        if sel.sum() == 0:
            return None
        y_at = mask_ys[sel].mean()
        # linear map: y_top -> val_top, y_bottom -> val_bottom
        frac = (y_at - y_top) / (y_bottom - y_top)
        val = val_top + frac * (val_bottom - val_top)
        return val, y_at

    labels = ["AR","1/20","1/10","1/5","1/2","1/1","2/1","5/1","10/1","20/1","1/0"]
    print("\n--- nRMSE (orange, left axis, range=%s) ---" % (left_range,))
    for i, lbl in enumerate(labels):
        res = value_at_x(oys, oxs, cat_x[i], y_top, y_bottom, left_range[1], left_range[0])
        if res:
            val, y_at = res
            print(f"  {lbl:>5}: nRMSE ~ {val:6.2f}%  (y_px={y_at:.1f})")

    print("--- Gap (blue, right axis, range=%s) ---" % (right_range,))
    for i, lbl in enumerate(labels):
        res = value_at_x(bys, bxs, cat_x[i], y_top, y_bottom, right_range[1], right_range[0])
        if res:
            val, y_at = res
            print(f"  {lbl:>5}: Gap ~ {val:6.2f}%  (y_px={y_at:.1f})")

analyze("fig3_ar_original.png", left_range=(30,120), right_range=(0,25))
print()
