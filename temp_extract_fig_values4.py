# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image

DIR = r"기본코드_(2)_AR_MLR_baseline_proposed_3가지_profit_figure코드"

def hex_to_rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i+2], 16) for i in (0, 2, 4)])

def get_frame(arr):
    H, W, _ = arr.shape
    dark = (arr.sum(axis=2) < 200)
    col_counts = dark.sum(axis=0)
    row_counts = dark.sum(axis=1)
    candidate_cols = np.where(col_counts > 0.5*H)[0]
    candidate_rows = np.where(row_counts > 0.3*W)[0]
    return candidate_cols.min(), candidate_cols.max(), candidate_rows.min(), candidate_rows.max()

def label_clusters_y(arr, hexcolor, x_lo, x_hi, y_top, y_bottom, tol=40):
    target = hex_to_rgb(hexcolor)
    diff = np.abs(arr.astype(int) - target.astype(int)).sum(axis=2)
    m = diff < tol
    region = np.zeros_like(m)
    region[max(0,y_top-10):y_bottom+10, x_lo:x_hi] = True
    m = m & region
    ys, xs = np.where(m)
    if len(ys) == 0:
        return []
    order = np.argsort(ys)
    ys_sorted = ys[order]
    # cluster by gaps
    clusters = []
    cur = [ys_sorted[0]]
    for y in ys_sorted[1:]:
        if y - cur[-1] > 15:
            clusters.append(cur)
            cur = [y]
        else:
            cur.append(y)
    clusters.append(cur)
    centers = [np.mean(c) for c in clusters]
    return centers

def analyze(fname):
    img = Image.open(f"{DIR}/{fname}").convert("RGB")
    arr = np.array(img)
    H, W, _ = arr.shape
    x_left, x_right, y_top, y_bottom = get_frame(arr)
    print(f"{fname}: frame x=({x_left},{x_right}) y=({y_top},{y_bottom})")

    # left axis tick labels: orange text, located left of x_left
    left_centers = label_clusters_y(arr, "#F97316", max(0,x_left-100), x_left-5, y_top, y_bottom)
    print("left axis label y-centers (top->bottom):", [round(c,1) for c in left_centers])

    # right axis tick labels: blue text, located right of x_right
    right_centers = label_clusters_y(arr, "#3B82F6", x_right+5, min(W,x_right+100), y_top, y_bottom)
    print("right axis label y-centers (top->bottom):", [round(c,1) for c in right_centers])
    return x_left, x_right, y_top, y_bottom, left_centers, right_centers

analyze("fig3_ar_original.png")
print()
analyze("fig6_mlr_original.png")
