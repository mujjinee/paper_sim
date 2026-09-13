# -*- coding: utf-8 -*-
# 기본코드_(2) 디렉토리의 fig3_ar_original.png / fig6_mlr_original.png에서
# "1/20" (W1=1,W2=20) 지점의 nRMSE/Gap 픽셀 좌표를 읽어 축 스케일로 역산한다.
import numpy as np
from PIL import Image

DIR = r"기본코드_(2)_AR_MLR_baseline_proposed_3가지_profit_figure코드"

def analyze(fname, orange_hex="#F97316", blue_hex="#3B82F6"):
    img = Image.open(f"{DIR}/{fname}").convert("RGB")
    arr = np.array(img)
    print(f"=== {fname} === shape={arr.shape}")

    def hex_to_rgb(h):
        h = h.lstrip("#")
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

    orange = np.array(hex_to_rgb(orange_hex))
    blue = np.array(hex_to_rgb(blue_hex))

    def mask_color(target, tol=25):
        diff = np.abs(arr.astype(int) - target.astype(int)).sum(axis=2)
        return diff < tol

    om = mask_color(orange)
    bm = mask_color(blue)
    ys, xs = np.where(om)
    print("orange pixel count:", len(xs), "x range:", xs.min() if len(xs) else None, xs.max() if len(xs) else None)
    ys2, xs2 = np.where(bm)
    print("blue pixel count:", len(xs2), "x range:", xs2.min() if len(xs2) else None, xs2.max() if len(xs2) else None)
    return arr, om, bm

analyze("fig3_ar_original.png")
analyze("fig6_mlr_original.png")
