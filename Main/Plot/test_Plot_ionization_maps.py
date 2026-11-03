# test_Plot_ionization_maps
"""画电离场的二维切片，用 5 个红移并排展示再电离的演化过程
（对应 fig3_ionization_maps）。

每个面板是三维二值场 field[n_grid//2] 的一张切片：
    深蓝 = 中性氢，琥珀 = 已电离
气泡个数 N_b 由 n_b = -ln(1-x_e)/<V_b> 定，气泡位置均匀随机、半径按 P(R) 抽样，
所以红移越低（x_e 越大）气泡越多、越大，最后连成一片 —— 这就是渗流。

坐标单位是 Mpc，盒子是边长 box_size 的周期立方体。

运行：
    cd Main/Plot && python test_Plot_ionization_maps.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from bubblemodel import maps as M
from bubblemodel import reionization as R

# ------------------------------------------------------------------ 参数
CFG = dict(
    z_mid=8.0,          # 再电离中点
    delta_z=2.0,        # 再电离宽度
    R_bar=5.0,          # 气泡半径中位数 [Mpc]
    sigma_lnR=0.5,      # ln R 的宽度
    R_min=0.02,
    R_max=40.0,         # 必须明显小于 box_size
    box_size=250.0,     # 盒子边长 [Mpc]
    n_grid=192,         # 网格边长，分辨率 250/192 ≈ 1.3 Mpc
    z_show=(12.0, 10.5, 9.5, 8.5, 7.5),
    seed=2024,
)

SAVE = True
OUT = Path(__file__).resolve().parents[2] / "Results" / "test_plot_ionization_maps.png"


def main():
    # 注意：不要用 twilight 之类的循环色图，它的 0 和 1 是同一个颜色，
    # 二值场会被画成一片空白。两色 ListedColormap 才可靠。
    cmap = ListedColormap(["#12122b", "#f5c451"])   # 中性=深蓝，电离=琥珀

    rng = np.random.default_rng(CFG["seed"])
    fields = {}
    for z in CFG["z_show"]:
        xe = float(R.x_e(z, CFG["z_mid"], CFG["delta_z"]))
        field, info = M.make_ionization_map(
            z, xe, rng, CFG["box_size"], CFG["n_grid"],
            CFG["R_bar"], CFG["sigma_lnR"], CFG["R_min"], CFG["R_max"])
        fields[z] = (field, info)
        print(f"z={z:5.2f}  x_e={info['x_e_measured']:7.4f}  "
              f"N_b={info['n_bubbles']:5d}")

    n = CFG["n_grid"]
    fig, axes = plt.subplots(1, len(CFG["z_show"]),
                             figsize=(2.9 * len(CFG["z_show"]), 3.4),
                             constrained_layout=True)
    axes = np.atleast_1d(axes)

    for j, (ax, z) in enumerate(zip(axes, CFG["z_show"])):
        field, info = fields[z]
        im = ax.imshow(field[n // 2].astype(float), origin="lower", cmap=cmap,
                       extent=[0, CFG["box_size"], 0, CFG["box_size"]],
                       interpolation="nearest", vmin=0, vmax=1)

        # 用角标文字标红移，不设 title；白字加深色描边，压在图上也能看清
        ax.text(0.04, 0.95,
                f"$z$ = {z:g}\n" rf"$\bar x_e$ = {info['x_e_measured']:.3f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=9,
                color="w",
                path_effects=[pe.withStroke(linewidth=2.2, foreground="#12122b")])

        ax.set_xlabel("Mpc")
        if j == 0:
            ax.set_ylabel("Mpc")
        else:
            ax.set_yticklabels([])

    # 一条共享色条，代替图例
    cb = fig.colorbar(im, ax=axes, shrink=0.8, ticks=[0.25, 0.75], pad=0.01)
    cb.ax.set_yticklabels(["neutral", "ionized"])

    if SAVE:
        fig.savefig(OUT, dpi=150, bbox_inches="tight")
        print(f"-> {OUT}")
    plt.show()


if __name__ == "__main__":
    main()
