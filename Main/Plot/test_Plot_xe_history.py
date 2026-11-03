# test_Plot_xe_history
"""画平均电离度 x_e 随红移的演化（对应 fig1_xe_history 的左边子图）。

    x_e(z) = 0.5 * (1 - tanh( (y - y_re) / Delta_y )),   y = (1+z)^1.5

每条曲线是一组 (z_mid, Delta_z)：z_mid 是 x_e=0.5 的红移，Delta_z 是过渡宽度。
横轴取 4 < z < 25，把两端"完全电离（x_e->0）"和"完全中性（x_e->1）"的平台都露出来。

运行：
    cd Main/Plot && python test_Plot_xe_history.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from bubblemodel import reionization as R

# ------------------------------------------------------------------ 参数
CURVES = [          # (z_mid, delta_z)
    (6.0, 1.5),
    (8.0, 2.0),
    (10.0, 2.5),
    (12.0, 3.0),
]
Z_MIN, Z_MAX, N_Z = 0.0, 25.0, 900      # 一直画到今天（z=0），让"完全电离"的平台占到足够宽度
YLIM = (-0.05, 1.05)                    # 上下留白，否则 x_e=1 / x_e=0 的平台会压在边框上
Z_INCREASING_RIGHT = False              # False: 时间向右（z 从 25 减到 2，文献常见）
                                        # True : z 从左到右增大（和 fig1_xe_history 一致）

SAVE = True
OUT = Path(__file__).resolve().parents[2] / "Results" / "test_plot_xe_history.png"


def main():
    z = np.linspace(Z_MIN, Z_MAX, N_Z)

    plt.figure()
    for z_mid, dz in CURVES:
        plt.plot(z, R.x_e(z, z_mid, dz),
                 label=rf"$z_{{\rm mid}}$={z_mid:g}, $\Delta z$={dz:g}")

    plt.xlabel(r"redshift $z$")
    plt.ylabel(r"mean ionized fraction $\bar{x}_e$")
    if Z_INCREASING_RIGHT:
        plt.xlim(Z_MIN, Z_MAX)
    else:
        plt.xlim(Z_MAX, Z_MIN)      # 时间向右：左端 z=25（早、中性），右端 z=2（晚、电离完）
    plt.ylim(*YLIM)
    plt.legend()

    if SAVE:
        plt.savefig(OUT, dpi=150, bbox_inches="tight")
        print(f"-> {OUT}")
    plt.show()


if __name__ == "__main__":
    main()
