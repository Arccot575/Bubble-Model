"""bubble model 再电离模拟 —— 分层实现。

设计原则：一层只做一件事，每层单独可测。

    constants      第 0 层  常数与单位（comoving Mpc 制）
    reionization   第 1 层  x_e(z) 与平均光深 tau
    bubbles        第 2 层  气泡半径分布 P(R)、<V_b>、由 x_e 反解 n_b
    window         第 3 层  窗函数平均 F(k)、Ibar(k)、卷积项 G(k)
    maps           第 4 层  撒气泡 -> 电离图 -> 测 x̄ 和 P(k)（验证用 ground truth）
    matter         第 5 层  CAMB 的 P_m(k, z)

尚未实现（下一步）：
    第 6 层  P_1b + P_2b 合成后接 Limber 投影
    第 7 层  C_l^{tau tau}

用法：
    from bubblemodel import constants, reionization, bubbles, window, maps, matter
"""

from . import bubbles, constants, maps, matter, reionization, window

__all__ = ["constants", "reionization", "bubbles", "window", "maps", "matter"]
__version__ = "0.2.0"
