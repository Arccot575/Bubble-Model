"""bubble model 再电离模拟 —— 分层实现。

设计原则：一层只做一件事，每层单独可测。

    constants      第 0 层  常数与单位（comoving Mpc 制）
    reionization   第 1 层  x_e(z) 与平均光深 tau
    bubbles        第 2 层  气泡半径分布 P(R)、<V_b>、由 x_e 反解 n_b
    maps           第 4 层  撒气泡 -> 电离图 -> 测 x̄ 和 P(k)（本示例的核心）

尚未实现、留给下一步的层（对应你笔记里的 F/G/I 和 C_l）：
    第 3 层  窗函数平均 <V W>/<V>、<V^2 W^2>/<V>
    第 5 层  CAMB 的 P_m(k, z)
    第 6 层  1-bubble + 2-bubble 合成 P_dx
    第 7 层  Limber 投影到 C_l^{tau tau}

用法：
    from bubblemodel import constants, reionization, bubbles, maps
"""

from . import bubbles, constants, maps, reionization

__all__ = ["constants", "reionization", "bubbles", "maps"]
__version__ = "0.1.0"
