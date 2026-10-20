"""第 0 层：常数与单位。

约定（整个包统一遵守，不要在任何地方就地换算）：

* 长度单位   : 共动 Mpc (comoving Mpc)
* 体积单位   : Mpc^3
* 数密度单位 : Mpc^-3
* 二维/三维功率谱单位 : Mpc^2 / Mpc^3

为什么单独一层？
    单位混乱是这类代码最隐蔽的 bug：图照样能画出来，只是差了 10 倍。
    把所有常数集中在这里、只在一个地方做 cm<->Mpc 的换算，
    以后任何单位错误都只可能出现在这一个文件里。
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad

# --------------------------------------------------------------------------- #
# 基本物理常数
# --------------------------------------------------------------------------- #
C_KM_S = 299792.458                    # 光速 [km/s]
MPC_TO_CM = 3.0856775814913673e24      # 1 Mpc = 3.086e24 cm

# 汤姆孙散射截面 6.652e-25 cm^2，换算成 Mpc^2
SIGMA_T_CM2 = 6.6524587321e-25
SIGMA_T = SIGMA_T_CM2 / MPC_TO_CM**2   # ≈ 6.99e-74 Mpc^2

# --------------------------------------------------------------------------- #
# 背景宇宙学：Planck 2018 (TT,TE,EE+lowE+lensing) 近似值
# --------------------------------------------------------------------------- #
H0_KM_S_MPC = 67.4          # 哈勃常数 [km/s/Mpc]
OMEGA_B = 0.0493            # 重子
OMEGA_C = 0.2650            # 冷暗物质
OMEGA_M = OMEGA_B + OMEGA_C
OMEGA_L = 1.0 - OMEGA_M     # 平坦宇宙

# 哈勃距离 c/H0 ≈ 4448 Mpc，后面 dchi = c dz/H(z) 会反复用到
HUBBLE_DISTANCE = C_KM_S / H0_KM_S_MPC

# --------------------------------------------------------------------------- #
# 共动质子数密度 n_p0
#     n_p0 = Omega_b * rho_crit / m_p
#     结果约 7.4e66 Mpc^-3（和原代码里的 7.4e66 一致，可互相印证）
# --------------------------------------------------------------------------- #
RHO_C_H2 = 2.77536627e11                            # 临界密度 [h^2 M_sun/Mpc^3]
M_P_SUN = 1.67262192369e-27 / 1.98847e30            # 质子质量 [M_sun]
RHO_CRIT = RHO_C_H2 * (H0_KM_S_MPC / 100.0) ** 2    # [M_sun/Mpc^3]
N_P0 = OMEGA_B * RHO_CRIT / M_P_SUN                 # ≈ 7.4e66 Mpc^-3

# 电子数密度相对质子数密度的修正（氦的电离贡献）
#   n_e/n_H = 1 + Y/(4X)，X=0.76, Y=0.24 -> 约 1.079
# 原代码没有加这一项；这里默认 1.0 保持可比，需要时显式传入 1.079。
N_E_OVER_N_P_DEFAULT = 1.0


# --------------------------------------------------------------------------- #
# 背景膨胀
# --------------------------------------------------------------------------- #
def efunc(z):
    """E(z) = H(z)/H0（平坦 LCDM）。"""
    z = np.asarray(z, dtype=float)
    return np.sqrt(OMEGA_M * (1.0 + z) ** 3 + OMEGA_L)


def hubble(z):
    """H(z) [km/s/Mpc]。"""
    return H0_KM_S_MPC * efunc(z)


def comoving_distance(z):
    """共动距离 chi(z) [Mpc]，chi = ∫0^z c dz'/H(z')。

    只接受标量；本包里 chi 只在作图/标注时用，不需要向量化。
    """
    if np.ndim(z) != 0:
        raise TypeError("comoving_distance 目前只支持标量输入")
    if z <= 0:
        return 0.0
    val, _ = quad(lambda zp: C_KM_S / hubble(zp), 0.0, float(z))
    return val


def summary():
    """打印本层的数值，用于自查（selftest 会调用）。"""
    print(f"  SIGMA_T  = {SIGMA_T:.4e} Mpc^2   (参考 6.99e-74)")
    print(f"  N_P0     = {N_P0:.4e} Mpc^-3    (参考 7.4e66)")
    print(f"  HUBBLE_D = {HUBBLE_DISTANCE:.1f} Mpc (参考 4448)")
    print(f"  chi(z=8) = {comoving_distance(8.0):.1f} Mpc")
