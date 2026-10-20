"""第 4 层（★核心）：把气泡真的画到网格上，得到一个电离场 x(r)。

这一层是整个示例的基础（ground truth）：

    * 它不依赖你对窗函数 W、对 <V W> 的任何推导；
    * 只依赖"球放进盒子里"这个图像；
    * 所以它能用来反查第 3 层（窗函数平均）和第 6 层（1-bubble/2-bubble）
      的解析式是否写对 —— 这是你之前最缺的东西。

三个函数构成一条链：

    make_ionization_map()   给定 z，撒气泡、光栅化，得到 0/1 的 x(r)
    measure_volume_fraction() 测图上的平均电离率 x̄
    power_spectrum()        由图 FFT 出 P(k)

还有一个独立参照：

    boolean_power_analytic() 布尔模型（泊松球并集）在 <V_b> n_b << 1 时的
                             一阶解析功率谱。它不经过 F/G/I，是干净的对照。
"""

from __future__ import annotations

import numpy as np

from . import bubbles


# --------------------------------------------------------------------------- #
# 窗函数
# --------------------------------------------------------------------------- #
def top_hat_window(q):
    """共动球 top-hat 的傅里叶变换 W(q) = 3/(q^3) [sin q - q cos q]。

    q = k*R。q -> 0 时 W -> 1（这里显式处理，避免 0/0）。
    注意：不同文献对 W 的定义可能相差一个体积因子 V(R)，
    本包统一用「无量纲」的 W，体积因子单独乘。
    """
    q = np.asarray(q, dtype=float)
    out = np.ones_like(q)
    big = np.abs(q) > 1e-6
    qq = q[big]
    out[big] = 3.0 / qq ** 3 * (np.sin(qq) - qq * np.cos(qq))
    return out


# --------------------------------------------------------------------------- #
# 光栅化
# --------------------------------------------------------------------------- #
def paint_spheres(n_grid, box_size, centers, radii, periodic=True):
    """把一组球涂到 n_grid^3 的布尔网格上（球心、半径单位都是 Mpc）。

    实现要点
    --------
    * 每个球只在自己的"包围盒"里算距离，比全场遍历快几百倍；
    * 包围盒允许越界，索引对 n_grid 取模 —— 这就是周期边界条件；
    * 用 Python 循环 + numpy 广播，每个球一次向量化运算。
      对 10^4 个球、10^2 个网格边长，耗时约几秒，够用。
    """
    n = int(n_grid)
    cell = box_size / n
    field = np.zeros((n, n, n), dtype=bool)

    if len(centers) == 0:
        return field

    half = 0.5 * box_size
    for (cx, cy, cz), R in zip(centers, radii):
        m = int(np.ceil(R / cell)) + 1

        i0 = int(np.floor(cx / cell)) - m
        i1 = int(np.floor(cx / cell)) + m
        j0 = int(np.floor(cy / cell)) - m
        j1 = int(np.floor(cy / cell)) + m
        k0 = int(np.floor(cz / cell)) - m
        k1 = int(np.floor(cz / cell)) + m

        ii = np.arange(i0, i1 + 1)
        jj = np.arange(j0, j1 + 1)
        kk = np.arange(k0, k1 + 1)

        # 单元中心坐标 -> 相对球心的周期位移，落在 [-L/2, L/2)
        dx = ((ii + 0.5) * cell - cx + half) % box_size - half
        dy = ((jj + 0.5) * cell - cy + half) % box_size - half
        dz = ((kk + 0.5) * cell - cz + half) % box_size - half

        r2 = dx[:, None, None] ** 2 + dy[None, :, None] ** 2 + dz[None, None, :] ** 2
        mask = r2 <= R * R
        if not mask.any():
            continue

        idx = np.nonzero(mask)
        if periodic:
            field[np.mod(ii[idx[0]], n), np.mod(jj[idx[1]], n), np.mod(kk[idx[2]], n)] = True
        else:
            iv, jv, kv = ii[idx[0]], jj[idx[1]], kk[idx[2]]
            ok = (iv >= 0) & (iv < n) & (jv >= 0) & (jv < n) & (kv >= 0) & (kv < n)
            field[iv[ok], jv[ok], kv[ok]] = True

    return field


# --------------------------------------------------------------------------- #
# 由 (z, x_e) 生成一张电离图
# --------------------------------------------------------------------------- #
def make_ionization_map(z, x_e_value, rng, box_size, n_grid,
                         R_bar=bubbles.DEFAULT_R_BAR,
                         sigma_lnR=bubbles.DEFAULT_SIGMA_LNR,
                         R_min=bubbles.DEFAULT_R_MIN,
                         R_max=bubbles.DEFAULT_R_MAX):
    """生成红移 z、平均电离率 x_e_value 处的电离图。

    步骤对应物理假设：
        1) <V_b>  由 P(R) 定；               （第 2 层）
        2) n_b    由 -ln(1-x_e)/<V_b> 定；   （式 (1) 反解）
        3) 气泡数 N_b 服从 Pois(n_b * V_box)  （泊松过程）
        4) 球心均匀分布，半径从同一截断分布抽样
        5) 并集 -> 0/1 场
    """
    mean_V = bubbles.mean_volume(R_bar, sigma_lnR, R_min, R_max)
    n_b = bubbles.n_b_from_xe(x_e_value, mean_V)

    n_expected = float(n_b) * box_size ** 3
    n_bubbles = int(rng.poisson(n_expected)) if np.isfinite(n_expected) else 0

    centers = rng.uniform(0.0, box_size, size=(n_bubbles, 3))
    radii = bubbles.sample_radii(rng, n_bubbles, R_bar, sigma_lnR, R_min, R_max)

    field = paint_spheres(n_grid, box_size, centers, radii)

    info = dict(
        z=z,
        x_e_target=x_e_value,
        mean_V=mean_V,
        n_b=n_b,
        n_bubbles_expected=n_expected,
        n_bubbles=n_bubbles,
        x_e_measured=float(field.mean()),
        occupancy=float(n_b * mean_V),   # n_b<V_b>，越小说明一阶近似越好
    )
    return field, info


# --------------------------------------------------------------------------- #
# 功率谱
# --------------------------------------------------------------------------- #
def power_spectrum(field, box_size, n_bins=24, k_min=None, k_max=None):
    """由 3D 场测功率谱 P(k)。

    归一化（最容易出错的地方，务必自己推一遍）
    ------------------------------------------
        delta_k = (1/V) ∫ delta(x) e^{-ikx} d^3x
                = (1/n^3) Σ delta(x) e^{-ikx}            （离散化，V = n^3 个体元）
        P(k)    = V <|delta_k|^2>

    所以先对 FFT 结果除以 n^3，再乘盒子体积。

    自检方法：喂一个方差为 sigma^2 的白噪声场，应当得到平的
    P(k) = sigma^2 * V_voxel（selftest 里有这条测试）。
    """
    n = field.shape[0]
    box_size = float(box_size)
    voxel = (box_size / n) ** 3

    delta = field.astype(np.float64) - float(field.mean())

    fk = np.fft.fftn(delta)
    fk /= n ** 3
    Pk3d = (fk.real ** 2 + fk.imag ** 2) * box_size ** 3
    del fk

    kx = 2.0 * np.pi * np.fft.fftfreq(n, d=box_size / n)
    kz = 2.0 * np.pi * np.fft.fftfreq(n, d=box_size / n)
    k_mag = np.sqrt(kx[:, None, None] ** 2 + kx[None, :, None] ** 2 + kz[None, None, :] ** 2)

    k_fundamental = 2.0 * np.pi / box_size
    if k_min is None:
        k_min = 1.2 * k_fundamental
    if k_max is None:
        k_max = 0.9 * np.pi / (box_size / n)

    edges = np.logspace(np.log10(k_min), np.log10(k_max), n_bins + 1)

    kflat = k_mag.ravel()
    sel = (kflat >= edges[0]) & (kflat < edges[-1])
    kk = kflat[sel]
    pp = Pk3d.ravel()[sel]

    idx = np.digitize(kk, edges) - 1
    idx = np.clip(idx, 0, n_bins - 1)

    counts = np.bincount(idx, minlength=n_bins)
    sums = np.bincount(idx, weights=pp, minlength=n_bins)

    good = counts > 0
    k_centers = np.sqrt(edges[:-1] * edges[1:])        # 对数箱的几何中心
    P = np.where(good, sums / np.maximum(counts, 1), np.nan)

    return k_centers[good], P[good], counts[good]


# --------------------------------------------------------------------------- #
# 解析参照：布尔模型的功率谱（一阶，n_b<V_b> << 1）
# --------------------------------------------------------------------------- #
def boolean_power_analytic(k, x_e_value, R_bar=bubbles.DEFAULT_R_BAR,
                           sigma_lnR=bubbles.DEFAULT_SIGMA_LNR,
                           R_min=bubbles.DEFAULT_R_MIN,
                           R_max=bubbles.DEFAULT_R_MAX):
    """泊松球并集（布尔模型）的功率谱，一阶近似：

        P(k) ≈ (1-x_e)^2 * [-ln(1-x_e)] * <V_b^2 W^2(kR)> / <V_b>

    推导思路（两行）：
        xi(r) = q^2 ( exp(n_b <V_cap(r)>) - 1 ) ≈ q^2 n_b <V_cap(r)>,  q = 1-x_e
        而 <V_cap> 的傅里叶变换是 <V_b^2 W^2>，再代入 n_b = -ln(1-x_e)/<V_b>。

    适用范围：n_b<V_b> = -ln(1-x_e) 不大时（x_e <~ 0.3）定量可靠；
    x_e = 0.5 时 n_b<V_b> = 0.69，一阶近似会偏低约 1.5 倍，只用来对形状。
    """
    k = np.atleast_1d(np.asarray(k, dtype=float))
    norm = bubbles.pdf_norm(R_bar, sigma_lnR, R_min, R_max)
    V1 = bubbles.mean_volume(R_bar, sigma_lnR, R_min, R_max)

    # 在对数网格上做定步长积分（W(kR) 会振荡，自适应 quad 反而容易失败）
    R = np.logspace(np.log10(R_min), np.log10(R_max), 6000)
    weight = bubbles.lognormal_pdf(R, R_bar, sigma_lnR) * bubbles.volume(R) ** 2 / norm

    # (n_k, n_R) 的外积，一次算完所有 k
    W2 = top_hat_window(k[:, None] * R[None, :]) ** 2
    moment = np.trapezoid(weight[None, :] * W2, R, axis=1)

    prefactor = (1.0 - x_e_value) ** 2 * (-np.log1p(-x_e_value))
    return prefactor * moment / V1
