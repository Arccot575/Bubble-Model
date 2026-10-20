"""bubble model 分层自检。

规矩：每一层都必须 PASS，才允许用它的结果去做下一层。
跑法（在 Main/ 目录下）：

    /home/arc/miniforge3/envs/COSMIC/bin/python selftest_bubblemodel.py

返回码 0 表示全部通过。
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bubblemodel import bubbles as B
from bubblemodel import constants as C
from bubblemodel import maps as M
from bubblemodel import reionization as R

_RESULTS = []


def check(name, condition, detail=""):
    ok = bool(condition)
    _RESULTS.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"   ({detail})" if detail else ""))
    return ok


def approx(a, b, rel=1e-6):
    return abs(a - b) / max(abs(b), 1e-300) < rel


def main():
    print("=" * 78)
    print("bubble model 自检")
    print("=" * 78)

    # ------------------------------------------------------------------ L0
    print("\n--- 第 0 层：常数与单位 ---")
    check("sigma_T 换算 cm^2 -> Mpc^2",
          approx(C.SIGMA_T, 6.99e-74, 2e-2), f"{C.SIGMA_T:.3e} Mpc^2")
    check("n_p0 = Omega_b rho_c / m_p",
          approx(C.N_P0, 7.4e66, 2e-2), f"{C.N_P0:.3e} Mpc^-3")
    check("哈勃距离 c/H0",
          approx(C.HUBBLE_DISTANCE, 4448.0, 1e-2), f"{C.HUBBLE_DISTANCE:.1f} Mpc")
    # 量纲检查：n_b * <V_b> 必须无量纲
    occ = B.n_b_from_xe(0.5, B.mean_volume()) * B.mean_volume()
    check("n_b * <V_b> 无量纲 (= -ln(1-x_e))",
          approx(occ, -np.log(0.5), 1e-6), f"{occ:.6f}")

    # ------------------------------------------------------------------ L1
    print("\n--- 第 1 层：x_e(z) 与光深 ---")
    check("x_e(z_mid) = 0.5", approx(float(R.x_e(8.0, 8.0, 2.0)), 0.5, 1e-10))
    zs = np.linspace(0.0, 30.0, 301)
    xe = R.x_e(zs, 8.0, 2.0)
    check("x_e 随 z 单调递减", bool(np.all(np.diff(xe) <= 1e-12)))
    check("x_e 落在 [0, 1]", bool(np.all((xe >= 0.0) & (xe <= 1.0))))
    tau = R.optical_depth(8.0, 2.0)
    check("tau 在物理范围 (0.04, 0.075)",
          0.04 < tau < 0.075, f"tau = {tau:.4f}, Planck = 0.054 +/- 0.007")
    tau_hi = R.optical_depth(8.0, 2.0, z_max=200.0)
    check("tau 对 z_max 收敛", approx(tau_hi, tau, 1e-3), f"{tau_hi:.4f} vs {tau:.4f}")

    # ------------------------------------------------------------------ L2
    print("\n--- 第 2 层：P(R) 与体积统计 ---")
    norm = B.pdf_norm()
    # pdf_norm 是「截断区间内包含的概率」，它 <= 1；差额才是截断丢掉的概率
    check("截断丢掉的总概率 < 0.1%", (1.0 - norm) < 1e-3,
          f"丢掉 {100 * (1.0 - norm):.5f}%")
    Rn = np.logspace(np.log10(B.DEFAULT_R_MIN), np.log10(B.DEFAULT_R_MAX), 200000)
    int_trunc = float(np.trapezoid(B.truncated_pdf(Rn), Rn))
    check("归一化后 ∫P_trunc dR = 1", approx(int_trunc, 1.0, 1e-6),
          f"{int_trunc:.10f}")

    # 解析矩 vs 数值积分：两条独立路径必须一致
    Rnum = np.logspace(np.log10(B.DEFAULT_R_MIN), np.log10(B.DEFAULT_R_MAX), 200000)
    V_num = float(np.trapezoid(B.lognormal_pdf(Rnum) * B.volume(Rnum), Rnum) / norm)
    check("解析 <V_b> 与数值积分一致（1e-4 内）",
          approx(B.mean_volume(), V_num, 1e-4),
          f"{B.mean_volume():.6f} vs {V_num:.6f} Mpc^3")

    loss = B.truncation_loss()
    check("默认参数下截断损失 < 1%", loss < 0.01, f"{100 * loss:.3f}%")

    V1 = B.mean_volume()
    V1_cf = B.mean_volume_closed_form()
    check("截断 <V_b> / 不截断 <V_b> = 1 - 截断损失",
          approx(V1 / V1_cf, 1.0 - loss, 1e-10),
          f"{V1:.2f} vs {V1_cf:.2f} Mpc^3")

    # 抽样器是否真的服从同一个分布
    rng = np.random.default_rng(7)
    n = 200000
    Rs = B.sample_radii(rng, n)
    check("抽样 ln R 的均值 = ln R_bar", abs(np.log(Rs).mean() - np.log(B.DEFAULT_R_BAR)) < 0.01,
          f"{np.log(Rs).mean():.4f}")
    check("抽样 ln R 的标准差 = sigma_lnR",
          abs(np.log(Rs).std() - B.DEFAULT_SIGMA_LNR) < 0.01,
          f"{np.log(Rs).std():.4f}")
    Vs = B.volume(Rs).mean()
    check("抽样 <V_b> 与解析 <V_b> 一致（2% 内）",
          abs(Vs / V1 - 1.0) < 0.02, f"{Vs:.1f} vs {V1:.1f} Mpc^3")

    # ------------------------------------------------------------------ 窗函数
    print("\n--- 窗函数 W(kR) ---")
    check("W(q->0) = 1", approx(float(M.top_hat_window(1e-9)), 1.0, 1e-8))
    # tan q = q 的第一个正根 q = 4.4934，是 W 的第一个零点
    check("W 在 q=4.4934 处过零（tan q = q 的首根）",
          abs(float(M.top_hat_window(4.4934))) < 1e-4,
          f"W(4.4934) = {float(M.top_hat_window(4.4934)):.2e}")
    check("W 在 q=2pi 处不为零（不是整数倍 π 处过零）",
          abs(float(M.top_hat_window(2 * np.pi))) > 0.05,
          f"W(2pi) = {float(M.top_hat_window(2 * np.pi)):.3f}")

    # ------------------------------------------------------------------ FFT
    print("\n--- 功率谱估计器的归一化 ---")
    # Parseval 恒等式（严格成立，不是统计检验）：
    #     Σ_k P(k) = V_voxel * Σ_x delta(x)^2
    rng = np.random.default_rng(11)
    n = 64
    box = 100.0
    delta = rng.normal(size=(n, n, n))
    delta -= delta.mean()
    fk = np.fft.fftn(delta) / n ** 3
    Pk3d = (fk.real ** 2 + fk.imag ** 2) * box ** 3
    lhs = Pk3d.sum()
    rhs = (box / n) ** 3 * (delta ** 2).sum()
    check("Parseval: Σ_k P(k) = V_voxel Σ_x δ^2",
          approx(lhs, rhs, 1e-10), f"{lhs:.6e} vs {rhs:.6e}")

    # 白噪声应当给出平的 P(k) = sigma^2 * V_voxel
    # 判据不能写成"每个箱都在 15% 内"：模式数少的箱本来就有统计涨落。
    # 正确的判据是「按泊松统计的预期离散度」，即 sigma_P/P = sqrt(2/N_modes)。
    k, P, cnt = M.power_spectrum(delta, box, n_bins=10)
    expected = 1.0 * (box / n) ** 3
    use = cnt >= 50
    dev = P[use] / expected - 1.0
    sigma_stat = np.sqrt(2.0 / cnt[use])
    within = np.all(np.abs(dev) < 4.0 * sigma_stat)
    check("白噪声 P(k) = sigma^2 V_voxel（在 4 sigma_统计 内）",
          bool(within),
          f"最大偏离 {np.max(np.abs(dev) / sigma_stat):.1f} sigma"
          f"（{int(use.sum())} 个箱，最小模式数 {int(cnt[use].min())}）")

    # ------------------------------------------------------------------ 地图
    print("\n--- 第 4 层：地图上的 x̄ 与解析式 ---")
    box_map = 200.0
    n_map = 128
    x_target = 0.5
    V1 = B.mean_volume()
    xe_analytic = 1.0 - np.exp(-B.n_b_from_xe(x_target, V1) * V1)

    measured = []
    for seed in range(8):
        rng = np.random.default_rng(100 + seed)
        field, info = M.make_ionization_map(
            z=8.0, x_e_value=x_target, rng=rng,
            box_size=box_map, n_grid=n_map)
        measured.append(info["x_e_measured"])
    measured = np.array(measured)

    err = measured.mean() - xe_analytic
    sem = measured.std(ddof=1) / np.sqrt(len(measured))
    check("地图实测 x̄ = 1 - exp(-n_b<V_b>)（3 sigma 内）",
          abs(err) < 3 * sem + 0.005,
          f"实测 {measured.mean():.4f} +/- {sem:.4f} vs 解析 {xe_analytic:.4f}")

    # 解析参照必须是正数
    kk = np.array([0.1, 0.3, 1.0])
    pa = M.boolean_power_analytic(kk, x_target)
    check("boolean_power_analytic 全为正", bool(np.all(pa > 0)), f"{pa}")

    # ------------------------------------------------------------------ 汇总
    print("\n" + "=" * 78)
    n_pass = sum(_RESULTS)
    n_all = len(_RESULTS)
    print(f"结果：{n_pass}/{n_all} 通过")
    if n_pass == n_all:
        print("全部通过 —— 可以放心用这些模块去算 F/G/I 和 C_l 了。")
        return 0
    print("有失败项 —— 先修好再往下走，别在坏地基上盖楼。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
