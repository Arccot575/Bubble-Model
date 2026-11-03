"""Cl_tautau: 从 3D 电离场功率谱 P_DeltaX(k,z) 用 Limber 投影到角功率谱。

C_l^{tau tau} = \int dchi  [W_tau(chi)]^2 / chi^2  * P_DeltaX(k=l/chi, z(chi))

W_tau(chi) = sigma_T * n_e0 * (1+z)^2 * x_bar_e(z)     [1/Mpc]

1-bubble 与 2-bubble 分开存，方便看谁在哪个尺度上主导。
"""
import sys
import numpy as np

sys.path.insert(0, "/home/arc/Bubble-model/Main")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from bubblemodel import constants as C, reionization as R, window as W, matter as Mat

Z_MID, DZ = 8.0, 2.0
RBAR, SIG, RMIN, RMAX = 5.0, 0.5, 0.02, 40.0
B_BIAS = 1.0

zg = np.linspace(5.0, 16.0, 45)
kc, tabs, h = Mat.camb_linear_power(list(zg))
chi = np.array([C.comoving_distance(z) for z in zg])
xe = np.array([R.x_e(z, Z_MID, DZ) for z in zg])
W_tau = C.SIGMA_T * C.N_P0 * (1.0 + zg) ** 2 * xe

kgrid = np.logspace(-4.5, 0.0, 200)
F = W.F_of_k(kgrid, RBAR, SIG, RMIN, RMAX)
Ibar = W.Ibar_of_k(kgrid, RBAR, SIG, RMIN, RMAX)
Pm = {z: Mat.MatterPower(kc, tabs[z]) for z in zg}

# G(k,z) 依赖 P_m(k',z)，每个 z 单独积一次
kp = np.logspace(-4.0, np.log10(40.0), 400)
F_kp = W.F_of_k(kp, RBAR, SIG, RMIN, RMAX)
Gz = {z: W.G_of_k(kgrid, Pm[z], F_kprime=F_kp, kprime_grid=kp) for z in zg}

ell = np.unique(np.round(np.logspace(np.log10(2), np.log10(2000), 60)))
c1 = np.zeros_like(ell, float)
c2 = np.zeros_like(ell, float)
for j, L in enumerate(ell):
    k = L / chi
    # 1-bubble: x_e(1-x_e)[F(k) + G(k)]
    f_k = np.interp(np.log(k), np.log(kgrid), F)
    g_k = np.array([np.interp(np.log(kk), np.log(kgrid), Gz[z]) for z, kk in zip(zg, k)])
    p1 = xe * (1 - xe) * (f_k + g_k)
    ib = np.interp(np.log(k), np.log(kgrid), Ibar)
    pref = (1 - xe) * np.log(1 - xe) * B_BIAS * ib - xe
    p2 = pref ** 2 * np.array([Pm[z](kk) for z, kk in zip(zg, k)])
    c1[j] = np.trapezoid(W_tau ** 2 / chi ** 2 * p1, chi)
    c2[j] = np.trapezoid(W_tau ** 2 / chi ** 2 * p2, chi)

print(f"W_tau 峰值 {W_tau.max():.3e} /Mpc,  chi in [{chi.min():.0f}, {chi.max():.0f}] Mpc")
print(f"tau_bar = {float(R.optical_depth(Z_MID, DZ)):.4f}")
print("\n  ell      C_l(1b)      C_l(2b)     l(l+1)C/2pi")
for L, a, b in zip(ell, c1, c2):
    tot = a + b
    print(f"  {L:6.0f}  {a:11.3e}  {b:11.3e}  {L*(L+1)*tot/2/np.pi:11.3e}")

var = np.sum((2 * ell + 1) * (c1 + c2)) / (4 * np.pi)
print(f"\n  rms(dtau) = {np.sqrt(var):.3e}  (对 l<=2000 求和)")
print(f"  rms(dtau)/tau_bar = {np.sqrt(var)/float(R.optical_depth(Z_MID, DZ)):.3f}")

fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
ax.loglog(ell, ell * (ell + 1) * c1 / (2 * np.pi), lw=2, label=r"$1$-bubble")
ax.loglog(ell, ell * (ell + 1) * c2 / (2 * np.pi), lw=2, label=r"$2$-bubble")
ax.loglog(ell, ell * (ell + 1) * (c1 + c2) / (2 * np.pi), "k--", lw=2, label="total")
ax.axvspan(2, 10, color="gray", alpha=0.12)
ax.set_xlabel(r"$\ell$")
ax.set_ylabel(r"$\ell(\ell+1)C_\ell^{\tau\tau}/2\pi$")
ax.set_title(rf"$\delta\tau$ angular power spectrum  ($\bar R$={RBAR:g} Mpc, "
             rf"$\sigma_{{\ln R}}$={SIG:g}, $z_{{\rm mid}}$={Z_MID:g})", fontsize=10)
ax.legend(fontsize=9)
ax.grid(alpha=0.3, which="both")
out = Path("/home/arc/Bubble-model/Results/fig8_tautau_cell.png")
fig.savefig(out, dpi=150)
print(f"\n  -> {out}")
