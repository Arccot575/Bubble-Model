"""诊断：2-bubble 项在 x_e→1 的尾巴上是否被人为放大。"""
import sys
import numpy as np

sys.path.insert(0, "/home/arc/Bubble-model/Main")
from bubblemodel import constants as C, reionization as R, window as W, matter as Mat

Z_MID, DZ = 8.0, 2.0
RBAR, SIG, RMIN, RMAX = 5.0, 0.5, 0.02, 40.0

zg = np.linspace(5.0, 16.0, 45)
kc, tabs, h = Mat.camb_linear_power(list(zg))
chi = np.array([C.comoving_distance(z) for z in zg])
xe = np.array([R.x_e(z, Z_MID, DZ) for z in zg])
W_tau = C.SIGMA_T * C.N_P0 * (1.0 + zg) ** 2 * xe
kgrid = np.logspace(-4.5, 0.0, 200)
F = W.F_of_k(kgrid, RBAR, SIG, RMIN, RMAX)
Ibar = W.Ibar_of_k(kgrid, RBAR, SIG, RMIN, RMAX)
Pm = {z: Mat.MatterPower(kc, tabs[z]) for z in zg}
kp = np.logspace(-4.0, np.log10(40.0), 400)
F_kp = W.F_of_k(kp, RBAR, SIG, RMIN, RMAX)
Gz = {z: W.G_of_k(kgrid, Pm[z], F_kprime=F_kp, kprime_grid=kp) for z in zg}


def compute(mask):
    out = []
    for L in (20, 120, 500):
        k = L / chi
        f_k = np.interp(np.log(k), np.log(kgrid), F)
        g_k = np.array([np.interp(np.log(kk), np.log(kgrid), Gz[z])
                        for z, kk in zip(zg, k)])
        p1 = xe * (1 - xe) * (f_k + g_k)
        ib = np.interp(np.log(k), np.log(kgrid), Ibar)
        pref = (1 - xe) * np.log(1 - xe) * ib - xe
        p2 = pref ** 2 * np.array([Pm[z](kk) for z, kk in zip(zg, k)])
        w2 = W_tau ** 2 / chi ** 2
        out.append((np.trapezoid(w2 * p1 * mask, chi),
                    np.trapezoid(w2 * p2 * mask, chi)))
    return out


for name, mask in [
    ("全部 z (5-16)", np.ones_like(xe)),
    ("0.02<x_e<0.98", ((xe > 0.02) & (xe < 0.98)).astype(float)),
    ("0.05<x_e<0.95", ((xe > 0.05) & (xe < 0.95)).astype(float)),
]:
    rows = compute(mask)
    s = "   ".join(f"l={L}: 1b={a:.2e} 2b={b:.2e} 2b/1b={b/a:.2f}"
                   for L, (a, b) in zip((20, 120, 500), rows))
    print(f"{name:16s} {s}")

print("\n各 z 上 1-bubble / 2-bubble 的权重（乘 W^2/chi^2 之前）：")
for z, x, ch in zip(zg[::4], xe[::4], chi[::4]):
    k = 120.0 / ch
    f_k = np.interp(np.log(k), np.log(kgrid), F)
    g_k = np.interp(np.log(k), np.log(kgrid), Gz[z])
    p1 = x * (1 - x) * (f_k + g_k)
    ib = np.interp(np.log(k), np.log(kgrid), Ibar)
    pref = (1 - x) * np.log(1 - x) * ib - x
    p2 = pref ** 2 * Pm[z](k)
    print(f"  z={z:5.1f}  x_e={x:.4f}  W^2={ (C.SIGMA_T*C.N_P0*(1+z)**2*x)**2:.2e}  "
          f"P1b={p1:.2e}  P2b={p2:.2e}")
