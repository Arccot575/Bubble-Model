
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt

def top_hat_window_function_W(kappa):
    """
    The Fourier transform of a real-space tophat window function with radius R

    Parameters:
    k = ell/chi:  ??
    R (float or array-like): Radius values.
    """
    # Handle kR -> 0 limit: W -> 1
    # Use np.where to avoid division by zero

    with np.errstate(invalid='ignore', divide='ignore'):
        W_val = np.where(np.abs(kappa) < 1e-10, 1.0,
                        3/(kappa)**3 * (np.sin(kappa) - kappa * np.cos(kappa)))
    return W_val

def averaged_window_function_W(kappa, sigma_lnR):

    def f_log(u):
        x = kappa * np.exp(u)

        return (
            np.sin(x) - x * np.cos(x)
        ) * np.exp(-u**2 / (2 * sigma_lnR**2))

    W_int, error_w_int = integrate.quad(
        f_log,
        -10 * sigma_lnR,
        10 * sigma_lnR,
        epsabs=1e-12,
        epsrel=1e-10,
        limit=1000
    )

    coeff = (
        3 * np.exp(-4.5 * sigma_lnR**2)
        / (np.sqrt(2*np.pi) * sigma_lnR * kappa**3)
    )

    return coeff * W_int

def averaged_squared_window_function_W_2(kappa, sigma_lnR):
    """
    Squared top-hat window function averaged over radius R with distribution P(R)
    """

    def f_log(u):
        theta = kappa * np.exp(u)

        return  (
            np.sin(theta)-theta*np.cos(theta)
        )**2 * np.exp(-u**2 / (2*sigma_lnR**2))

    W_int, error_w_int = integrate.quad(
        f_log,
        -10 * sigma_lnR,
        10 * sigma_lnR,
        epsabs=1e-12,
        epsrel=1e-10,
        limit=1000
    )

    coeff = 9* np.exp(-9*sigma_lnR**2) / (np.sqrt(2*np.pi) * sigma_lnR * kappa**6)

    return coeff * W_int


def main():
    sigma_lnR = [0.2, 0.5, 1.0]
    ## kappa 采样建议用对数采样，线性采样会导致10^-3处样本点极少
    ## 但是考虑到对于W_bar,在kappa=20处会有明显振荡，对数采样在这个附近取点较稀疏，可能会带来一点问题。
    kappa = np.logspace(-3, np.log10(20), 1000)
    # kappa = np.linspace(1e-3, 20, 1000)

    top_hat_W = np.array([
                top_hat_window_function_W(k) for k in kappa
            ])

    plt.figure()
    plt.loglog(kappa, top_hat_W**2, label=r'top-hat', linestyle='--')

    for s in sigma_lnR:
        # W_kR_bar = np.array([
        #     averaged_window_function_W(k, s) for k in kappa
        # ])

        # plt.loglog(
        #     kappa,
        #     W_kR_bar**2,
        #     label=rf'$\sigma_{{\ln R}} = {s}$'
        # )

        # plt.xlabel(r"$k\bar{R}$")
        # plt.ylabel(r"$\langle W_R(k)\rangle**2$")
        # plt.xlim(1e-3, 20)
        # plt.ylim(1e-10, 10)
        # plt.legend()

        

        W_kR2_bar = np.array([
            averaged_squared_window_function_W_2(k, s) for k in kappa
        ])

        plt.loglog(
            kappa,
            W_kR2_bar,
            label=rf'$\sigma_{{\ln R}} = {s}$'
        )

        plt.xlabel(r"$k\bar{R}$")
        plt.ylabel(r"$\langle W_R(k)\rangle**2$")
        plt.xlim(1e-3, 20)
        plt.ylim(1e-6, 1e4)
        plt.legend()

    plt.show()

if __name__ == "__main__":
    main()