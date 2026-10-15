# test_Plot_PR
from os import name

import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt

def Reff_to_Rbar(Reff, sigma_lnR):
    return Reff * np.exp(-4 * sigma_lnR**2)

def Rbar_to_Reff(Rbar, sigma_lnR):
    return Rbar * np.exp(4 * sigma_lnR**2)

def Reff_to_R0(Reff, sigma_lnR):
    return Reff * np.exp(-2.5 * sigma_lnR**2)


def bubble_radius_distribution_P(sigma_lnR, R_bar, R):
        """
        Calculate the the bubble radius distribution P(R) of the H II region.

        Args:
            R_bar (float): characteristic size, default = 5.0
            R (float or array-like): Radius values.
            sigma_lnR (float): width of the distribution, default = ln(2)

        Returns:
            float or array-like: The PDF evaluated at the given radius values.
        """


        P_R = (
                1 / (R * sigma_lnR * np.sqrt(2 * np.pi))
            ) * np.exp(
                    - (np.log(R) - np.log(R_bar))**2 / (2 * sigma_lnR**2)
                )
        return P_R

def main():
    sigma_lnR = [0.2, 1.0]
    color = ['orange', 'blue']
    R_eff = 55 # Mpc
    R = np.logspace(-1, 2, 1000)

    plt.figure()

    for i,s in enumerate(sigma_lnR):
        R_bar = Reff_to_Rbar(R_eff, s)
        R_0 = Reff_to_R0(R_eff, s)
        P_R = bubble_radius_distribution_P(s, R_bar, R)

        plt.semilogx(R, R*P_R, label=rf'$\sigma_{{\ln R}} = {s}$', color=color[i])
        plt.axvline(x=R_0, label=rf'$R_0$ for $\sigma_{{\ln R}} = {s}$', color=color[i], linestyle='--')

    plt.xlabel(r"$R[Mpc]$")
    plt.ylabel(r"$R P(R)$")
    plt.legend()

    plt.show()

if __name__ == "__main__":
    main()
