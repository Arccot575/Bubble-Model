import numpy as np
import matplotlib.pyplot as plt
from Cl_tautau_bubble_model import bubble_radius_distribution_P, volume_V, top_hat_window_function_W, average_volume_V_bar
import scipy.integrate as integrate

R_max = 100  # Maximum radius for integration, adjust as needed

def averaged_window_function_W_kR(k, R = R_bar):
    """
    Compute the averaged window function W(k, R) for given k and R.
    """
    def V_bar(R):
        return volume_V(R)*bubble_radius_distribution_P(R)

    Aver_V, error_Aver_V = integrate.quad(
        V_bar,
        0,
        R_max
    )

    def Wk_bar(R):
        return bubble_radius_distribution_P(R) * volume_V(R) * top_hat_window_function_W(k, R)

    Aver_Wk, error_Aver_Wk = 1/Aver_V * integrate.quad(
        Wk_bar,
        0,
        R_max
    )

    return Aver_Wk

if __name__ == "__main__":
    R_bar = 55 # Mpc
    # Example usage
    k = np.linspace(1e-5, 20, 100)  # Example k values
    kR = k * R_bar
    result = [averaged_window_function_W_kR(k_i, R_bar) for k_i in k]

    plt.figure()
    plt.plot(kR, result**2, label='Averaged Window Function W(k, R)')
    plt.xlabel('kR')
    plt.ylabel('Averaged W(k, R)')
    plt.title('Averaged Window Function')

    plt.xlim(1e-3, 20)
    plt.legend()
    plt.grid()
    plt.show()