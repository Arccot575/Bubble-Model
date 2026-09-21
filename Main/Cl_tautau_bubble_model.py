import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.interpolate import interp1d

import camb
from camb import model


def x_bar_e(z, y_re=5.0, delta_y=19.0):
    """
    Calculate the average ionization fraction of hydrogen in a plasma given the redshift z.

    Parameters:
    z (float): Redshift value.
    y_re (float): Reference value.
    delta_y (float): Width of the distribution.

    Returns:
    float: The average ionization fraction of hydrogen.
    """
    def y(z):
        y = (1 + z) ** (3/2)
        return y

    x_bar_e = 0.5 * (1 - np.tanh((y(z) - y_re) / delta_y))
    return x_bar_e


def bubble_radius_distribution_P(R, R_bar=5, sigma_lnR=np.log(2)):
    """
    Calculate the the bubble radius distribution P(R) of the H II region.

    Parameters:
    R_bar (float): characteristic size, default = 5.0
    R (float or array-like): Radius values.
    sigma_lnR (float): width of the distribution, default = ln(2)

    Returns:
    float or array-like: The PDF evaluated at the given radius values.
    """
    # Calculate the PDF using the log-normal distribution formula
    P_R = (1 / (R * sigma_lnR * np.sqrt(2 * np.pi))) * np.exp(- (np.log(R) - np.log(R_bar))**2 / (2 * sigma_lnR**2))
    return P_R


# === Parameter Set ===
R_max = 60 # Max radius for integrations and ploting
R_bar = 5  # characteristic size
sigma_lnR = np.log(2)  # width of the distribution
b = 1.0 # bubble bias, for simplicity, the model assumes it to be a constant


R_values = np.linspace(0.1, R_max, 100)


# === Now get matter power spectra and sigma8 at redshift 0 and 0.8 ===
# parameters can all be passed as a dict as above, or you can call
# separate functions to set up the parameter object
pars = camb.set_params(H0=67.5, ombh2=0.022, omch2=0.122, ns=0.965)
# Note non-linear corrections couples to smaller scales than you want
pars.set_matter_power(
    redshifts=[6,7,8,10],
    kmax=2.0
)

# Linear spectra
pars.NonLinear = model.NonLinear_none
results = camb.get_results(pars)
kh,z,Pk = results.get_matter_power_spectrum(
    minkh=1e-4,
    maxkh=1,
    npoints=300
)

# Interpolate matter power spectrum at target redshift (z=8 -> index 2)
z_index = 2  # z=8 corresponds to the third redshift in [6,7,8,10]
P_matter_interp = interp1d(kh, Pk[z_index, :], kind='linear', fill_value='extrapolate')


def bubble_volume_V(R):
    """
    The volume function for the ionized bubble, which is assumed as a sphere.
    """
    return 4/3 * np.pi * R ** 3

def top_hat_window_function_W(k, R):
    """
    The Fourier transform of a real-space tophat window function with radius R

    Parameters:
    k = ell/chi:  ??
    R (float or array-like): Radius values.
    """
    kR = np.asarray(k) * np.asarray(R)

    # Handle kR -> 0 limit: W -> 1
    # Use np.where to avoid division by zero

    with np.errstate(invalid='ignore', divide='ignore'):
        W_val = np.where(np.abs(kR) < 1e-10, 1.0,
                        3/(kR)**3 * (np.sin(kR) - kR * np.cos(kR)))
    return W_val


def averaged_bubble_volume_V():
    """
    Bubble volume averaged over the distribution P(R)
    """

    def f(R):
        return bubble_radius_distribution_P(R)*bubble_volume_V(R)

    Vb, error_Vb = integrate.quad(
        f,
        0,
        R_max
    )

    return Vb

def averaged_window_function_W(k,R):
    
    


# === F(k): single-k scalar version ===
def one_bubble_F(k):
    """
    Compute F(k) for a single scalar k using numerical integration.
    """
    def f(R):
        return bubble_radius_distribution_P(R, R_bar, sigma_lnR) * (bubble_volume_V(R) * top_hat_window_function_W(k, R))**2
    
    Int_F, error_F = integrate.quad(
        f,
        0,
        R_max
    )
    
    V_b = averaged_bubble_volume_V
    F_val = Int_F/V_b

    return F_val

def one_bubble_G(k):


    return  0

# === I(k): single-k scalar version ===
def two_bubble_I_single(k):
    """
    Compute I(k) for a single scalar k using numerical integration.
    """    
    def f(R):
        return bubble_radius_distribution_P(R) * volume_V(R) * top_hat_window_function_W(k, R)
    
    Int_I, error_I = integrate.quad(
        f,
        0,
        R_max
    )
    V_b = averaged_bubble_volume_V()
    
    I_val = bubble_radius_distribution_P(R_bar) * Int_I/V_b

    return I_val

# Vectorize: allows I to accept array k
I_vec = np.vectorize(two_bubble_I_single)

def I(k):
    return I_vec(k)


# === Fast power spectrum functions using pre-computed interpolators ===

def one_bubble_P_1b(k, x_e):
    """
    1-bubble term of the ionization power spectrum.
    """
    k = np.atleast_1d(k)
    F_val = F_interp(k)
    G_val = G_interp(k)

    P1_val = x_e * (1 - x_e) * (F_val + G_val)

    return P1_val

def two_bubble_P_2b(k, x_e):
    """
    2-bubble term of the ionization power spectrum.
    """
    k = np.atleast_1d(k)
    I_val = I_interp(k)
    P_val = P_matter_interp(k)  # matter power spectrum from CAMB

    P2_val = ((1-x_e) * np.log(1-x_e) * I_val-x_e)**2 * P_val

    return P2_val

def total_P_DeltaXe(k, x_e):
    """
    Total power spectrum = 1-bubble + 2-bubble terms
    """
    P1_val = one_bubble_P_1b(k, x_e)
    P2_val = two_bubble_P_2b(k, x_e)

    return P1_val + P2_val



# plt.figure(figsize=(8, 6))
# k_test = np.logspace(-2, 0, 30)  # fewer points for faster test
# plt.loglog(k_test, P_DeltaXe(k_test, x_bar_e(8)), label=f'P_1b at z=8')
# plt.xlabel('k [1/Mpc]')
# plt.ylabel('$$P_{\\Delta X_e\\Delta X_e}$$ [Mpc^3]')
# plt.legend()
# plt.grid()
# plt.show()


from astropy.cosmology import Planck18
def chi_of_z(z):
    return Planck18.comoving_distance(z).value


def dchi_dz(z):
    """
    Derivative of comoving distance with respect to redshift: dχ/dz = c / H(z)
    """
    return Planck18.hubble_distance.value / Planck18.efunc(z)


sigma_T = 6.98e-74 #Mpc^2
n_p0 = 7.4e66 #Mpc^-3


def Cl_tautau(ell):

    def integrand(z):

        chi = chi_of_z(z)

        a = 1/(1+z)

        k = (ell + 0.5)/chi

        x_e = x_bar_e(z)

        return (
            sigma_T**2
            * n_p0**2
            / (a**4 * chi**2)
            * P_DeltaXe(k, x_e)
            * dchi_dz(z)
        )

    return integrate.quad(
        integrand,
        6,
        10
    )[0]

if __name__ == "__main__":
    ell_grid = np.arange(2, 3000)

    Cl_tautau_grid = np.array([
        Cl_tautau(ell)
        for ell in ell_grid
    ])

    plt.semilogy(
        ell_grid,
        ell_grid*(ell_grid + 1)*Cl_tautau_grid / (2 * np.pi)
    )

    plt.xlabel(r"$\ell$")
    plt.ylabel(r"$\ell(\ell + 1)C_\ell^{\tau\tau}/(2\pi)$")
    plt.xlim([2, 3000])
    plt.show()

