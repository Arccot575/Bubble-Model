#Bubble-Model
import numpy as np

class BubbleModel():
    """
    Class for the bubble model cauculation, including
    """
    def __init__(self, R_bar: float, R_max: float, sigma_lnR: float, bias: float, ):
        """
        Class initialisation, 

        Args:
            R_bar: Mean ionized bubble radius
            R_max: Maxium radius used in integral and calculation
            sigma_lnR: Standard Deviation of the log-normal distribution of bubbles
            bias: bubble bias
        """
        self.R_bar = R_bar
        self.R_max = R_max
        self.sigma_lnR = sigma_lnR
        self.b = bias


    def x_bar_e(z, y_re=5.0, delta_y=19.0):
        """
        Calculate the average ionization fraction of hydrogen in a plasma given the redshift z.

        Args:
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


    def bubble_radius_distribution_P(self, R):
        """
        Calculate the the bubble radius distribution P(R) of the H II region.

        Args:
            R_bar (float): characteristic size, default = 5.0
            R (float or array-like): Radius values.
            sigma_lnR (float): width of the distribution, default = ln(2)

        Returns:
            float or array-like: The PDF evaluated at the given radius values.
        """

        P_R = (1 / (R * sigma_lnR * np.sqrt(2 * np.pi))) * np.exp(- (np.log(R) - np.log(R_bar))**2 / (2 * sigma_lnR**2))
        return P_R

    @staticmethod
    def _bubble_volume_V(R):
        """
        The volume function for the ionized bubble, which is assumed as a sphere.
        """
        return 4/3 * np.pi * R ** 3

    @staticmethod
    def _top_hat_window_function_W(k, R):
        """
        The Fourier transform of a real-space tophat window function with radius R

        Arg:
            k = ell/chi: Fourier conterpart of relative position x
            R (float or array-like): Radius values.
        """
        kR = np.asarray(k) * np.asarray(R)

        # Handle kR -> 0 limit: W -> 1
        # Use np.where to avoid division by zero
        with np.errstate(invalid='ignore', divide='ignore'):
            W_val = np.where(np.abs(kR) < 1e-10, 1.0,
                            3/(kR)**3 * (np.sin(kR) - kR * np.cos(kR)))
            
        return W_val


    def averaged_bubble_volume_V(self):
        """
        Bubble volume averaged over radius R with given distribution P(R). 
        """

        V_bar = 4/3*np.pi * self.R_bar**3 * np.exp(4.5*self.sigma_lnR**2)

        return V_bar

    def one_bubble_F(k):
        """
        Compute F(k) for a single scalar k using numerical integration.
        """
        def f(R):
            return bubble_radius_distribution_P(R, R_bar, sigma_lnR) * (_bubble_volume_V(R) * _top_hat_window_function_W(k, R))**2
        
        Int_F, error_F = integrate.quad(
            f,
            0,
            R_max
        )
        
        V_b = averaged_bubble_volume_V()
        F_val = Int_F/V_b

        return F_val

    def one_bubble_G():

        return G_val

    def one_bubble_power_spectra():

        return P_1b

    def two_bubble_power_spectra(self):
        
        return P_2b

    def total_power_spectra():
        return P_bubble
    

    