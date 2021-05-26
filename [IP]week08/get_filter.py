import numpy as np


def my_get_Gaussian2D_mask(msize, sigma=1):
    n = msize//2
    y, x = np.mgrid[-n:n+1, -n:n+1]

    gaus2D = (1/(2*np.pi*(sigma**2)))*np.exp(-(x**2+y**2)/(2*sigma**2))
    gaus2D /= np.sum(gaus2D)

    return gaus2D