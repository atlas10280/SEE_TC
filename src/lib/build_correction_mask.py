import numpy as np
from numpy.linalg import lstsq

def build_correction_mask(estimated_flat_field, degree = 12):
    FF_est = estimated_flat_field.copy()
    # Create 32x32 grid of x, y coordinates
    grid_size = FF_est.shape[0]
    x = np.linspace(-1, 1, grid_size)
    y = np.linspace(-1, 1, grid_size)
    X, Y = np.meshgrid(x, y)

    # Generate example data from a polynomial surface
    # Z = correction_factor
    Z = FF_est

    # Flatten grid data for fitting
    x_flat = X.flatten()
    y_flat = Y.flatten()
    z_flat = FF_est.flatten()

    # Fit polynomial of degree d
    coeffs = fit_polynomial_2d(x_flat, y_flat, z_flat, degree)

    # Create fitted surface
    Z_fitted = polynomial_2d(X, Y, coeffs, degree)
    return(Z_fitted)

def polynomial_2d(x, y, coeffs, degree):
    """
    Evaluate a 2D polynomial.
    """
    poly = sum(
        coeffs[k] * x**(k // (degree + 1)) * y**(k % (degree + 1))
        for k in range(len(coeffs))
    )
    return poly

def fit_polynomial_2d(x, y, z, degree=2):
    """
    Fits a 2D polynomial of a given degree to the data.
    """
    # Generate polynomial basis terms
    num_terms = (degree + 1) * (degree + 1)
    G = np.zeros((x.size, num_terms))
    for i in range(degree + 1):
        for j in range(degree + 1):
            G[:, i * (degree + 1) + j] = (x ** i) * (y ** j)

    # Solve for the coefficients
    coeffs, _, _, _ = lstsq(G, z, rcond=None)
    return coeffs