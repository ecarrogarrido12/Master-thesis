import numpy as np
from scipy.integrate import quad
from scipy.linalg import eigh
import matplotlib.pyplot as plt

# Numerical parameters
L = 100  # Domain [-L, L]
N = 5000  # Number of grid points
h = 2 * L / (N - 1)  # Step size

# Physical parameters
mu = 0.1
gamma = 0.1
x0 = -5  # Localization of the heterogeneity
phi_star = -4.3820971947072636  # Obtained from the Melnikov function


# Unstable: -4.3820971947072636
# Stable: -2.072957588537434


# Definition of the background state
def u_minus_star():
    # Compute psi
    psi = (1 / 3) * np.arcsin(- (3 * np.sqrt(3) / 2) * gamma * mu)
    # Compute u_+^*
    return -(2 / np.sqrt(3)) * np.sin(psi + np.pi / 3)


def u_plus_star():
    # Compute psi
    psi = (1 / 3) * np.arcsin(- (3 * np.sqrt(3) / 2) * gamma * mu)
    # Compute u_+^*
    return -(2 / np.sqrt(3)) * np.sin(psi - np.pi / 3)


# Definition of the heterogeneity
def s(x):
    return np.sqrt(1 - np.tanh((x - x0)) ** 2)


# Definition of u_h and u_h_prime
def u_h(x, phi):
    return np.tanh((x - phi) / np.sqrt(2))


def u_h_prime(x, phi):
    return (1 - np.tanh((x - phi) / np.sqrt(2)) ** 2) / np.sqrt(2)


# Definition of psi_b and psi_u
def psi_b(x, phi):
    return (1 - np.tanh((x - phi) / np.sqrt(2)) ** 2) / np.sqrt(2)


def v(x, phi):
    return 3 * (x - phi) / 4 + np.sinh(np.sqrt(2) * (x - phi)) / np.sqrt(2) + np.sqrt(2) * np.sinh(
        2 * np.sqrt(2) * (x - phi)) / 16


def psi_u(x, phi):
    return v(x, phi) * psi_b(x, phi)


# Definition of A and B
def integrand_A(z, phi):
    return (gamma + s(z) * u_h(z, phi)) * psi_u(z, phi)


def A(x, phi):
    if np.isscalar(x):
        I = quad(integrand_A, phi, x, args=(phi,), limit=1000)
        return I[0]
    else:
        result = np.zeros_like(x, dtype=float)
        for i, xi in enumerate(x):
            I = quad(integrand_A, phi, xi, args=(phi,), limit=1000)
            result[i] = I[0]
        return result


def integrand_B(z, phi):
    return (gamma + s(z) * u_h(z, phi)) * psi_b(z, phi)


def B(x, phi):
    if np.isscalar(x):
        I = quad(integrand_B, -np.inf, x, args=(phi,), limit=1000)
        return -I[0]
    else:
        result = np.zeros_like(x, dtype=float)
        for i, xi in enumerate(x):
            I = quad(integrand_B, -np.inf, xi, args=(phi,), limit=500)
            result[i] = -I[0]
        return result


# Definition of p and the front
def p(x):
    return A(x, phi_star) * psi_b(x, phi_star) + B(x, phi_star) * psi_u(x, phi_star)


def u_mu(x):
    if np.isscalar(x):
        return u_h(x, phi_star) + mu * p(x)
    else:
        result = np.empty_like(x, dtype=float)
        result[x > 10] = u_plus_star()
        result[x < -18] = u_minus_star()
        mask = (x >= -18) & (x <= 10)
        result[mask] = u_h(x[mask], phi_star) + mu * p(x[mask])
        return result


# Compute numerical approximation of the principal eigenvalue
def numerical_approximation_pe(x):
    # Construct the discretized Laplacian
    main_diag = -2 * np.ones(N) / (h ** 2)
    off_diag = np.ones(N - 1) / (h ** 2)
    K = np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)

    # Construct the discretized reaction operator
    M = np.diag(1 - 3 * (u_mu(x) ** 2) + mu * s(x))

    # Impose no-flux - Neumann BCs
    T = K + M
    T[-1:] /= 2
    T[:1] /= 2

    # Compute principal eigenvalue
    eigenvalues = eigh(T, eigvals_only=True)
    return max(eigenvalues.real)


# Plot theoretical approximation of the principal eigenvalue
def integrand_lambda_hat(z):
    return (s(z) - 6 * u_h(z, phi_star) * p(z)) * (u_h_prime(z, phi_star)) ** 2


def theoretical_approximation_pe():
    one_o_u_h_prime_square = (3 * np.sqrt(2)) / 4
    I = quad(integrand_lambda_hat, -18, 10, limit=500)[0]
    lambda_hat = I * one_o_u_h_prime_square
    return mu * lambda_hat


def main():
    # Data structure
    x = np.linspace(-L, L, N)
    # Compute principal eigenvalues
    theo_approx = theoretical_approximation_pe()
    num_approx = numerical_approximation_pe(x)
    print("Theoretical approximation", theo_approx)
    print("Numerical approximation", num_approx)
    # Plot principal eigenvalue
    plt.figure(1, figsize=(8, 5))
    plt.plot(0, 0, '.', markersize=12, color='red', label=r'$\mu = 0$')
    plt.scatter(theo_approx, 0, marker='x', color='black', s=50, label=fr'$\mu = {mu:.1f}$ (theoretical)')
    plt.scatter(num_approx, 0, marker='o', s=50, facecolors='none',
                edgecolors='blue', label=fr'$\mu = {mu:.1f}$ (numerical)')
    plt.xlim(-0.03, 0.03)
    plt.ylim(-0.02, 0.02)
    plt.xlabel(r'Re $\lambda$', fontsize=18)
    plt.ylabel(r'Im $\lambda$', fontsize=18)
    plt.grid(True)
    plt.legend(fontsize=18)
    plt.savefig('Plots\\shifted_eigenvalue_unstable_' + str(mu) + '.png')
    plt.show()


if __name__ == "__main__":
    main()
