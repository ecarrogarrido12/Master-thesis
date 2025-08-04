import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad


# Definition of the heterogeneity
def s(x, x0):
    return np.sqrt(1 - np.tanh((x - x0)) ** 2)


# Definition of u_h
def u_h(x, phi):
    return np.tanh((x - phi) / np.sqrt(2))


# Definition of psi_b and psi_u
def psi_b(x, phi):
    return (1 - np.tanh((x - phi) / np.sqrt(2)) ** 2) / np.sqrt(2)


def v(x, phi):
    return 3 * (x - phi) / 4 + np.sinh(np.sqrt(2) * (x - phi)) / np.sqrt(2) + np.sqrt(2) * np.sinh(
        2 * np.sqrt(2) * (x - phi)) / 16


def psi_u(x, phi):
    return v(x, phi) * psi_b(x, phi)


# Definition of A and B
def integrand_A(z, x0, phi, gamma):
    return (gamma + s(z, x0) * u_h(z, phi)) * psi_u(z, phi)


def A(x, x0, phi, gamma):
    result = np.zeros_like(x, dtype=float)
    for i, xi in enumerate(x):
        I = quad(integrand_A, phi, xi, args=(x0, phi, gamma), limit=500)
        result[i] = I[0]
    return result


def integrand_B(z, x0, phi, gamma):
    return (gamma + s(z, x0) * u_h(z, phi)) * psi_b(z, phi)


def B(x, x0, phi, gamma):
    result = np.zeros_like(x, dtype=float)
    for i, xi in enumerate(x):
        I = quad(integrand_B, -np.inf, xi, args=(x0, phi, gamma), limit=500)
        result[i] = -I[0]
    return result


# Definition of the front
def u_mu(x, x0, phi_star, gamma, mu):
    p = A(x, x0, phi_star, gamma) * psi_b(x, phi_star) + B(x, x0, phi_star, gamma) * psi_u(x, phi_star)
    return u_h(x, phi_star) + mu * p


def main():
    # Physical parameters
    mu = 0.1
    gamma = 0.1
    x0 = -5
    phi_star = -2.072957588537434

    # Data structures
    x = np.linspace(-10, 10, 1000)
    y = u_mu(x, x0, phi_star, gamma, mu)

    # Plot front solution (approximation)
    plt.plot(x[:], y, 'r')
    plt.plot(x[:], u_h(x, phi_star), '--', label="u_h(x)")
    plt.xlabel("x", fontsize=18)
    plt.ylabel(r"$u_\mu$", fontsize=18)
    plt.legend()
    plt.savefig("HeN_SF_sech.png")
    plt.show()


if __name__ == "__main__":
    main()
