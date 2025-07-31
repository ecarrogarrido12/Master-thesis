import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Physical parameters
gamma_prm = 1 / (3 * np.sqrt(6)) + 0.01
sgn_gamma_prm = 1 if gamma_prm > 0 else (0 if gamma_prm == 0 else -1)
x0 = -5
a = 100


# Closed-form of the roots
def phi_star():
    psi = (1 / 3) * np.arcsin(- 3 * np.sqrt(6) * gamma_prm)
    y1 = (2 / np.sqrt(3)) * np.sin(psi)
    y2 = -(2 / np.sqrt(3)) * np.sin(psi + sgn_gamma_prm * np.pi / 3)
    return x0 - np.sqrt(2) * np.arctanh([y1, y2])


# Definition of the heterogeneity
def s(x):
    return a*np.sqrt(1 - np.tanh(a*(x - x0)) ** 2)


# Definition of u_h and its derivative
def u_h(x, phi):
    return np.tanh((x - phi) / np.sqrt(2))


def u_h_prime(x, phi):
    return (1 - np.tanh((x - phi) / np.sqrt(2)) ** 2) / np.sqrt(2)


# Computation of the Melnikov function for the dirac delta heterogeneity
def M_delta(phi):
    return -2 * gamma_prm - u_h(x0, phi) * u_h_prime(x0, phi)


# Computation of the Melnikov function for general heterogeneity
def integrand(z, phi):
    return u_h(z, phi) * s(z) * u_h_prime(z, phi)


def M(phi):
    I = quad(integrand, -np.inf, np.inf, args=(phi, x0), limit=500)
    return -2 * gamma_prm - I[0]


def main():
    # Data structures
    phi = np.linspace(-15, 10, 500)

    # Plot the Melnikov function with its roots
    plt.figure(1, figsize=(8, 5))
    plt.plot(phi[:], M_delta(phi), 'r')
    plt.plot(phi[:], M(phi), 'b--')
    plt.xlabel('$\phi$', fontsize=18)
    plt.ylabel('$M(\phi)$', fontsize=18)
    plt.axhline(y=0, color='k', linestyle='--')
    if abs(gamma_prm) <= 1 / (3 * np.sqrt(6)): plt.scatter(phi_star(), [0, 0], marker='x', color='k', s=80)
    plt.savefig('Plots//HeN_MF_delta_' + str(gamma_prm) + '.png')
    plt.show()


if __name__ == "__main__":
    main()
