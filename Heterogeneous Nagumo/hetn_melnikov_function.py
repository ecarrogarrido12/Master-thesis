import numpy as np
from scipy.optimize import root_scalar
from scipy.integrate import quad
import matplotlib.pyplot as plt

heterogeneity = 'sech'
seed = 42


# Definition of the heterogeneity
def s(x, x0):
    if heterogeneity == 'sech': return 10*np.sqrt(1 - np.tanh(10*(x - x0)) ** 2)
    if heterogeneity == 'periodic': return np.sin(np.pi * x)
    return 0 * x


# Definition of u_0 and its derivative
def u_h(x, phi):
    return np.tanh((x - phi) / np.sqrt(2))


def u_h_prime(x, phi):
    return (1 - np.tanh((x - phi) / np.sqrt(2)) ** 2) / np.sqrt(2)


# Compute the Melnikov function
def integrand(z, phi, x0):
    return u_h(z, phi) * s(z, x0) * u_h_prime(z, phi)


def M(phi, x0, gamma):
    I = quad(integrand, -np.inf, np.inf, args=(phi, x0), limit=500)
    return -2 * gamma - I[0]


def main():
    # Physical parameters
    gamma_prm = 0.1
    x0 = -5

    # Root finding
    if heterogeneity == 'sech' and gamma_prm == 0.1:
        root1 = root_scalar(M, bracket=[-2.5, -2], method='brentq', args=(x0, gamma_prm))
        root2 = root_scalar(M, bracket=[-5, -4], method='brentq', args=(x0, gamma_prm))
        print(root1.root, root2.root)

    # Plot Melnikov function
    phi_vals = np.linspace(-10, 10, 5000)
    M_vals = [M(phi, x0, gamma_prm) for phi in phi_vals]
    plt.figure(1, figsize=(8, 5))
    plt.plot(phi_vals[:], M_vals, 'r')
    if heterogeneity == 'sech' and gamma_prm == 0.1:
        plt.plot(root1.root, 0, 'ko')  # black dot
        plt.plot(root2.root, 0, 'ko')
        plt.text(root1.root, 0.1, f'{root1.root:.3f}', ha='left', va='bottom', fontsize=12)
        plt.text(root2.root, 0.1, f'{root2.root:.3f}', ha='right', va='bottom', fontsize=12)
    plt.axhline(0, color='black', linestyle='--')
    plt.axhline(-2 * gamma_prm, color='red', linestyle='--', label=r'$-2\gamma$')
    plt.xlabel(r'$\phi$', fontsize=18)
    plt.xlim(-10, 10)
    plt.ylabel(r'$\mathcal{M}(\phi)$', fontsize=18)
    plt.savefig('Plots//HeN_MF_' + heterogeneity + '_.png')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
