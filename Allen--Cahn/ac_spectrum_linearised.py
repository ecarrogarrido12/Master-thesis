import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh

# Type of test: 'zero', 'one' or 'tanh', depending on the stationary solution u^*(xi).
ss = 'one'

# Numerical parameters
L = 100  # Domain [-L, L]
N = 5000  # Number of grid points
h = 2 * L / (N - 1)  # Step size


# Definition of u^*(\xi)
def u_star(x):
    if ss == 'one': return np.ones_like(x)
    if ss == 'zero': return np.zeros_like(x)
    return np.tanh(x / np.sqrt(2))


def main():
    # Data structure
    x = np.linspace(-L, L, N)

    # Matrix diagonals
    main_diag = -2 / (h ** 2) + (1 - 3 * u_star(x) ** 2)
    off_diag = np.ones(N - 1) / (h ** 2)

    # Impose no-flux - Neumann BCs
    main_diag[-1] = main_diag[-1]/2
    main_diag[0] = main_diag[0]/2

    # Construct sparse matrix
    A = diags([off_diag, main_diag, off_diag], offsets=[-1, 0, 1], format='csr')

    # Compute eigenvalues
    eigenvalues = eigsh(A, k=1000, which='SM', return_eigenvectors=False)
    plt.figure(1, figsize=(8, 5))
    plt.plot(np.real(eigenvalues), np.imag(eigenvalues), 'r.')
    plt.xlabel(r'Re $\lambda$', fontsize=18)
    if ss == 'one':
        plt.plot(-2, 0, 'o', markersize=10, markeredgewidth=1.5, markeredgecolor='black', markerfacecolor='none')
        plt.text(-2, 0.005, f'{-2}', horizontalalignment='center', fontsize=18, color='black')
    elif ss == 'zero':
        plt.plot(1, 0, 'o', markersize=10, markeredgewidth=1.5, markeredgecolor='black', markerfacecolor='none')
        plt.text(1, 0.005, f'{1}', horizontalalignment='center', fontsize=18, color='black')
    elif ss == 'front':
        circle_points = [0, -2, -3 / 2]
        for point in circle_points:
            plt.plot(point, 0, 'o', markersize=10, markeredgewidth=1.5, markeredgecolor='black', markerfacecolor='none')
            plt.text(point, 0.005, f'{point}', horizontalalignment='center', fontsize=18, color='black')
    plt.xlim(-4, 1)
    plt.ylabel(r'Im $\lambda$', fontsize=18)
    plt.grid()
    plt.savefig('Plots//AC_spectrum_' + ss + '.png')
    plt.show()


if __name__ == "__main__":
    main()
