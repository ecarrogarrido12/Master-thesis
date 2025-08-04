import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.integrate import solve_ivp
from matplotlib.ticker import MaxNLocator
from scipy.integrate import quad

# Type of test
heterogeneity = 'sech'

# Seed
seed = 42

# Physical parameters
mu = 0.1
gamma = 0.01

# Numerical parameters
L = 10.0
N = 500
h = 2 * L / N
T = 1000.0
t_steps = 40000
x_0 = -L / 2


# Set initial condition
def u_0(x):
    # Compute psi
    psi = (1 / 3) * np.arcsin(- (3 * np.sqrt(3) / 2) * gamma)
    # Compute u_+^*
    u_plus_star = -(2 / np.sqrt(3)) * np.sin(psi - np.pi / 3)
    return u_plus_star * np.ones_like(x)


# Set heterogeneity
def s(x):
    if heterogeneity == 'periodic':
        return np.sin(np.pi * x)
    if heterogeneity == 'random':
        np.random.seed(seed)
        return np.random.normal(loc=0.0, scale=1.0, size=len(x))
    if heterogeneity == 'sech':
        return np.sqrt(1 - np.tanh((x - x_0)) ** 2)
    return np.zeros_like(x)


# Physical function
def f(u, x):
    return u - u ** 3 + gamma + mu * s(x) * u


def rhs(t, u, Ah, x):
    return Ah @ u + f(u, x)


# Compute approximated solution
def u_plus_mu(x):
    result = np.zeros_like(x)
    for i, xi in enumerate(x):
        def integrand(z):
            return np.exp(-np.sqrt(2) * np.abs(xi - z)) * s(z)

        I = quad(integrand, -np.inf, np.inf)[0]
        result[i] = u_0(xi) * mu / (2 * np.sqrt(2)) * I
    return u_0(x) + result


def main():
    # Data structures
    x = np.linspace(-L, L, N + 1)
    main_diag = -2.0 * np.ones(N + 1)
    below_diag = 1.0 * np.ones(N)
    upper_diag = np.copy(below_diag)
    # No-flux Neumann bc
    below_diag[-1] = 2.0
    upper_diag[0] = 2.0
    Ah = diags(diagonals=[below_diag, main_diag, upper_diag],
               offsets=[-1, 0, 1],
               shape=(N + 1, N + 1),
               format='csr') * (1 / h ** 2)

    # Solve PDE using Method of lines and Radau
    sol = solve_ivp(fun=rhs,
                    t_span=(0, T),
                    y0=u_0(x),
                    method='Radau',
                    t_eval=np.linspace(*(0, T), t_steps),
                    args=(Ah, x))

    # Plot solution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, sol.y[:, 0], 'b--', label=r"$u(x, t)$ at $t = %i$s" % sol.t[0])
    ax.plot(x, u_plus_mu(x), 'r', label=r'$u_+^\mu(x)$')
    ax.legend(fontsize=12)
    ax.set_xlim(-10, 10)
    ax.set_ylim(0.95, 1.05)
    ax.set_xlabel(r'$x$', fontsize=18)
    ax.tick_params(axis='both', labelsize=12)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    plt.savefig('PBS_' + heterogeneity + '_approx.png')
    plt.show()


if __name__ == '__main__':
    main()
