import numpy as np
import matplotlib.pyplot as plt


# Define the Hamiltonian H(u1, u2)
def H(u1, u2):
    return (1 / 4) * u1 ** 4 - (1 / 2) * u1 ** 2 - (1 / 2) * u2 ** 2


def main():
    # Create a grid of u1 and u2 values
    u1 = np.linspace(-2, 2, 1000)
    u2 = np.linspace(-2, 2, 1000)
    U1, U2 = np.meshgrid(u1, u2)

    # Compute H values on the grid
    H_values = H(U1, U2)

    # Specify the contour levels (adjust as needed)
    levels = [-0.5, -0.25, -0.1]

    # Plot the contour lines
    plt.figure(figsize=(8, 5))
    plt.contour(U1, U2, H_values, levels=levels, colors='k')
    circle_points = [0, -1, 1]
    for point in circle_points:
        plt.plot(point, 0, markersize=10, markeredgewidth=1.5, markeredgecolor='red', markerfacecolor='none')
        plt.text(point, 0.005, f'{point}', horizontalalignment='center', fontsize=18, color='red')
    plt.xlabel(r'$u^*$', fontsize=18)
    plt.ylabel(r'$u^*_x$', rotation=0, labelpad=20, fontsize=18)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.savefig("Plots//level_sets.png")
    plt.show()


if __name__ == "__main__":
    main()
