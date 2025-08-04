import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Numerical parameters
N = 81
L = 40
n_steps = 9

# Initialize the array that storages the states u(t)
U = []
s = []

# Read file
with open("simulation_data_barkley_model.txt", "r") as file:
    # Read heterogeneity
    for line in file:
        if line.startswith("#"):
            break
        else:
            row = list(map(float, line.split())); s.append(row)
    s = np.array(s)
    # Read states
    u_t = []
    for line in file:
        if line.startswith("#"):
            u_t = np.array(u_t)
            U.append(u_t)
            u_t = []
        else:
            row = list(map(float, line.split()))
            u_t.append(row)
    U = np.array(U)

# Plot simulations
for n in range(n_steps):
    plt.figure(n + 1, (5, 5))
    extent = [0, L, 0, L]
    plt.imshow(U[n], cmap='viridis', origin='lower', extent=extent, vmin=0, vmax=1)
    plt.xlabel('x', fontsize=18)
    plt.ylabel('y', fontsize=18)
    plt.savefig('het_bm_sw_'+str(n + 1)+'.png')


# Create animation
fig, ax = plt.subplots(figsize=(8, 4.5))
extent = [0, L, 0, L]
cax = ax.imshow(U[0:], cmap='viridis', origin='lower', extent=extent)
cbar = fig.colorbar(cax)
cbar.set_label(r'u((x, y), t)', fontsize=18)
ax.set_xlabel('x', fontsize=18)
ax.set_ylabel('y', fontsize=18)


def update(frame):
    cax.set_data(U[frame])
    return cax,


ani = FuncAnimation(fig, update, frames=len(U[:]), blit=False)
ani.save('barkley_model_spiral_wave_breakup.mp4', writer='ffmpeg')