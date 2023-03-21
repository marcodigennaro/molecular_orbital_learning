#https://scipython.com/blog/visualizing-the-real-forms-of-the-spherical-harmonics/

import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from scipy.special import sph_harm

plt.rc('text', usetex=True)

# Grids of polar and azimuthal angles
theta = np.linspace(0, np.pi, 100)
phi = np.linspace(0, 2*np.pi, 100)
# Create a 2-D meshgrid of (theta, phi) angles.
theta, phi = np.meshgrid(theta, phi)
# Calculate the Cartesian coordinates of each point in the mesh.
xyz = np.array([np.sin(theta) * np.sin(phi),
                np.sin(theta) * np.cos(phi),
                np.cos(theta)])

def plot_Y(ax, el, m, distortion=1e-3):
    """Plot the spherical harmonic of degree el and order m on Axes ax."""

    # NB In SciPy's sph_harm function the azimuthal coordinate, theta,
    # comes before the polar coordinate, phi.
    Y = sph_harm(abs(m), el, phi, theta)

    # Linear combination of Y_l,m and Y_l,-m to create the real form.
    if m < 0:
        Y = np.sqrt(2) * (-1)**m * Y.imag
    elif m > 0:
        Y = np.sqrt(2) * (-1)**m * Y.real

    Yx, Yy, Yz = np.abs(Y) * xyz

    # Distorting perfect orbitals
    Yx += np.random.normal(0, distortion, size=Yx.shape)
    Yy += np.random.normal(0, distortion, size=Yy.shape)
    Yz += np.random.normal(0, distortion, size=Yz.shape)

    # Colour the plotted surface according to the sign of Y.
    cmap = plt.cm.ScalarMappable(cmap=plt.get_cmap('seismic'))

    cmap.set_clim(-0.5, 0.5)

    ax.plot_surface(Yx, Yy, Yz,
                    facecolors=cmap.to_rgba(Y.real),
                    rstride=2, cstride=2)

    # Draw a set of x, y, z axes for reference.
    ax_lim = 0.5
    ax.plot([-ax_lim, ax_lim], [0,0], [0,0], c='0.5', lw=1, zorder=10)
    ax.plot([0,0], [-ax_lim, ax_lim], [0,0], c='0.5', lw=1, zorder=10)
    ax.plot([0,0], [0,0], [-ax_lim, ax_lim], c='0.5', lw=1, zorder=10)
    # Set the Axes limits and title, turn off the Axes frame.
    #ax.set_title(r'$Y_{{{},{}}}$'.format(el, m))
    ax_lim = 0.5
    ax.set_xlim(-ax_lim, ax_lim)
    ax.set_ylim(-ax_lim, ax_lim)
    ax.set_zlim(-ax_lim, ax_lim)

    x_surf = np.linspace(-ax_lim, ax_lim, 10)
    y_surf = np.linspace(-ax_lim, ax_lim, 10)
    z_surf = np.linspace(-ax_lim, ax_lim, 10)
    X, Y = np.meshgrid(x_surf, y_surf)

    ax.plot_surface(
        X, Y, np.zeros((10,10)), cmap='viridis', alpha=0.2
        )

    ax.axis('off')


fig = plt.figure(figsize=plt.figaspect(1.))
ax = fig.add_subplot(projection='3d')
l, m = 3, 0
plot_Y(ax, l, m)

plt.savefig('Y{}_{}.png'.format(l, m))
#plt.show()

Path('FIGS').mkdir(exist_ok=True)
df = pd.DataFrame()

l = 2
for m in range(-l, l):
    #for m in [0]:
    fig = plt.figure(figsize=plt.figaspect(1.))
    ax = fig.add_subplot(projection='3d')
    plot_Y(ax, l, m)

    # for rotate the axes and update.
    for idx in range(4):
        ang1 = random.uniform(-90, 90)
        ang2 = random.uniform(-180, 180)
        ang3 = random.uniform(-180, 180)
        ax.view_init(ang1, ang2, ang3)
        fig_name = 'd_{}_{}.png'.format(m, idx)
        plt.savefig(f'FIGS/{fig_name}', dpi=100)

        idx_dict = { 'fig_name':fig_name,
                     'm':m+2,
                     'ang1': ang1,
                     'ang2': ang2,
                     'ang3': ang3,
                     }

        idx_df = pd.DataFrame([idx_dict])

        df = pd.concat([df, idx_df], axis=0, ignore_index=True)

df.to_csv('distorted_orbitals.csv')
