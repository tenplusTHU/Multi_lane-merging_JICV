"""
    Description: ColorBar
    Author: Tenplus
    Create-time: 2023-07-07
    Update-time: 2023-07-07
    Note: # v1.0 colorBar
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_colorBar():
    color_map = plt.cm.get_cmap(name='brg')
    color_range = [0.5, 1]  # corresponding to v: [0, v_max]

    fig = plt.figure(figsize=(7, 0.8))
    ax_bar = fig.add_subplot()
    ax_bar.set_yticks([])
    ax_bar.set_ylim([0, 1])
    ax_bar.set_xlim([0.5, 1])
    ax_bar.tick_params(axis="x", direction="in", which="major")

    for i in np.linspace(0.5, 1, 1000, endpoint=True):
        ax_bar.vlines(i, 0, 1, colors=color_map(i))

    ax_bar.set_xticks(np.linspace(0.5, 1, 6, endpoint=True))
    ax_bar.set_xticklabels(np.arange(0, 26, 5))

    ax_bar.xaxis.set_tick_params(labelsize=9)
    ax_bar.set_xlabel('Velocity ColorBar (m/s)', size=9)

    plt.tight_layout()
    plt.savefig('colorBar.svg')
    plt.show()


if __name__ == "__main__":
    plot_colorBar()
