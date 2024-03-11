"""
    Description: Functions to draw the figures of vehicle trajectory..
    Author: Tenplus
    Create-time: 2022-02-21
    Update-time: 2023-02-15, # V1.2 新学期 Check 代码
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Arial'


def showTrajs(Trajs_dict, direction=1, FLAG_legend=0, label_size=12, tick_size=11):
    if Trajs_dict:
        """ create figures """
        if direction:
            fig = plt.figure(figsize=(4.5, 6))
            ax1 = fig.add_subplot(311)
            ax2 = fig.add_subplot(312)
            ax3 = fig.add_subplot(313)

            label_font = {'size': label_size}
            # for ax_ in [ax1, ax2, ax3]:
            #     ax_.set_xlabel(r'$t~\mathrm{(s)}$', label_font)
            ax3.set_xlabel(r'$t~\mathrm{(s)}$', label_font)

            ax1.set_ylabel(r'$a~\mathrm{(mm/s^2)}$', label_font)
            ax2.set_ylabel(r'$v~\mathrm{(mm/s)}$', label_font)
            ax3.set_ylabel(r'$p~\mathrm{(mm)}$', label_font)

            for ax_ in [ax1, ax2, ax3]:
                ax_.tick_params(axis='x', labelsize=tick_size)
                ax_.tick_params(axis='y', labelsize=tick_size)

        else:
            fig = plt.figure(figsize=(11, 2.3))
            ax1 = fig.add_subplot(141)
            ax2 = fig.add_subplot(142)
            ax3 = fig.add_subplot(143)
            ax4 = fig.add_subplot(144)

            label_font = {'size': 11}
            for ax_ in [ax1, ax2, ax3]:
                ax_.set_xlabel(r'$t~\mathrm{(s)}$', label_font)

            ax1.set_ylabel(r'$p~\mathrm{(mm)}$', label_font)
            ax2.set_ylabel(r'$v~\mathrm{(mm/s)}$', label_font)
            ax3.set_ylabel(r'$a~\mathrm{(mm/s^2)}$', label_font)

            ax4.set_xlabel(r'$a~\mathrm{(mm/s^2)}$', label_font)
            ax4.set_ylabel(r'$v~\mathrm{(mm/s)}$', label_font)

        """ show trajectories """
        for traj_id, args in Trajs_dict.items():
            traj = args['traj']
            traj = np.array(traj)
            c = args['color'] if 'color' in args else 'k'
            lw = args['lw'] if 'lw' in args else 2
            ls = args['ls'] if 'ls' in args else '-'
            alpha = args['alpha'] if 'alpha' in args else 1
            order = args['order'] if 'order' in args else 5

            ax3.plot(traj[:, 0], traj[:, 1], c=c, lw=lw, ls=ls, alpha=alpha, zorder=order, label=traj_id)
            ax2.plot(traj[:, 0], traj[:, 2], c=c, lw=lw, ls=ls, alpha=alpha, zorder=order, label=traj_id)
            if len(traj[0]) > 3:
                ax1.plot(traj[:-1, 0], traj[1:, 3], c=c, lw=lw, ls=ls, alpha=alpha, zorder=order, label=traj_id)

            # Lane-change point
            LC_t0 = args['LC_t0'] if 'LC_t0' in args else None
            if LC_t0:
                LC_midT = LC_t0 + 2
                k_ = np.argmin(abs(traj[:, 0] - LC_midT))
                ax3.plot(traj[k_, 0], traj[k_, 1], c=c, marker='o', markersize=3)

            if direction == 0:
                ax4.plot(traj[:, 3], traj[:, 2], c=c, lw=lw, ls=ls, alpha=alpha, zorder=order)

        if FLAG_legend:
            ax = {1: ax1, 2: ax2, 3: ax3}
            ax[FLAG_legend].legend()

        plt.tight_layout()

        if direction == 1:
            return ax1, ax2, ax3
        else:
            return ax1, ax2, ax3, ax4
