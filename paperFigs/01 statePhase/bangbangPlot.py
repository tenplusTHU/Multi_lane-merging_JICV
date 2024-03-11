"""
    Bang-Bang Plot, exercise.
    Author: Tenplus
    Create-time: 2022-03-24
    Update-time: 2022-06-14    V2.0 # 修改类后重画图
"""

import numpy as np
import matplotlib.pyplot as plt
from objects.Road import Road
from cloudController.bangbangController import BBC

""" Global Parameters  """
j_max = BBC.j_max
j_min = BBC.j_min


class StatesTrajs_clusters:
    """ theoretical analysis """

    def __init__(self):
        """ plot setting """
        fig = plt.figure(figsize=(4, 3))
        self.ax = ax = fig.add_subplot()
        plt.tick_params(labelsize=11)
        font_label = {'size': 12}
        ax.set_xlabel(r'$a_j^{\ast}$', font_label)
        ax.set_ylabel(r'$v_j^{\ast}$', font_label)

        self.rho = [j_max, j_min, 0]
        self.a_range = a_range = [-4, 4]
        self.v_range = v_range = [0, 35]
        ax.set_xlim(a_range)
        ax.set_xticks([a_range[0], 0, a_range[1]])
        ax.set_xticklabels([r'$a_\mathrm{min}$', 0, r'$a_\mathrm{max}$'], fontsize=12)

        ax.set_ylim(v_range)
        ax.set_yticks([0, 16, 32])
        ax.set_yticklabels([0, r'$v_0}$', r'$v_{M*}}$'], fontsize=12, rotation=90)

        plt.tight_layout()

        """ plot lines """
        self.plot_statesTrajs()
        self.plot_min2max2min(v0=16, v_min=8, color='k')
        self.plot_max2zero2min(a_max=1, color='b')
        self.plot_max2min()
        # self.plot_min2max2min(v0=32, v_min=24, color='b')
        # self.plot_min2max2min(v0=32, v_min=24, color='k')

        plt.legend(loc=2, fontsize=11)
        # plt.savefig('../../05 论文写作/02 fig/' + 'state_trajectory.svg', dpi=600)

    def plot_statesTrajs(self):
        """ plot trajs clusters """
        lw_back = 1
        for v0 in np.arange(0, self.v_range[1] + 1, 8):
            a = np.linspace(self.a_range[0], self.a_range[1], 100, endpoint=True)

            ''' j_max '''
            v_pos = a ** 2 / 2 / self.rho[0] + v0
            label = r'$\rho=j_\mathrm{max}$' if v0 == 0 else None
            self.ax.plot(a, v_pos, 'b', ls='-.', lw=lw_back, zorder=2, label=label)

            arrow_a = 1.3
            arrow_v = arrow_a ** 2 / 2 / self.rho[0] + v0
            arrow_k = arrow_a / self.rho[0]
            self.ax.annotate('', xy=(arrow_a + 0.1, arrow_v + 0.1 * arrow_k), xytext=(arrow_a, arrow_v),
                             arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3", fc='b', ec='b'))

            ''' j_min '''
            v_neg = a ** 2 / 2 / self.rho[1] + v0
            label = r'$\rho=j_\mathrm{min}$' if v0 == 0 else None
            self.ax.plot(a, v_neg, 'k', ls='--', lw=lw_back, zorder=1, label=label)

            arrow_a = - 1.3
            arrow_v = arrow_a ** 2 / 2 / self.rho[1] + v0
            arrow_k = arrow_a / self.rho[1]
            self.ax.annotate('', xy=(arrow_a - 0.1, arrow_v - 0.1 * arrow_k), xytext=(arrow_a, arrow_v),
                             arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3", fc='k', ec='k'))

        ''' j=0 '''
        a_ver = [a - 3 for a in range(0, 7) if a != 3]
        self.ax.vlines(a_ver, self.v_range[0], self.v_range[1], 'gray', ls=':', lw=lw_back, zorder=0, label=r'$\rho=0$')

        for a_ in a_ver:
            if a_ > 0:
                arrow_v = 15
                label = r'$\rho=0$' if a_ == 1 else None
                self.ax.annotate('', xytext=(a_, arrow_v), xy=(a_, arrow_v + 0.1),
                                 arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3", fc='gray', ec='gray'),
                                 label=label)
            else:
                arrow_v = 15.8
                self.ax.annotate('', xytext=(a_, arrow_v), xy=(a_, arrow_v - 0.1),
                                 arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3", fc='gray', ec='gray'))

    def plot_max2min(self, v0=16, vf=32, color='r'):
        self.ax.scatter(0, v0, marker='s', s=40, c=color, zorder=9)
        self.ax.scatter(0, vf, marker='*', s=100, c=color, zorder=9)
        lw = 2

        a_bound = np.sqrt((vf - v0) / (1 / 2 / j_max - 1 / 2 / j_min))
        a_range = np.linspace(0, a_bound, 50)
        v_up = a_range ** 2 / 2 / j_max + v0
        v_down = a_range ** 2 / 2 / j_min + vf
        self.ax.plot(a_range, v_up, c=color, lw=lw, zorder=9)
        self.ax.plot(a_range, v_down, c=color, lw=lw, zorder=9)

        ''' add arrows '''
        a = 1.3
        v_lo = a ** 2 / 2 / j_max + v0
        k_lo = a / j_max
        self.ax.annotate('', xy=(a + 0.1, v_lo + 0.1 * k_lo), xytext=(a, v_lo),
                         arrowprops=dict(headwidth=6, headlength=6, fc=color, ec=color))

        v_up = a ** 2 / 2 / j_min + vf
        k_up = a / j_min
        self.ax.annotate('', xy=(a - 0.1, v_up - 0.1 * k_up), xytext=(a, v_up),
                         arrowprops=dict(headwidth=6, headlength=6, fc=color, ec=color))

    def plot_max2zero2min(self, v0=16, vf=32, a_max=2, color='r'):
        self.ax.scatter(0, v0, marker='s', s=40, c=color, zorder=9)
        self.ax.scatter(0, vf, marker='*', s=80, c=color, zorder=9)
        lw = 2

        a_range = np.linspace(0, a_max, 50)
        v_up = a_range ** 2 / 2 / j_max + v0
        v_down = a_range ** 2 / 2 / j_min + vf
        self.ax.plot(a_range, v_up, c=color, lw=lw, zorder=9)
        self.ax.plot(a_range, v_down, c=color, lw=lw, zorder=9)
        self.ax.plot([a_max] * 2, [v_up[-1], v_down[-1]], c=color, lw=lw, zorder=9)

        ''' add arrows '''
        arrow_v = (v0 + vf) / 2
        self.ax.annotate('', xytext=(a_max, arrow_v), xy=(a_max, arrow_v + 0.1),
                         arrowprops=dict(headwidth=6, headlength=6, fc=color, ec=color))

    def plot_min2max2min(self, v0=16, vf=32, v_min=8, color='r'):
        self.ax.scatter(0, v0, marker='s', s=40, c=color, zorder=9)
        self.ax.scatter(0, vf, marker='*', s=80, c=color, zorder=9)
        lw = 2

        a_left = -np.sqrt((v0 - v_min) / (1 / 2 / j_max - 1 / 2 / j_min))
        a_right = np.sqrt((vf - v_min) / (1 / 2 / j_max - 1 / 2 / j_min))

        a_range = np.linspace(a_left, a_right, 50)
        v_up = a_range ** 2 / 2 / j_max + v_min
        self.ax.plot(a_range, v_up, c=color, lw=lw, zorder=5)

        a_range = np.linspace(a_left, 0, 50)
        v_up = a_range ** 2 / 2 / j_min + v0
        self.ax.plot(a_range, v_up, c=color, lw=lw, zorder=5)

        a_range = np.linspace(0, a_right, 50)
        v_up = a_range ** 2 / 2 / j_min + vf
        self.ax.plot(a_range, v_up, c=color, lw=lw, zorder=5)

        ''' add arrows '''
        a = 1.3
        v_lo = a ** 2 / 2 / j_max + v_min
        k_lo = a / j_max
        self.ax.annotate('', xy=(a + 0.1, v_lo + 0.1 * k_lo), xytext=(a, v_lo),
                         arrowprops=dict(headwidth=6, headlength=6, fc=color, ec=color))

        a = -1.3
        v_up = a ** 2 / 2 / j_min + v0
        k_up = a / j_min
        self.ax.annotate('', xy=(a - 0.1, v_up - 0.1 * k_up), xytext=(a, v_up),
                         arrowprops=dict(headwidth=6, headlength=6, fc=color, ec=color))


def plot_setting():
    fig = plt.figure(figsize=(10, 2.3))
    font_label = {'size': 11}
    ax1 = fig.add_subplot(141)
    ax2 = fig.add_subplot(142)
    ax3 = fig.add_subplot(143)
    ax4 = fig.add_subplot(144)

    for ax_ in [ax1, ax2, ax3]:
        ax_.set_xlabel(r'$t~\mathrm{(s)}$', font_label)

    ax1.set_ylabel(r'$p~\mathrm{(m)}$', font_label)
    ax2.set_ylabel(r'$v~\mathrm{(m/s)}$', font_label)
    ax3.set_ylabel(r'$a~\mathrm{(m/s^2)}$', font_label)

    ax4.set_xlabel(r'$a~\mathrm{(m/s^2)}$', font_label)
    ax4.set_ylabel(r'$v~\mathrm{(m/s)}$', font_label)

    ax2.yaxis.set_major_locator(plt.MultipleLocator(5))
    ax3.yaxis.set_major_locator(plt.MultipleLocator(1))
    ax4.yaxis.set_major_locator(plt.MultipleLocator(5))

    plt.tight_layout()
    return ax1, ax2, ax3, ax4


def show_stage3_max2zero2min(bbc: BBC):
    ax1, ax2, ax3, ax4 = plot_setting()

    """ only the pole """
    t1_pole, t2_pole, pf_pole = bbc.stage2_max2min.get_element()
    traj = bbc.stage2_max2min.get_traj()
    t, a, v, p = zip(*traj)
    color, lw, order = 'r', 2, 10
    t_lim = [-0.6, 18.6]
    ax1.set_xlim(t_lim)
    ax2.set_xlim(t_lim)
    ax3.set_xlim(t_lim)

    p_lim = [-70, 220]
    ax1.set_ylim(p_lim)

    ax1.plot(t, p, c=color, lw=lw, zorder=order)
    ax1.plot(t[-1], p[-1], 'o', c=color, zorder=order)
    ax2.plot(t, v, c=color, lw=lw, zorder=order)
    ax3.plot(t, a, c=color, lw=lw, zorder=order)
    ax4.plot(a, v, c=color, lw=lw, zorder=order)

    pf_list = np.linspace(Road.MERGE_ZONE_LENGTH, pf_pole, 4, endpoint=True)
    alpha_list = np.linspace(1, 0.4, len(pf_list) - 1, endpoint=True)

    for pf in pf_list[:-1]:
        alpha = alpha_list[list(pf_list).index(pf)]
        color, order = 'b', 2
        traj = bbc.stage3_max2zero2min.get_traj(pf)
        t, a, v, p = zip(*traj)

        ax1.plot(t, p, c=color, alpha=alpha, lw=lw, zorder=order)
        ax1.plot(t[-1], p[-1], 'o', c=color)
        ax2.plot(t, v, c=color, alpha=alpha, lw=lw, zorder=order)
        ax3.plot(t, a, c=color, alpha=alpha, lw=lw, zorder=order)
        ax4.plot(a, v, c=color, alpha=alpha, lw=lw, zorder=order)

    # plt.savefig('../../03 Data saving/bangbangController/' + 'bbc_optTF_2.svg', dpi=600)
    plt.show()


def show_stage3_min2max2min(bbc: BBC):
    ax1, ax2, ax3, ax4 = plot_setting()
    ax3.yaxis.set_major_locator(plt.MultipleLocator(2))

    """ only the pole """
    traj = bbc.stage2_max2min.get_traj()
    t, a, v, p = zip(*traj)
    color, lw, order = 'r', 2, 9

    ax1.plot(t, p, c=color, lw=lw, zorder=order)
    ax1.plot(t[-1], p[-1], 'o', c=color)
    ax2.plot(t, v, c=color, lw=lw, zorder=order)
    ax3.plot(t, a, c=color, lw=lw, zorder=0)
    ax4.plot(a, v, c=color, lw=lw, zorder=order)

    # v_min_list = np.linspace(0, bbc.v_min, 4, endpoint=True)
    v_min_list = [0, 3, 5.5, bbc.v_min]
    alpha_list = np.linspace(1, 0.4, len(v_min_list) - 1, endpoint=True)

    for v_min in v_min_list[:-1]:
        alpha = alpha_list[list(v_min_list).index(v_min)]
        color, order = 'k', 2
        traj = bbc.stage3_min2max2min.get_traj(['v_min', v_min])
        t, a, v, p = zip(*traj)

        ax1.plot(t, p, c=color, alpha=alpha, lw=lw, zorder=order)
        ax1.plot(t[-1], p[-1], 'o', c=color)
        ax2.plot(t, v, c=color, alpha=alpha, lw=lw, zorder=order)
        ax3.plot(t, a, c=color, alpha=alpha, lw=lw, zorder=order)
        ax4.plot(a, v, c=color, alpha=alpha, lw=lw, zorder=order)

    # plt.savefig('../../03 Data saving/bangbangController/' + 'bbc_optPF_0.svg', dpi=600)
    plt.show()


def show_stage5(bbc: BBC):
    ax1, ax2, ax3, ax4 = plot_setting()
    ax3.yaxis.set_major_locator(plt.MultipleLocator(1))

    t1, t2, t3, pf = bbc.stage3_min2max2min.get_element(['v_min', bbc.v_min])
    tf_begin = bbc.t0 + t1 + t2 + t3
    tf_end = (Road.MERGE_ZONE_LENGTH - pf) / bbc.v_min + tf_begin

    tf_list = np.linspace(tf_begin, tf_end, 4, endpoint=True)
    alpha_list = np.linspace(1, 0.4, len(tf_list), endpoint=True)

    lw = 2

    for tf in tf_list:
        alpha = alpha_list[list(tf_list).index(tf)]
        color, order = 'k', 2
        traj = bbc.stage5_min2max2zero2max2min.get_traj(tf)
        t, a, v, p = zip(*traj)

        ax1.plot(t, p, c=color, alpha=alpha, lw=lw, zorder=order)
        ax1.plot(t[-1], p[-1], 'o', c=color)
        ax2.plot(t, v, c=color, alpha=alpha, lw=lw, zorder=order)
        ax3.plot(t, a, c=color, alpha=alpha, lw=lw, zorder=order)
        ax4.plot(a, v, c=color, alpha=alpha, lw=lw, zorder=order)

    # plt.savefig('../../03 Data saving/bangbangController/' + 'bbc_minV_1.svg', dpi=600)
    plt.show()


def show_trajCases(bbc: BBC):
    """ create figs and axes """
    fig = plt.figure(figsize=(5.5, 4.5))

    font_label = {'size': 12}
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    for ax_ in [ax1, ax2, ax3, ax4]:
        size = 11
        ax_.xaxis.set_tick_params(labelsize=size)
        ax_.yaxis.set_tick_params(labelsize=size)

    for ax_ in [ax1, ax2, ax3]:
        ax_.set_xlabel(r'$t~\mathrm{(s)}$', font_label)

    ax1.set_ylabel(r'$p_k~\mathrm{(m)}$', font_label)
    ax2.set_ylabel(r'$v_k~\mathrm{(m/s)}$', font_label)
    ax3.set_ylabel(r'$a_k~\mathrm{(m/s^2)}$', font_label)

    ax4.set_xlabel(r'$a_k~\mathrm{(m/s^2)}$', font_label)
    ax4.set_ylabel(r'$v_k~\mathrm{(m/s)}$', font_label)

    markersize = 5
    """ case a """
    # t1_pole, t2_pole, pf_pole = bbc.stage2_max2min.get_element()
    traj = bbc.stage2_max2min.get_traj()
    t, a, v, p = zip(*traj)
    color, lw, order = 'r', 2, 10
    ax1.plot(t, p, c=color, lw=lw, zorder=order)
    ax1.plot(t[-1], p[-1], 'o', c=color, zorder=order, markersize=markersize)
    ax2.plot(t, v, c=color, lw=lw, zorder=order)
    ax3.plot(t, a, c=color, lw=lw, zorder=order)
    ax4.plot(a, v, c=color, lw=lw, zorder=order)

    ax4.plot(a[0], v[0], 's', c=color, zorder=order, markersize=5)
    ax4.plot(a[-1], v[-1], 'o', c=color, zorder=order, markersize=markersize)

    """ case b """
    v_min_list = [9, 7, 5]
    # alpha_list = np.linspace(1, 0.4, len(v_min_list) - 1, endpoint=True)
    alpha_list = np.linspace(1, 0.6, len(v_min_list), endpoint=True)

    for v_min in v_min_list:
        alpha = alpha_list[list(v_min_list).index(v_min)]
        color, order = 'b', 8
        traj = bbc.stage3_min2max2min.get_traj(['v_min', v_min])
        t, a, v, p = zip(*traj)

        ax1.plot(t, p, c=color, alpha=alpha, lw=lw, zorder=order)
        ax1.plot(t[-1], p[-1], 'o', c=color, markersize=markersize, zorder=order)
        ax2.plot(t, v, c=color, alpha=alpha, lw=lw, zorder=order)
        ax3.plot(t, a, c=color, alpha=alpha, lw=lw, zorder=order)
        ax4.plot(a, v, c=color, alpha=alpha, lw=lw, zorder=order)

    """ case c """
    t1, t2, t3, pf = bbc.stage3_min2max2min.get_element(['v_min', bbc.v_min])
    tf_begin = bbc.t0 + t1 + t2 + t3
    tf_end = (Road.MERGE_ZONE_LENGTH - pf) / bbc.v_min + tf_begin

    tf_list = np.linspace(tf_begin, tf_end, 4, endpoint=True)
    alpha_list = np.linspace(0.5, 1, len(tf_list), endpoint=True)

    lw = 2

    for tf in tf_list[1:]:
        alpha = alpha_list[list(tf_list).index(tf)]
        color, order = 'k', 2
        traj = bbc.stage5_min2max2zero2max2min.get_traj(tf)
        t, a, v, p = zip(*traj)

        ax1.plot(t, p, c=color, alpha=alpha, lw=lw, zorder=order)
        ax1.plot(t[-1], p[-1], 'o', c=color, markersize=markersize)
        ax2.plot(t, v, c=color, alpha=alpha, lw=lw, zorder=order)
        ax3.plot(t, a, c=color, alpha=alpha, lw=lw, zorder=order)
        ax4.plot(a, v, c=color, alpha=alpha, lw=lw, zorder=order)

    plt.tight_layout()

    plt.savefig('bbp.svg')


if __name__ == '__main__':
    """ create a bbc """
    t0, p0, v0, a0 = 0, -60, 10, 0.2
    v_des, v_min = 20, 5
    bbc = BBC(t0, p0, v0, a0, v_des, v_min)

    show_trajCases(bbc)

    # show_stage3_max2zero2min(bbc)
    # show_stage3_min2max2min(bbc)
    # show_stage5(bbc)

    plt.show()
