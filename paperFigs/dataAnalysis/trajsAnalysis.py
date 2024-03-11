"""
    Description: Case 2 Results Analysis ...
    Author: Tenplus
    Create-time: 2023-07-03
    Update-time: 2023-07-06, # V1.2, 轨迹性能分析
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
from objects.Road import Road
from objects.Vehicle import Vehicle
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from utilities.plotResults import show_trajs_threeLanes

current_file = os.path.abspath(__file__)
current_folder = os.path.dirname(current_file)

rampTraj_des = np.load(current_folder + '\\rampDesired.npy')


class Trajectory:
    range_Dis = {'R0': [-300, 250], 'M1': [-500, 250], 'M2': [-600, 250]}

    def __init__(self, veh_name, veh_traj):
        self.name = veh_name
        self.traj = traj = veh_traj
        self.route = veh_name[:2]

        """ deal with vehicle longitudinal trajectory """
        self.time_c = traj[:, 0].astype(float)
        self.long_p = traj[:, 1].astype(float)
        self.long_v = traj[:, 2].astype(float)

        self.v_aver = self.calculate_averageV()
        self.delay, self.k_range = self.calculate_delay()

    def calculate_delay(self):
        delay = []
        k_range = []

        if self.route[0] == 'M':
            v_des = 25 if self.route == 'M2' else 20
            p_left = Road.M2_SCH_RANGE[0] if self.route == 'M2' else Road.M1_SCH_RANGE[0]
            p_right = 200

            k_range = np.where((self.long_p >= p_left) & (self.long_p <= p_right))[0]
            k_st = k_range[0]
            t0, p0 = self.time_c[k_st], self.long_p[k_st]

            ''' calculate delay '''
            for k_ in k_range:
                t_, p_ = self.time_c[k_], self.long_p[k_]
                t_des = (p_ - p0) / v_des + t0
                t_delay = t_ - t_des
                if t_delay < 1e-4:
                    t_delay = 0

                delay.append([t_, t_delay])

        elif self.route == 'R0':
            p_left = -150
            p_right = 200
            k_range = np.where((self.long_p >= p_left) & (self.long_p <= p_right))[0]
            k_st = k_range[0]
            t0, p0 = self.time_c[k_st], self.long_p[k_st]

            t_diff = t0 - rampTraj_des[0, 0]
            p_diff = p0 - rampTraj_des[0, 1]

            t_ref = rampTraj_des[:, 0] + t_diff
            p_ref = rampTraj_des[:, 1] + p_diff

            for k_ in k_range:
                t_ = self.time_c[k_]
                t_des = np.interp(self.long_p[k_], p_ref, t_ref)
                t_delay = t_ - t_des
                if t_delay < 1e-4:
                    t_delay = 0

                delay.append([t_, t_delay])

        return np.array(delay), k_range

    def plot_delay(self):
        fig = plt.figure()
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)
        ax1.plot(self.delay[:, 0], self.delay[:, 1], 'r')
        ax2.plot(self.time_c[self.k_range], self.long_v[self.k_range], 'k')

    def calculate_averageV(self):
        range_dis = self.range_Dis[self.name[:2]]
        k_range = np.where((self.long_p >= range_dis[0]) & (self.long_p <= range_dis[1]))[0]
        t0, p0 = self.time_c[k_range[0]], self.long_p[k_range[0]]
        tf, pf = self.time_c[k_range[-1]], self.long_p[k_range[-1]]
        v_aver = (pf - p0) / (tf - t0)

        return v_aver


class TrajsAnalysis:
    v_max = 25
    color_map = plt.cm.get_cmap(name='brg')
    color_range = [0.5, 1]  # corresponding to v: [0, v_max]
    alpha = {'R0': 1.0, 'M1': 1.2, 'M2': 1.3}

    def __init__(self, dir):
        self.dir = dir
        self.vehs_R0, self.vehs_M1, self.vehs_M2 = {}, {}, {}

        self.vehs_All, self.t_end_list = self.loadData()

        # self.totalDelay_list = self.calculate_Delays()
        # self.averV_R0, self.averV_M1, self.averV_M2 = self.calculate_AverageV()

    def loadData(self):
        vehsID_R0, vehsID_M1, vehsID_M2 = [], [], []
        t_end_list = {}

        for file_name in os.listdir(self.dir):
            if file_name.endswith('.npy'):
                veh_name = file_name[:-4]
                veh_traj = np.load(os.path.join(self.dir, file_name))

                veh_route = veh_name[:2]
                vehID = int(veh_name.split('_')[1])
                t_end_list[veh_name] = float(veh_traj[-1, 0])

                if veh_route == 'R0':
                    self.vehs_R0[veh_name] = Trajectory(veh_name, veh_traj)
                    vehsID_R0.append(vehID)
                elif veh_route == 'M1':
                    self.vehs_M1[veh_name] = Trajectory(veh_name, veh_traj)
                    vehsID_M1.append(vehID)
                elif veh_route == 'M2':
                    self.vehs_M2[veh_name] = Trajectory(veh_name, veh_traj)
                    vehsID_M2.append(vehID)

        """ Difference set: check the completeness """
        vehsID_diff_R0 = list(set(np.arange(max(vehsID_R0) + 1)) - set(vehsID_R0))
        vehsID_diff_M1 = list(set(np.arange(max(vehsID_M1) + 1)) - set(vehsID_M1))
        vehsID_diff_M2 = list(set(np.arange(max(vehsID_M2) + 1)) - set(vehsID_M2))

        print(f'**** vehs ID differences **** \n'
              f'R0: {vehsID_diff_R0} \n'
              f'M1: {vehsID_diff_M1} \n'
              f'M2: {vehsID_diff_M2} \n')

        vehs_All = dict(self.vehs_R0, **self.vehs_M1, **self.vehs_M2)
        return vehs_All, t_end_list

    def calculate_Delays(self, interval=10):
        t_end = max(self.t_end_list.values())
        totalDelay_list = []

        for step in np.arange(0, round(t_end * 10 + 1), interval):  # 10: control the interval
            totalDelay = 0
            for veh in self.vehs_All.values():
                alpha = self.alpha[veh.name[:2]]
                delay = np.interp(step / 10, veh.delay[:, 0], veh.delay[:, 1])
                totalDelay += delay * alpha
            totalDelay_list.append([step / 10, totalDelay])

        return np.array(totalDelay_list)

    def calculate_AverageV(self):
        averV_R0 = np.mean([veh.v_aver for veh in self.vehs_R0.values()])
        averV_M1 = np.mean([veh.v_aver for veh in self.vehs_M1.values()])
        averV_M2 = np.mean([veh.v_aver for veh in self.vehs_M2.values()])

        return averV_R0, averV_M1, averV_M2


def plot_longPos(vehs_set, ms=0.2, ax=None, xlim=[50, 360], ylim=[-350, 250], fig_size=(5, 2.4)):
    if not ax:
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111)

    for traj in vehs_set.values():
        norm_v = np.interp(traj.long_v, [0, TrajsAnalysis.v_max], TrajsAnalysis.color_range)
        colors = TrajsAnalysis.color_map(norm_v)
        ax.scatter(traj.time_c, traj.long_p, c=colors, s=ms)

    """ axes setting """
    sz = 10
    font_label = {'size': sz}
    ax.xaxis.set_tick_params(labelsize=sz)
    ax.yaxis.set_tick_params(labelsize=sz)

    ax.set_xlabel(r'$t~\mathrm{(s)}$', font_label)
    route = list(vehs_set.keys())[0][:2]
    ax.set_ylabel(r'$Route~%s: p~\mathrm{(m)}$' % route, font_label)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.xaxis.set_major_locator(plt.MultipleLocator(50))
    ax.yaxis.set_major_locator(plt.MultipleLocator(200))

    plt.tight_layout()

    return ax


def add_subAxes(ax, vehs_set, xyPos=[0.95, 0.05, 0.25, 0.3], ms=0.2, xlim=[260, 290], ylim=[-180, 40]):
    sub_ax = ax.inset_axes(xyPos)
    sub_ax.set_xticks([])
    sub_ax.set_yticks([])

    for traj in vehs_set.values():
        norm_v = np.interp(traj.long_v, [0, TrajsAnalysis.v_max], TrajsAnalysis.color_range)
        colors = TrajsAnalysis.color_map(norm_v)
        sub_ax.scatter(traj.time_c, traj.long_p, c=colors, s=ms)

        sub_ax.set_xlim(xlim)
        sub_ax.set_ylim(ylim)

    mark_inset(ax, sub_ax, loc1=1, loc2=3, fc="none", ec='k', ls='--', lw=0.8)
    # ax.indicate_inset_zoom(sub_ax)
    plt.tight_layout()


def show_delaysBoth(trajs_DCG, trajs_SUMO, interval=15):
    delays_DCG = trajs_DCG.calculate_Delays()
    delays_SUMO = trajs_SUMO.calculate_Delays()
    print(f'**** FINAL DELAYS **** \n'
          f'*** DCG: {delays_DCG[-1]} \n'
          f'*** SUMO: {delays_SUMO[-1]} \n')

    c_dcg, c_sumo = 'blue', 'purple'
    """ create axes & settings """
    fig = plt.figure(figsize=(5.5, 2.4))
    ax = fig.add_subplot()

    lw = 2
    """ plot lines """
    l1 = ax.plot(delays_DCG[:, 0], delays_DCG[:, 1], color=c_dcg, label='DCG-based', lw=lw, zorder=10)
    k_dcg = np.arange(0, np.shape(delays_DCG)[0] + 1, interval)
    ax.scatter(delays_DCG[k_dcg, 0], delays_DCG[k_dcg, 1], marker='o', s=18, color=c_dcg, zorder=10)

    l2 = ax.plot(delays_SUMO[:, 0], delays_SUMO[:, 1], color=c_sumo, label='SUMO default', lw=lw)
    k_sumo = np.arange(0, np.shape(delays_SUMO)[0] + 1, interval)
    ax.scatter(delays_SUMO[k_sumo, 0], delays_SUMO[k_sumo, 1], marker='s', s=18, color=c_sumo)

    ax.legend(l1 + l2, [l.get_label() for l in l1 + l2], fontsize=9)

    lw_sub = 1.8
    """ add sub_axes """
    sub_ax = ax.inset_axes([0.85, 0.3, 0.4, 0.4])
    sub_ax.plot(delays_DCG[:, 0], delays_DCG[:, 1], color=c_dcg, label='DCG-based', lw=lw_sub, zorder=10)
    k_dcg = np.arange(0, np.shape(delays_DCG)[0] + 1, interval * 2)
    sub_ax.scatter(delays_DCG[k_dcg, 0], delays_DCG[k_dcg, 1], marker='o', s=12, color=c_dcg, zorder=10)

    # sub_ax.plot(delays_SUMO[:, 0], delays_SUMO[:, 1], color=c_sumo, label='SUMO default', lw=lw_sub)
    # k_sumo = np.arange(0, np.shape(delays_SUMO)[0] + 1, interval * 2)
    # sub_ax.scatter(delays_SUMO[k_sumo, 0], delays_SUMO[k_sumo, 1], marker='s', s=12, color=c_sumo)

    sub_ax.yaxis.tick_right()

    sz = 10
    font_label = {'size': sz}
    ax.xaxis.set_tick_params(labelsize=sz)
    ax.yaxis.set_tick_params(labelsize=sz)
    sub_ax.xaxis.set_tick_params(labelsize=sz - 1)
    sub_ax.yaxis.set_tick_params(labelsize=sz - 1)

    ax.set_xlabel(r'$t~\mathrm{(s)}$', font_label)
    ax.set_ylabel(r'Cost$:~J~\mathrm{(s)}$', font_label)

    ax.set_xlim([-20, 430])
    sub_ax.set_xlim([-20, 420])
    ax.set_ylim([-700, 12500])
    sub_ax.set_ylim([-50, 500])

    ax.yaxis.set_major_locator(plt.MultipleLocator(4000))
    sub_ax.yaxis.set_major_locator(plt.MultipleLocator(250))

    # mark_inset(ax, sub_ax, loc1=2, loc2=4, fc="none", ec='gray', ls='--', alpha=0.5, lw=0.8)

    plt.tight_layout()


def show_delaysSubAxes(trajs_DCG, trajs_SUMO, interval=15):
    delays_DCG = trajs_DCG.calculate_Delays()
    delays_SUMO = trajs_SUMO.calculate_Delays()
    print(f'**** FINAL DELAYS **** \n'
          f'*** DCG: {delays_DCG[-1]} \n'
          f'*** SUMO: {delays_SUMO[-1]} \n')

    c_dcg, c_sumo = 'blue', 'purple'
    """ create axes & settings """
    fig = plt.figure(figsize=(5, 4))
    ax_dcg = fig.add_subplot(211)
    ax_sumo = fig.add_subplot(212)

    lw = 2
    """ plot lines """
    ax_dcg.plot(delays_DCG[:, 0], delays_DCG[:, 1], color=c_dcg, label='DCG-based', lw=lw, zorder=10)
    k_dcg = np.arange(0, np.shape(delays_DCG)[0] + 1, interval)
    ax_dcg.scatter(delays_DCG[k_dcg, 0], delays_DCG[k_dcg, 1], marker='o', s=18, color=c_dcg, zorder=10)
    ax_dcg.legend(fontsize=9)

    ax_sumo.plot(delays_SUMO[:, 0], delays_SUMO[:, 1], color=c_sumo, label='SUMO default', lw=lw)
    k_sumo = np.arange(0, np.shape(delays_SUMO)[0] + 1, interval)
    ax_sumo.scatter(delays_SUMO[k_sumo, 0], delays_SUMO[k_sumo, 1], marker='s', s=18, color=c_sumo)
    ax_sumo.legend(fontsize=9)

    """ axes setting """
    sz = 10
    font_label = {'size': sz}

    for ax_ in [ax_dcg, ax_sumo]:
        ax_.xaxis.set_tick_params(labelsize=sz)
        ax_.yaxis.set_tick_params(labelsize=sz)
        ax_.set_xlabel(r'$t~\mathrm{(s)}$', font_label)
        ax_.set_ylabel(r'Cost$:~J~\mathrm{(s)}$', font_label)
        ax_.set_xlim([-20, 430])
        ax_.xaxis.set_major_locator(plt.MultipleLocator(100))

    ax_dcg.set_ylim([-50, 520])
    ax_dcg.yaxis.set_major_locator(plt.MultipleLocator(250))
    ax_sumo.set_ylim([-700, 12500])
    ax_sumo.yaxis.set_major_locator(plt.MultipleLocator(4000))


    plt.tight_layout()


if __name__ == "__main__":
    """ DCG """
    file_name = '20240116_1617_58' + '/trajs'
    dir = '../../../04 Data saving/movingProcess/' + file_name
    trajsAna_DCG = TrajsAnalysis(dir)

    show_trajs_threeLanes(trajsAna_DCG.vehs_All)
    # ax_R0 = plot_longPos(trajsAna_DCG.vehs_R0)
    # ax_R0 = plot_longPos(trajsAna_DCG.vehs_R0, fig_size=(6, 2.4))
    # add_subAxes(ax_R0, trajsAna_DCG.vehs_R0)
    # plt.savefig('Trajs_DCG_R0.svg')

    # ax_M2 = plot_longPos(trajsAna_DCG.vehs_M2, ylim=[-600, 250])
    # # add_subAxes(ax_M2, trajsAna_DCG.vehs_M2, xlim=[255, 280], ylim=[-450, -120])
    # plt.savefig('Trajs_DCG_M2.svg')

    """ SUMO """
    # file_name = '20230706_1649_56_SUMO-M1 0.52有拥堵'
    # dir = '../../../04 Data saving/movingProcess/' + file_name
    # trajsAna_SUMO = TrajsAnalysis(dir)

    # ax_R0 = plot_longPos(trajsAna_SUMO.vehs_R0)
    # ax_R0 = plot_longPos(trajsAna_SUMO.vehs_R0, fig_size=(6, 2.4))
    # add_subAxes(ax_R0, trajsAna_SUMO.vehs_R0, xlim=[245, 275], ylim=[-83, 173])
    # plt.savefig('Trajs_SUMO_R0.svg')

    # ax_M2 = plot_longPos(trajsAna_SUMO.vehs_M2, ylim=[-600, 250])
    # # add_subAxes(ax_M2, trajsAna_SUMO.vehs_M2, xlim=[285, 315], ylim=[-450, -110])
    # plt.savefig('Trajs_SUMO_M2.svg')

    """ calculate the delay """
    # show_delaysSubAxes(trajsAna_DCG, trajsAna_SUMO)
    # plt.savefig('DelayCompare.svg')

    plt.show()
