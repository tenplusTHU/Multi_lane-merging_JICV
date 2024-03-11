"""
    Description: Longitudinal StatePhase for both main and ramp vehicle, tidy version
    Author: Tenplus
    Create-time: 2022-02-21
"""

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from objects.Vehicle import Vehicle
from objects.Road import Road
from utilities.plotResults import show_severalTrajs
from cloudController.bangbangController import BBC
from routes.route_static import Route_staticInput


class Long_StatePhase:
    j_min = BBC.j_min
    j_max = BBC.j_max
    pf_road = Road.MERGE_ZONE_LENGTH
    RAMP_LIMITED = Road.RAMP_LIMITED  # constant speed for ramp vehicles in limited area

    def __init__(self, t0=0.0, p0=-200.0, v0=Vehicle.v_R0, a0=0.0,
                 route='R0', v_des=Vehicle.v_des['M1'], v_min=Vehicle.v_min['R0']):
        """ seven input variables """
        self.t0 = t0
        self.p0 = p0
        self.v0 = v0
        self.a0 = a0
        self.route = route
        self.v_des = v_des
        self.v_min = v_min

        """ calibrate the start position for ramp vehicles """
        if route[0] == 'R' and p0 < self.RAMP_LIMITED:
            self.p_st = self.RAMP_LIMITED
            self.t_st = (self.p_st - p0) / v0 + t0
        else:
            self.p_st, self.t_st = p0, t0

        """ calculate the key points & lines in the vehicle statePhase """
        self.bbc = BBC(self.t_st, self.p_st, v0, a0, self.v_des, self.v_min)
        self.pole = self.__get_pole()
        self.adjustD_tf, self.adjustD_pf, self.adjustD_whole_FLAG = self.__get_opt_tf()
        self.boundP_tf, self.boundP_pf, self.boundP_cutOff_FLAG = self.__get_opt_pf()
        self.tf_Range = self.__get_tfRange()

        """ intersection and planned_trajectory """
        self.intersection = None
        self.traj_inter, self.traj_planned = None, None

    def __get_pole(self):
        if self.bbc.stage0_FLAG:
            return [self.bbc.t0, self.bbc.p0]
        if self.bbc.stage1_FLAG:
            return [self.bbc.t0 + self.bbc.stage1_t_one, self.bbc.stage1_pf_one]
        else:
            t1, t2, pf = self.bbc.stage2_max2min.get_element()
            return [self.t_st + t1 + t2, pf]

    def __get_opt_tf(self):
        if self.a0 > 0 and not self.bbc.stage0_FLAG and not self.bbc.stage1_FLAG:  # debug, 2023-02-14
            _, _, pfMax = self.bbc.stage2_zero2min.get_element()
        else:
            pfMax = self.pf_road

        adjustD_pf = np.linspace(self.pole[1], min(pfMax, self.pf_road), 3, endpoint=True)
        adjustD_tf = []

        for pf in adjustD_pf:
            if not self.bbc.stage0_FLAG and not self.bbc.stage1_FLAG:  # in most cases
                t1, t2, t3, _ = self.bbc.stage3_max2zero2min.get_element(pf)
                adjustD_tf.append(self.t_st + t1 + t2 + t3)
            else:  # debug 2022-11-04, coincide with the pole
                tf = (pf - self.pole[1]) / self.v_des + self.pole[0]
                adjustD_tf.append(tf)

        if adjustD_pf[-1] != self.pf_road:
            adjustD_whole_FLAG = False  # having a turning point
            pf_append = np.linspace(adjustD_pf[-1], self.pf_road, 3, endpoint=True)[1:]
            tf_append = (pf_append - adjustD_pf[-1]) / self.v_des + adjustD_tf[-1]
            adjustD_pf = np.append(adjustD_pf, pf_append)
            adjustD_tf = np.append(adjustD_tf, tf_append)
        else:
            adjustD_whole_FLAG = True  # only one straight line

        # debug
        # print(f'FLAG: {adjustD_whole_FLAG}, turn: {adjustD_tf[-3]}')
        # plt.figure()
        # plt.plot(adjustD_tf, adjustD_pf, 'b', lw=2)
        # if not adjustD_whole_FLAG:
        #     plt.scatter(adjustD_tf[-3], adjustD_pf[-3], 'o', color='b')
        # plt.show()
        return np.array(adjustD_tf), adjustD_pf, adjustD_whole_FLAG

    def __get_opt_pf(self):
        t1, t2, t3, pf = self.bbc.stage3_min2max2min.get_element(['v_min', self.v_min])
        tf_arc = self.t_st + t1 + t2 + t3  # the maximum tf for the three-stage deceleration profile

        num_points = int(tf_arc - self.pole[0]) * 2 + 2
        vMin_list = np.linspace(self.v0, self.v_min, num_points, endpoint=True)     # note: not v0 here
        boundP_tf, boundP_pf = [], []

        for v_min in vMin_list:
            t1, t2, t3, pf = self.bbc.stage3_min2max2min.get_element(['v_min', v_min])
            if pf:
                boundP_tf.append(self.t_st + t1 + t2 + t3)
                boundP_pf.append(pf)

        boundP_tf = np.array(boundP_tf)
        boundP_pf = np.array(boundP_pf)
        boundP_cutOff_FLAG = True if max(boundP_pf) > self.pf_road else False
        if boundP_cutOff_FLAG:
            id_pfMax = np.argmax(boundP_pf)
            id_left = np.where(boundP_pf[0: id_pfMax] <= self.pf_road)[0]
            boundP_tf = boundP_tf[id_left]
            boundP_pf = boundP_pf[id_left]

        if boundP_cutOff_FLAG:
            pf_append = np.linspace(boundP_pf[-1], self.pf_road, 3, endpoint=True)[1:]
            tf_append = (pf_append - boundP_pf[-1]) / self.v_des + boundP_tf[-1]
        else:
            if self.v_min > 0:  # straight line, slope: v_min
                pf_append = np.linspace(boundP_pf[-1], self.pf_road, 3, endpoint=True)[1:]
                tf_append = (pf_append - boundP_pf[-1]) / self.v_min + boundP_tf[-1]
            else:  # v_min = 0
                t_app = (self.pf_road - boundP_pf[-1]) / self.v_des
                pf_append = np.array([boundP_pf[-1]] * 2)
                tf_append = np.array([t_app / 2, t_app]) + boundP_tf[-1]

        boundP_tf = np.append(boundP_tf, tf_append)
        boundP_pf = np.append(boundP_pf, pf_append)

        # plt.figure()
        # plt.plot(boundP_tf, boundP_pf, 'k', lw=2)
        # plt.show()

        return boundP_tf, boundP_pf, boundP_cutOff_FLAG

    def __get_tfRange(self):
        t_left = (self.pf_road - self.pole[1]) / self.v_des + self.pole[0]
        t_right = self.boundP_tf[-1] if self.boundP_pf[-1] == self.pf_road else np.inf
        return [t_left, t_right]

    def plot_phase(self, ax=None, c=None, input_tf=None, input_pf=None, overlap_FLAG='sweet'):
        if ax is None:
            fig = plt.figure(figsize=(4, 2.7))
            ax = fig.add_subplot()
            plt.tick_params(labelsize=10)
            font_label = {'size': 11}
            ax.set_xlabel(r'final time $t_f~\mathrm{(s)}$', font_label)
            ax.set_ylabel(f'final position $p_f$ (m)', font_label)
            ax.patch.set_fc('gray')
            ax.patch.set_alpha(0.1)

        ax.yaxis.set_major_locator(plt.MultipleLocator(40))
        # ax_road.xaxis.set_major_locator(plt.MultipleLocator(5))

        ''' temp '''
        ax.set_xlim([8, 31])
        ax.set_ylim([60, Road.MERGE_ZONE_LENGTH])

        lw = 2
        alpha=.2
        ''' draw key points and lines '''
        ax.plot(self.pole[0], self.pole[1], 'o', c='r' if not c else c, zorder=9,
                label=r'$j_\mathrm{max}~\to~j_\mathrm{min}$')
        ax.plot([self.pole[0], self.tf_Range[0]], [self.pole[1], self.pf_road],
                c='r' if not c else c, linestyle='-', lw=lw, zorder=8)
        # ax.plot(self.adjustD_tf, self.adjustD_pf, c='b' if not c else c, ls='--', lw=1, zorder=7,
        #         label=r'$j_\mathrm{max}~\to~0~\to~j_\mathrm{min}$')  # label='optimal-time'
        ax.plot(self.boundP_tf, self.boundP_pf, c='k' if not c else c, lw=lw, zorder=7,
                label=r'$j_\mathrm{min}~\to~j_\mathrm{max}~\to~j_\mathrm{min}$')  # label='shortest-distance'

        ''' fill the available zones '''
        ax.fill_betweenx(self.adjustD_pf, self.adjustD_tf, (self.adjustD_pf - self.pole[1]) / self.v_des + self.pole[0],
                         fc='b' if not c else c, alpha=alpha, zorder=3)  # label='sweet zone'
        if self.boundP_cutOff_FLAG:
            ax.fill_between(self.boundP_tf, self.boundP_pf, np.interp(self.boundP_tf, self.adjustD_tf, self.adjustD_pf),
                            fc='b' if not c else c, alpha=alpha, zorder=5)  # label='normal zone'
        else:
            ax.fill_between(self.boundP_tf[:-2], self.boundP_pf[:-2],
                            np.interp(self.boundP_tf[:-2], self.adjustD_tf, self.adjustD_pf), fc='b' if not c else c,
                            alpha=alpha, zorder=5)  # label='normal zone'
            pf_dash = np.linspace(self.boundP_pf[-3], self.pf_road, 20, endpoint=True)
            tf_dash = (pf_dash - self.boundP_pf[-3]) / self.v_des + self.boundP_tf[-3]
            ax.plot(tf_dash, pf_dash, 'gray', linestyle='-.', lw=lw / 2)
            ax.fill_between(tf_dash, pf_dash, np.maximum(np.interp(tf_dash, self.adjustD_tf, self.adjustD_pf), pf_dash),
                            fc='b' if not c else c, alpha=alpha, zorder=5)
            tf_const = np.linspace(self.boundP_tf[-3], self.boundP_tf[-1], 100, endpoint=True)
            pf_const = np.linspace(self.boundP_pf[-3], self.boundP_pf[-1], len(tf_const), endpoint=True)
            ax.fill_between(tf_const, pf_const, np.minimum(np.interp(tf_const, tf_dash, pf_dash),
                                                           np.interp(tf_const, self.adjustD_tf, self.adjustD_pf)),
                            fc='b' if not c else c, alpha=alpha, zorder=5)

        ''' draw the input states '''
        self.get_intersection(input_tf, input_pf, overlap_FLAG)
        if self.intersection and input_tf:
            ax.scatter(input_tf, input_pf, c='purple', marker='x', zorder=9)  # label='input states'
            ax.plot([self.intersection[0], input_tf], [self.intersection[1], input_pf], 'purple',
                    linestyle='--', lw=lw / 2, zorder=4)
            tf_min = (input_pf - self.pole[1]) / self.v_des + self.pole[0]
            ax.annotate("", xytext=(input_tf, input_pf), xy=(tf_min, input_pf),
                        arrowprops=dict(arrowstyle="->", color='purple', connectionstyle='angle3', lw=1.5), zorder=7)
            # ax.annotate(r"$j_\mathrm{cost}=%.1f~\mathrm{s}$" % (input_tf - tf_min), xy=(tf_min, input_pf),
            #             xytext=(tf_min - 4.5, input_pf + 20), zorder=10)  # TODO: position adjust

        plt.tight_layout()
        # # plt.savefig('../../03 Data saving/bangbangController/' + 'sp_9.svg', dpi=600)
        return ax

    def get_intersection(self, input_tf=None, input_pf=None, overlap_FLAG='sweet'):
        if input_tf is not None and input_pf is not None:
            """ special cases """
            if not self.adjustD_whole_FLAG and overlap_FLAG == 'sweet' and input_tf >= self.adjustD_tf[-3] and abs(
                    (input_tf - self.adjustD_tf[-3]) * self.v_des + self.adjustD_pf[-3] - input_pf) < 0.1 * self.v_des:
                self.intersection = [self.adjustD_tf[-3], self.adjustD_pf[-3], 'sweet_zone']  # special, opt_tf
            elif self.boundP_cutOff_FLAG and input_tf >= self.boundP_tf[-3] and abs(
                    (input_tf - self.boundP_tf[-3]) * self.v_des + self.boundP_pf[-3] - input_pf) < 0.1 * self.v_des:
                self.intersection = [self.boundP_tf[-3], self.boundP_pf[-3], 'normal_zone']  # special, opt_pf
            elif input_pf == self.pf_road and abs(input_tf - self.tf_Range[0]) < 1e-4:
                self.intersection = [self.pole[0], self.pole[1], 'sweet_zone']
            elif input_pf == self.pf_road and abs(input_tf - self.tf_Range[-1]) < 1e-2:
                self.intersection = [self.boundP_tf[-1], self.boundP_pf[-1], 'normal_zone']  # special, end
            elif self.bbc.stage0_FLAG and abs(
                    (input_tf - self.pole[0]) * self.v_des + self.pole[1] - input_pf) < 0.1 * self.v_des:
                self.intersection = [self.pole[0], self.pole[1], 'sweet_zone']  # special, pole
            elif self.pole[1] <= input_pf <= self.pf_road and overlap_FLAG == 'sweet' and \
                    (input_pf - self.pole[1]) / self.v_des + self.pole[0] <= input_tf <= \
                    np.interp(input_pf, self.adjustD_pf, self.adjustD_tf):  # case: sweet zone
                pf_diff = self.v_des * (self.adjustD_tf - input_tf) + input_pf - self.adjustD_pf
                inter_tf = np.interp(0, pf_diff, self.adjustD_tf)
                inter_pf = np.interp(0, pf_diff, self.adjustD_pf)
                self.intersection = [inter_tf, inter_pf, 'sweet_zone']
            elif self.boundP_cutOff_FLAG:
                if self.pole[0] <= input_tf <= self.boundP_tf[-1] and np.interp(
                        input_tf, self.boundP_tf, self.boundP_pf) <= input_pf <= \
                        min((input_tf - self.pole[0]) * self.v_des + self.pole[1], self.pf_road):
                    pf_diff = self.v_des * (self.boundP_tf - input_tf) + input_pf - self.boundP_pf
                    inter_tf = np.interp(0, pf_diff, self.boundP_tf)
                    inter_pf = np.interp(0, pf_diff, self.boundP_pf)
                    self.intersection = [inter_tf, inter_pf, 'normal_zone']
                else:
                    print(f'invalid input (tf,pf): {input_tf, input_pf} at get_intersection, tf_range: {self.tf_Range}')
                    print(f'(t, p, v, a): {self.t_st, self.p_st, self.v0, self.a0}')
                    print(f'(v_des, v_min): {self.v_des, self.v_min}')
                    sys.exit()
            else:  # self.boundP_cutOff_FLAG = False
                tf_dash = (self.pf_road - self.boundP_pf[-3]) / self.v_des + self.boundP_tf[-3]
                tf_end = self.boundP_tf[-1] if self.v_min != 0 else np.inf

                if abs((input_tf - self.boundP_tf[-3]) * self.v_min + self.boundP_pf[-3] - input_pf) < 1e-4:
                    self.intersection = [input_tf, input_pf, 'const_zone']  # debug here, 2022-11-23
                elif self.pole[0] <= input_tf <= self.boundP_tf[-3] and np.interp(
                        input_tf, self.boundP_tf, self.boundP_pf) <= input_pf <= \
                        min((input_tf - self.pole[0]) * self.v_des + self.pole[1], self.pf_road):
                    pf_diff = self.v_des * (self.boundP_tf - input_tf) + input_pf - self.boundP_pf
                    inter_tf = np.interp(0, pf_diff, self.boundP_tf)
                    inter_pf = np.interp(0, pf_diff, self.boundP_pf)
                    self.intersection = [inter_tf, inter_pf, 'normal_zone']
                elif self.boundP_tf[-3] <= input_tf <= tf_dash and (
                        input_tf - self.boundP_tf[-3]) * self.v_des + self.boundP_pf[-3] <= input_pf <= \
                        min((input_tf - self.pole[0]) * self.v_des + self.pole[1], self.pf_road):
                    pf_diff = self.v_des * (self.boundP_tf - input_tf) + input_pf - self.boundP_pf
                    inter_tf = np.interp(0, pf_diff, self.boundP_tf)
                    inter_pf = np.interp(0, pf_diff, self.boundP_pf)
                    self.intersection = [inter_tf, inter_pf, 'normal_zone']
                elif self.boundP_tf[-3] <= input_tf <= tf_end and (
                        input_tf - self.boundP_tf[-3]) * self.v_min + self.boundP_pf[-3] <= input_pf <= \
                        min((input_tf - self.boundP_tf[-3]) * self.v_des + self.boundP_pf[-3], self.pf_road):
                    inter_tf = (input_pf - self.boundP_pf[-3] - input_tf * self.v_des + self.boundP_tf[
                        -3] * self.v_min) / (self.v_min - self.v_des)
                    inter_pf = (input_pf * self.v_min - self.boundP_pf[-3] * self.v_des + (
                            self.boundP_tf[-3] - input_tf) * self.v_des * self.v_min) / (self.v_min - self.v_des)
                    self.intersection = [inter_tf, inter_pf, 'const_zone']
                else:
                    print(f'invalid input (tf,pf): {input_tf, input_pf} at get_intersection, tf_range: {self.tf_Range}')
                    print(f'(t, p, v, a): {self.t_st, self.p_st, self.v0, self.a0}')
                    print(f'(v_des, v_min): {self.v_des, self.v_min}')
                    sys.exit()

        return self.intersection

    def get_trajectory_withPF(self, tf=None, pf=None, pf_end=Road.MERGE_ZONE_LENGTH + 50, overlap_FLAG='sweet'):
        tf = self.tf_Range[0] if tf is None else tf
        pf = self.pf_road if pf is None else pf

        self.get_intersection(tf, pf, overlap_FLAG)
        if abs(self.intersection[0] - self.pole[0]) < 0.1:  # debug here
            if self.bbc.stage0_FLAG:
                self.traj_inter = None
            elif self.bbc.stage1_FLAG:
                self.traj_inter = self.bbc.stage2_zero2min.get_traj()
            else:
                self.traj_inter = self.bbc.stage2_max2min.get_traj()
        elif self.intersection[2] == 'sweet_zone':
            self.traj_inter = self.bbc.stage3_max2zero2min.get_traj(self.intersection[1])
        elif self.intersection[2] == 'normal_zone':
            self.traj_inter = self.bbc.stage3_min2max2min.get_traj(['tf', self.intersection[0]])
        elif self.intersection[2] == 'const_zone':
            self.traj_inter = self.bbc.stage5_min2max2zero2max2min.get_traj(self.intersection[0])

        self.traj_planned = self.uniform_till(self.traj_inter, pf_end=pf_end)
        if self.t_st - self.t0 >= 0.1:
            self.traj_planned = self.uniform_before(self.traj_planned, self.t_st - self.t0)
        return self.traj_planned

    def uniform_till(self, traj, pf_end=None):
        pf_end = Road.MERGE_ZONE_LENGTH + 50 if pf_end is None else pf_end
        if traj:
            t, planned_a, planned_v, planned_p = zip(*traj)
        else:
            t, planned_a, planned_v, planned_p = [self.t0 + 1 / 10], [self.a0], [self.v0], [self.p0 + self.v0 * 0.1]

        t_uniform = (pf_end - planned_p[-1]) / self.v_des
        t_ = np.arange(1, t_uniform * 10 + 1) / 10
        planned_a = np.append(planned_a, np.zeros(np.shape(t_)))
        planned_v = np.append(planned_v, np.ones(np.shape(t_)) * self.v_des)
        planned_p = np.append(planned_p, planned_p[-1] + t_ * self.v_des)
        t = np.append(t, t[-1] + t_)
        return list(zip(t, planned_a, planned_v, planned_p))

    def uniform_before(self, traj, t_uniform):
        if traj:
            t, planned_a, planned_v, planned_p = zip(*traj)
        else:
            t, planned_a, planned_v, planned_p = [], [], [], []

        if t_uniform >= 0.1:
            t_ = np.arange(self.t0 * 10 + 1, self.t_st * 10 + 1) / 10
            planned_a = np.insert(planned_a, 0, np.zeros(np.shape(t_)))
            planned_v = np.insert(planned_v, 0, self.v0 * np.ones(np.shape(t_)))

            uniform_p = (t_ - self.t0) * self.v0 + self.p0
            delta_p = np.array(planned_p) - self.p_st
            planned_p = np.append(uniform_p, uniform_p[-1] + delta_p)

            t = np.append(t_, (np.array(t) - self.t_st) + t_[-1])
        return list(zip(t, planned_a, planned_v, planned_p))


if __name__ == '__main__':
    """ 1. generate Long_statePhase, corresponding to the demo in BBC.py """
    t1, p1, v1, a1 = 0, -60, 10, 0
    v_des, v_min = 20, 5
    route = 'R0'

    s1 = Long_StatePhase(t1, p1, v1, a1, route=route, v_des=v_des, v_min=v_min)
    # # ax = s1.plot_phase()
    # ax = s1.plot_phase(input_tf=18, input_pf=160)
    # plt.savefig('phase_1.svg')

    """ 2. show planned trajectory for given input """
    tf, pf, overlap_FLAG = 16, 160, 'sweet'
    s1.plot_phase(input_tf=tf, input_pf=pf, overlap_FLAG=overlap_FLAG)
    # traj = s1.get_trajectory_withPF(tf, pf, pf_end=200, overlap_FLAG='sweet')
    traj = s1.get_trajectory_withPF(pf_end=200)
    show_severalTrajs({'k': traj})

    """ **** Debug **** """
    # t0, p0, v0, a0 = 58.9, -335.0578642702442, 15.955895547627486, -1.8653295963081042
    # v_des, v_min = 20, 10
    # s1 = Long_StatePhase(t0, p0, v0, a0, v_des=v_des, v_min=v_min, route='M1')
    # s1.plot_phase()

    plt.show()
