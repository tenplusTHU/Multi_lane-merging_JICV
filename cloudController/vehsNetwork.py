"""
    Description: structure of the vehicle network
    Author: Tenplus
    Create-time: 2022-11-08
"""

import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from math import factorial
from objects.Road import Road
from matplotlib.patches import Polygon, Arc
from itertools import combinations
from objects.Vehicle import Vehicle
from cloudController.statePhase import Long_StatePhase
from cloudController.graphSearch_DCG import Search_DCG
from utilities.plotResults import show_trajs_threeLanes

""" Global Parameters  """
vn_width = 1  # unitization
vn_height = 1


class Node_VehGroup:
    Markers = {'r': 'o', 'b': 's', 'purple': 'p'}  # only the leader
    MarkerGaps = {'o': 0.03, 's': 0.03, 'p': 0.03}
    MarkerSize = {'o': 6.8, 's': 6, 'p': 7.7}
    ScheRanges = {'R0': Road.R0_SCH_RANGE, 'M1': Road.M2_SCH_RANGE, 'M2': Road.M2_SCH_RANGE}  # debug, 2022-12-28
    Layers = {'R0': -vn_height, 'M1': 0, 'M2': vn_height}

    def __init__(self, veh: Vehicle):
        self.veh = veh
        self.name = veh.name
        self.typeP = veh.name.split('_')[2]
        self.sizeP = len(veh.followers) + 1

        self.color = veh.color
        self.color_F = self.color if self.typeP == 'CAVs' else 'gray'
        self.marker = self.Markers[veh.color]
        self.markerGap = self.MarkerGaps[self.marker]
        self.markerSize = self.MarkerSize[self.marker]

        pos_0 = np.interp(veh.long_p, self.ScheRanges[veh.route], [0, vn_width])
        self.position = [pos_0 - self.markerGap * (self.sizeP - 1), pos_0]  # [left, right]
        self.layer = self.Layers[veh.route]
        if veh.route == 'M2' and veh.lat_p < Road.LANE_WIDTH:
            self.layer = self.Layers['M1']  # debug 2022-12-28, for vehicle that is lane-changing now.

        """ veh_M1 that is lane-changing now """
        self.FLAG_M1_nowLC = False  # veh_M1_LC that keeps in the original lane, 2022-12-29

    def plot_followers(self, ax, color, order=8):
        for i in range(self.sizeP)[1:]:
            ax.plot(self.position[1] - self.markerGap * i, self.layer, color=color,
                    marker=self.marker, markersize=self.markerSize, zorder=order)


class VehsNetwork:
    """ global parameters """
    LC_max_numP = 2  # constraint: the maximum number of platoons the LC simultaneously
    LC_max_sizeP_CAVs = 2  # restriction: CAVs platoons with sizeP > 2 do not change lanes. Luo said

    # note: 1) mixed platoon do not LC; 2) only CAVs platoon with sizeP <= 2 LC.

    def __init__(self, vehs_R0: dict, vehs_M1: dict, vehs_M2: dict,
                 rootVehL_M1=None, rootVehL_M2=None,
                 LC_max_numP=LC_max_numP, LC_max_sizeP_CAVs=LC_max_sizeP_CAVs, ):
        """ initialize """
        self.vehs_All = dict(vehs_R0, **vehs_M1, **vehs_M2)  # {veh_name: veh}
        if not self.vehs_All:
            print('vehs_All is None at VehsNetwork.')  # add, 2023-02-15
            sys.exit()

        self.LC_max_numP = LC_max_numP
        self.LC_max_sizeP_CAVs = LC_max_sizeP_CAVs

        self.rootVehL_M1 = rootVehL_M1
        self.rootVehL_M2 = rootVehL_M2  # 2022-12-29, for continuous

        """ get the sorted [veh_name] in each **route** """
        self.vehsL_R0, self.vehsL_M1 = [], []
        vehsL_M2_posDict = {}

        for veh in self.vehs_All.values():
            if veh.route == 'R0' and veh.name[-1] == 'L':
                self.vehsL_R0.append(veh.name)
            elif veh.route == 'M1' and veh.name[-1] == 'L':
                self.vehsL_M1.append(veh.name)
            elif veh.route == 'M2' and veh.name[-1] == 'L':
                vehsL_M2_posDict[veh.name] = veh.long_p
        vehsL_M2_posDict = sorted(vehsL_M2_posDict.items(), key=lambda item: item[1], reverse=True)
        self.vehsL_M2 = [vehPos[0] for vehPos in vehsL_M2_posDict]

        self.vehsNum_R0 = len(self.vehsL_R0)  # only leader
        self.vehsNum_M1 = len(self.vehsL_M1)
        self.vehsNum_M2 = len(self.vehsL_M2)

        """ cps mapping and create vehicle-group nodes in the network """  # nodes in each route!
        self.Nodes_All, self.Nodes_R0, self.Nodes_M1, self.Nodes_M2, self.pos_bound = self.__CPS_mapping()
        self.pos_root = self.pos_bound[1] + 0.13 * vn_width
        self.pos_text = self.pos_bound[1] + 0.02 * vn_width
        self.pos_xlim = [-0.05 * vn_width + self.pos_bound[0], 0.25 * vn_width + self.pos_bound[1]]

        """ initialize vehicle group and lane-change vehicle information  """
        self.vehs_M1_LC, self.vehs_M1_noLC = [], []  # [veh_name]: list
        self.sequence_Lane1, self.sequence_Lane2 = None, None  # [veh_name]: list

        self.LC_M1_names, self.LC_M1_cases, self.LC_cases_num = self.__get_LC_M1_cases()
        self.statePhases_All, self.statePhases_LC = self.__generate_statePhases()

        self.schedules = []
        self.optSchedule = None

        """ create figures """
        self.fig_whole, self.fig_subs = None, None

    def __CPS_mapping(self):
        """ create vehicle group nodes """
        Nodes_R0 = {}
        for veh_name in self.vehsL_R0:
            Nodes_R0[veh_name] = Node_VehGroup(self.vehs_All[veh_name])

        Nodes_M1 = {}
        for veh_name in self.vehsL_M1:
            Nodes_M1[veh_name] = Node_VehGroup(self.vehs_All[veh_name])

        Nodes_M2 = {}
        for veh_name in self.vehsL_M2:
            Nodes_M2[veh_name] = Node_VehGroup(self.vehs_All[veh_name])
            ''' deal with veh_M1_LC that is now lane-changing '''  # add 2022-12-29
            veh = self.vehs_All[veh_name]
            if veh.name[0:2] == 'M1' and veh.lat_p < Road.LANE_WIDTH:
                Nodes_M2[veh_name].FLAG_M1_nowLC = True

        ''' get the pos_bound '''
        pos_bound = [np.inf, -np.inf]
        for node in dict(Nodes_R0, **Nodes_M1, **Nodes_M2).values():
            pos_bound[0] = min(pos_bound[0], node.position[0])
            pos_bound[1] = max(pos_bound[1], node.position[1])

        Nodes_All = dict(Nodes_R0, **Nodes_M1, **Nodes_M2)
        return Nodes_All, Nodes_R0, Nodes_M1, Nodes_M2, pos_bound

    def __get_LC_M1_cases(self):
        """ sort [M1_veh_name] list by priority: 1) sizeP, 2) typeP, 3) position """
        LC_M1_sorted = []  # [veh_name]: sorted list
        for sizeP in np.arange(self.LC_max_sizeP_CAVs) + 1:
            # CAVs platoon
            for veh_name in self.vehsL_M1:
                veh = self.vehs_All[veh_name]
                if veh.sizeP == sizeP and (veh.typeP == 'CAVs' or veh.sizeP == 1):
                    LC_M1_sorted.append(veh_name)
            # HDVs platoon do not LC
        # print(f'LC_M1_sorted: {LC_M1_sorted}')  # sorted completed

        """ get all the LC combinations """
        LC_veh_M1_cases = {0: [[]]}  # {LC_numP: [veh_name]}
        LC_cases_num = 1
        for LC_numP in np.arange(min(len(LC_M1_sorted), self.LC_max_numP)) + 1:
            LC_veh_M1_cases[LC_numP] = list(combinations(LC_M1_sorted, LC_numP))
            LC_cases_num += factorial(len(LC_M1_sorted)) // (
                    factorial(LC_numP) * factorial(len(LC_M1_sorted) - LC_numP))
        # print(f'LC_veh_M1_cases: {LC_veh_M1_cases}, cases_num: {LC_cases_num}')
        return LC_M1_sorted, LC_veh_M1_cases, LC_cases_num

    def __generate_statePhases(self):
        statePhases_All, statePhases_LC = {}, {}

        ''' generate statePhases for all vehicles '''
        for veh_name in self.vehsL_R0 + self.vehsL_M1 + self.vehsL_M2:
            # print(veh_name)
            veh = self.vehs_All[veh_name]
            t, p, v, a = veh.time_c, veh.long_p, veh.long_v, veh.long_a
            v_des, v_min = veh.v_des, veh.v_min
            statePhases_All[veh.name] = Long_StatePhase(t, p, v, a, veh.route, v_des=v_des, v_min=v_min)
            # statePhases_All[veh.name].plot_phase()

        ''' generate statePhases for vehicles with lane-change behavior '''
        for veh_name in self.LC_M1_names:
            veh = self.vehs_All[veh_name]
            t, p, v, a = veh.time_c, veh.long_p, veh.long_v, veh.long_a
            v_des, v_min = Vehicle.v_des['M2'], Vehicle.v_min['M2']
            statePhases_LC[veh.name] = Long_StatePhase(t, p, v, a, 'M2', v_des=v_des, v_min=v_min)
            # statePhases_LC[veh.name].plot_phase()

        return statePhases_All, statePhases_LC

    def get_optimalSchedule(self):
        self.schedules = []  # refresh
        case_id = 0

        ''' schedule for each LC scheme '''
        for LC_numP in self.LC_M1_cases:
            for LC_vehsName_M1 in self.LC_M1_cases[LC_numP]:
                case_id += 1
                schedule = Search_DCG(self, LC_vehsName_M1)
                self.schedules.append(schedule)
                # TODO
                # print(f'**** scheduling, case: {case_id}/{self.LC_cases_num}, LC_vehsName_M1: {LC_vehsName_M1} \n'
                #       f'     J: %.3f s, time counting: %.3f s' % (schedule.J_total, schedule.t_search_all))

        ''' get the optimal schedule '''
        J_list = [schedule.J_total for schedule in self.schedules]
        Jmin_index = J_list.index(min(J_list))

        self.optSchedule = self.schedules[Jmin_index]
        return self.optSchedule

    def showDCG_whole(self):
        """ axes setting """
        if not self.fig_whole:
            self.fig_whole = plt.figure(figsize=(5, 3))
        ax = self.fig_whole.add_subplot(label='whole')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylim([-1.3 * vn_height, 1.3 * vn_height])
        ax.set_xlim(self.pos_xlim)

        plt.tight_layout()

        """ plot nodes """
        for node in self.Nodes_All.values():
            ax.plot(node.position[1], node.layer, c=node.color, marker=node.marker, ms=node.markerSize, zorder=9)
            node.plot_followers(ax, color=node.color_F, order=8)

        """ plot links """
        for r0 in range(self.vehsNum_R0)[:-1]:
            thisNode = self.Nodes_R0[self.vehsL_R0[r0]]
            nextNode = self.Nodes_R0[self.vehsL_R0[r0 + 1]]
            self.__plot_dashLines(ax, [thisNode.position[0], thisNode.layer], [nextNode.position[1], nextNode.layer])
        for m1 in range(self.vehsNum_M1)[:-1]:
            thisNode = self.Nodes_M1[self.vehsL_M1[m1]]
            nextNode = self.Nodes_M1[self.vehsL_M1[m1 + 1]]
            self.__plot_dashLines(ax, [thisNode.position[0], thisNode.layer], [nextNode.position[1], nextNode.layer])
        for m2 in range(self.vehsNum_M2)[:-1]:
            thisNode = self.Nodes_M2[self.vehsL_M2[m2]]
            nextNode = self.Nodes_M2[self.vehsL_M2[m2 + 1]]
            self.__plot_dashLines(ax, [thisNode.position[0], thisNode.layer], [nextNode.position[1], nextNode.layer])

        ''' links between M1-R0 & M1-M2'''
        for node_M1 in self.Nodes_M1.values():
            for node_R0 in self.Nodes_R0.values():
                self.__plot_dashLines(ax, [node_M1.position[1], node_M1.layer], [node_R0.position[1], node_R0.layer])

        for node_M2 in self.Nodes_M2.values():
            if node_M2.FLAG_M1_nowLC is False:  # add, 2022-12-29
                for node_M1 in self.Nodes_M1.values():
                    self.__plot_dashLines(ax, [node_M1.position[1], node_M1.layer],
                                          [node_M2.position[1], node_M2.layer])

        ''' plot the two roots and the links '''  # modified, 2023-02-15
        if self.vehsL_R0 or self.vehsL_M1:
            ax.scatter(self.pos_root, 0, marker='D', fc='w', ec='k', zorder=8, s=30)
            if self.vehsL_R0:
                node_R0_0 = self.Nodes_R0[self.vehsL_R0[0]]  # first node in R0
                self.__plot_dashLines(ax, [self.pos_root, 0], [node_R0_0.position[1], node_R0_0.layer])
            if self.vehsL_M1:
                node_M1_0 = self.Nodes_M1[self.vehsL_M1[0]]
                self.__plot_dashLines(ax, [self.pos_root, 0], [node_M1_0.position[1], node_M1_0.layer])

        if self.vehsL_M1 or self.vehsL_M2:
            ax.scatter(self.pos_root, vn_height, marker='D', fc='w', ec='k', zorder=8, s=30)
            if self.vehsL_M1:
                node_M1_0 = self.Nodes_M1[self.vehsL_M1[0]]
                self.__plot_dashLines(ax, [self.pos_root, vn_height], [node_M1_0.position[1], node_M1_0.layer])
            if self.vehsL_M2:
                node_M2_0 = self.Nodes_M2[self.vehsL_M2[0]]
                self.__plot_dashLines(ax, [self.pos_root, vn_height], [node_M2_0.position[1], node_M2_0.layer])

        return ax

    def addSequence_wholeDCG_forDemo(self, LC_vehPos_M1, pos_R0, pos_M1_LC):
        """ useless, only for demo """
        ax = self.fig_whole.get_axes()[0]

        """ generate sequence """
        # vehs_M1_LC & vehs_M1_noLC, id from 0
        self.vehs_M1_LC, self.vehs_M1_noLC = [], []  # refresh here
        for pos in range(self.vehsNum_M1):
            if pos in LC_vehPos_M1:
                self.vehs_M1_LC.append(self.vehsL_M1[pos])
            else:
                self.vehs_M1_noLC.append(self.vehsL_M1[pos])

        # vehicle sequence on main-lane 1
        string = [i for i in range(self.vehsNum_R0 + len(self.vehs_M1_noLC))]
        # pos_R0 = [2, 3, 5, 6]  # given vehicles_R0 positions
        pos_M1_noLC = list(set(string) - set(pos_R0))
        self.sequence_Lane1 = np.empty(self.vehsNum_R0 + len(self.vehs_M1_noLC), dtype=object)
        self.sequence_Lane1[pos_R0] = self.vehsL_R0
        self.sequence_Lane1[pos_M1_noLC] = self.vehs_M1_noLC

        # vehicle sequence on main-lane 2
        string = [i for i in range(self.vehsNum_M2 + len(self.vehs_M1_LC))]
        # pos_M1_LC = [2, 4]  # given vehicles_M1_LC positions
        pos_M2 = list(set(string) - set(pos_M1_LC))
        self.sequence_Lane2 = np.empty(self.vehsNum_M2 + len(self.vehs_M1_LC), dtype=object)
        self.sequence_Lane2[pos_M1_LC] = self.vehs_M1_LC
        self.sequence_Lane2[pos_M2] = self.vehsL_M2

        """ plot the link and arrows between nodes """
        # sequence_Lane1
        if len(self.sequence_Lane1) > 0:
            first_node = self.Nodes_All[self.sequence_Lane1[0]]
            self.__plot_solidLines_withArrow(ax, [self.pos_root, 0], [first_node.position[1], first_node.layer])
            for i in range(len(self.sequence_Lane1))[:-1]:
                this_node = self.Nodes_All[self.sequence_Lane1[i]]
                next_node = self.Nodes_All[self.sequence_Lane1[i + 1]]
                self.__plot_solidLines_withArrow(ax, [this_node.position[0], this_node.layer],
                                                 [next_node.position[1], next_node.layer])

        # sequence_Lane2
        if len(self.sequence_Lane2) > 0:
            first_node = self.Nodes_All[self.sequence_Lane2[0]]
            self.__plot_solidLines_withArrow(ax, [self.pos_root, vn_height],
                                             [first_node.position[1], first_node.layer])
            for i in range(len(self.sequence_Lane2))[:-1]:
                this_node = self.Nodes_All[self.sequence_Lane2[i]]
                next_node = self.Nodes_All[self.sequence_Lane2[i + 1]]
                self.__plot_solidLines_withArrow(ax, [this_node.position[0], this_node.layer],
                                                 [next_node.position[1], next_node.layer])

    def showDCG_subs(self, fig_subs=None, LC_vehsPos_M1=None, fig_size=(4.5, 3)):
        """ axes setting """
        if not fig_subs:
            if not self.fig_subs:
                self.fig_subs = plt.figure(figsize=fig_size)
            fig_subs = self.fig_subs
        else:
            self.fig_subs = fig_subs

        for ax in fig_subs.get_axes():
            ax.remove()

        ax_lo = fig_subs.add_subplot(2, 1, 2, label='lower')
        ax_up = fig_subs.add_subplot(2, 1, 1, label='upper')

        for ax in [ax_lo, ax_up]:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim([-0.05 * vn_width + self.pos_bound[0], 0.25 * vn_width + self.pos_bound[1]])
        ax_lo.set_ylim([-1.3 * vn_height, 0.3 * vn_height])
        ax_up.set_ylim([-0.3 * vn_height, 1.3 * vn_height])
        plt.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)

        """ load or generate the lane-change info of vehs_M1_LC """
        if LC_vehsPos_M1 is None:  # [list], for demo
            LC_vehNum_M1 = 2  # TODO: number of LC vehicles
            LC_vehsPos_M1 = random.sample(range(self.vehsNum_M1), LC_vehNum_M1)
            LC_vehsPos_M1.sort()
        ''' refresh the LC_vehPos_M1 '''
        self.vehs_M1_LC, self.vehs_M1_noLC = [], []  # refresh here
        for pos in range(self.vehsNum_M1):
            if pos in LC_vehsPos_M1:
                self.vehs_M1_LC.append(self.vehsL_M1[pos])
            else:
                self.vehs_M1_noLC.append(self.vehsL_M1[pos])

        """ plot nodes """
        # R0 and M2
        for node in dict(self.Nodes_R0, **self.Nodes_M2).values():
            ax = ax_up if node.name in self.vehsL_M2 else ax_lo
            ax.plot(node.position[1], node.layer, c=node.color, marker=node.marker, ms=node.markerSize, zorder=9)
            node.plot_followers(ax, color=node.color_F, order=8)

        # M1, with & without LC
        for node in self.Nodes_M1.values():
            if node.name in self.vehs_M1_noLC:
                ax_self, ax_shadow = ax_lo, ax_up
            else:
                ax_self, ax_shadow = ax_up, ax_lo

            ax_self.plot(node.position[1], node.layer, c=node.color, marker=node.marker, ms=node.markerSize, zorder=9)
            ax_shadow.plot(node.position[1], node.layer, c='lightgrey', marker=node.marker, ms=node.markerSize,
                           zorder=5)
            node.plot_followers(ax_self, color=node.color_F, order=8)
            node.plot_followers(ax_shadow, color='lightgrey', order=5)

        """ plot links """
        for r0 in range(self.vehsNum_R0)[:-1]:
            thisNode = self.Nodes_R0[self.vehsL_R0[r0]]
            nextNode = self.Nodes_R0[self.vehsL_R0[r0 + 1]]
            self.__plot_dashLines(ax_lo, [thisNode.position[0], thisNode.layer], [nextNode.position[1], nextNode.layer])
        for m1 in range(self.vehsNum_M1)[:-1]:
            thisNode = self.Nodes_M1[self.vehsL_M1[m1]]
            nextNode = self.Nodes_M1[self.vehsL_M1[m1 + 1]]
            self.__plot_dashLines(ax_lo, [thisNode.position[0], thisNode.layer], [nextNode.position[1], nextNode.layer])
            self.__plot_dashLines(ax_up, [thisNode.position[0], thisNode.layer], [nextNode.position[1], nextNode.layer])
        for m2 in range(self.vehsNum_M2)[:-1]:
            thisNode = self.Nodes_M2[self.vehsL_M2[m2]]
            nextNode = self.Nodes_M2[self.vehsL_M2[m2 + 1]]
            self.__plot_dashLines(ax_up, [thisNode.position[0], thisNode.layer], [nextNode.position[1], nextNode.layer])

        ''' links between M1-R0 & M1-M2'''
        for node_M1 in self.Nodes_M1.values():
            if node_M1.name in self.vehs_M1_noLC:
                for node_R0 in self.Nodes_R0.values():
                    self.__plot_dashLines(ax_lo,
                                          [node_M1.position[1], node_M1.layer], [node_R0.position[1], node_R0.layer])
        for node_M1 in self.Nodes_M1.values():
            if node_M1.name in self.vehs_M1_LC:
                for node_M2 in self.Nodes_M2.values():
                    if node_M2.FLAG_M1_nowLC is False:  # debug, 2022-12-29
                        self.__plot_dashLines(ax_up, [node_M1.position[1], node_M1.layer],
                                              [node_M2.position[1], node_M2.layer])

        ''' plot the two roots and the links '''
        if self.vehsL_R0 or self.vehs_M1_noLC:
            ax_lo.scatter(self.pos_root, 0, marker='D', fc='w', ec='k', zorder=8, s=30)
            if self.vehsL_R0:
                node_R0_0 = self.Nodes_R0[self.vehsL_R0[0]]  # first node in R0
                self.__plot_dashLines(ax_lo, [self.pos_root, 0], [node_R0_0.position[1], node_R0_0.layer])
            if self.vehs_M1_noLC:
                node_M1_0 = self.Nodes_M1[self.vehsL_M1[0]]
                self.__plot_dashLines(ax_lo, [self.pos_root, 0], [node_M1_0.position[1], node_M1_0.layer])  # 2022-12-29

        if self.vehs_M1_LC or self.vehsL_M2:
            ax_up.scatter(self.pos_root, vn_height, marker='D', fc='w', ec='k', zorder=8, s=30)
            if self.vehs_M1_LC:
                node_M1_LC_0 = self.Nodes_M1[self.vehs_M1_LC[0]]
                self.__plot_dashLines(ax_up, [self.pos_root, vn_height], [node_M1_LC_0.position[1], node_M1_LC_0.layer])
            if self.vehsL_M2:
                node_M2_0 = self.Nodes_M2[self.vehsL_M2[0]]
                self.__plot_dashLines(ax_up, [self.pos_root, vn_height], [node_M2_0.position[1], node_M2_0.layer])

        return ax_lo, ax_up

    def addSequences_subsDCG(self, pos_R0=None, pos_M1_LC=None, opt_J1=None, opt_J2=None):
        """ load axes """
        fig_subs = self.fig_subs

        for ax_ in fig_subs.get_axes():
            if ax_.get_label() == 'lower':
                ax_lo = ax_
            elif ax_.get_label() == 'upper':
                ax_up = ax_
            else:
                print(f'Error: cannot find available axes')
                sys.exit()

        """ load or generate the sequence """
        # R0-M1
        string = [i for i in range(self.vehsNum_R0 + len(self.vehs_M1_noLC))]
        if pos_R0 is None:  # generate
            pos_R0 = random.sample(list(string), self.vehsNum_R0)
            pos_R0.sort()
        pos_M1_noLC = list(set(string) - set(pos_R0))
        self.sequence_Lane1 = np.empty(self.vehsNum_R0 + len(self.vehs_M1_noLC), dtype=object)
        self.sequence_Lane1[pos_R0] = self.vehsL_R0
        self.sequence_Lane1[pos_M1_noLC] = self.vehs_M1_noLC

        # M1-M2
        string = [i for i in range(self.vehsNum_M2 + len(self.vehs_M1_LC))]
        if pos_M1_LC is None:  # generate
            pos_M1_LC = random.sample(list(string), len(self.vehs_M1_LC))
            pos_M1_LC.sort()
        pos_M2 = list(set(string) - set(pos_M1_LC))
        self.sequence_Lane2 = np.empty(self.vehsNum_M2 + len(self.vehs_M1_LC), dtype=object)
        self.sequence_Lane2[pos_M2] = self.vehsL_M2
        self.sequence_Lane2[pos_M1_LC] = self.vehs_M1_LC

        """ plot the link and arrows between nodes """
        # sequence_Lane1
        if len(self.sequence_Lane1) > 0:
            first_node = self.Nodes_All[self.sequence_Lane1[0]]
            self.__plot_solidLines_withArrow(ax_lo, [self.pos_root, 0], [first_node.position[1], first_node.layer])
            for i in range(len(self.sequence_Lane1))[:-1]:
                this_node = self.Nodes_All[self.sequence_Lane1[i]]
                next_node = self.Nodes_All[self.sequence_Lane1[i + 1]]
                self.__plot_solidLines_withArrow(ax_lo, [this_node.position[0], this_node.layer],
                                                 [next_node.position[1], next_node.layer])
            if opt_J1 is not None:
                ax_lo.text(self.pos_text, -1.1 * vn_height, r'$J_{M1}^*=%.1f~\mathrm{s}$' % opt_J1, size=12)

        # sequence_Lane2
        if len(self.sequence_Lane2) > 0:
            first_node = self.Nodes_All[self.sequence_Lane2[0]]
            self.__plot_solidLines_withArrow(ax_up, [self.pos_root, vn_height],
                                             [first_node.position[1], first_node.layer])
            for i in range(len(self.sequence_Lane2))[:-1]:
                this_node = self.Nodes_All[self.sequence_Lane2[i]]
                next_node = self.Nodes_All[self.sequence_Lane2[i + 1]]
                self.__plot_solidLines_withArrow(ax_up, [this_node.position[0], this_node.layer],
                                                 [next_node.position[1], next_node.layer])
            if opt_J2 is not None:
                ax_up.text(self.pos_text, -0.1 * vn_height, r'$J_{M2}^*=%.1f~\mathrm{s}$' % opt_J2, size=12)

    @staticmethod
    def __plot_dashLines(ax, start, end, color='gray', lw=0.6, alpha=0.6, order=0):
        ax.plot([start[0], end[0]], [start[1], end[1]], color=color, ls='--', lw=lw, alpha=alpha, zorder=order)

    @staticmethod
    def __plot_solidLines_withArrow(ax, start, end, color='k', lw=2, order=7):
        # solid line1
        ax.plot([start[0], end[0]], [start[1], end[1]], color=color, lw=lw, zorder=order)

        # arrow
        center = [(start[0] + end[0]) / 2, (start[1] + end[1]) / 2]
        angle = [end[0] - start[0], end[1] - start[1]]
        angle = np.array(angle) / np.linalg.norm(angle) / 50  # normalize

        ax.annotate('', xytext=(round(center[0], 4), round(center[1], 4)),
                    xy=(round(center[0] + angle[0], 4), round(center[1] + angle[1], 4)),
                    arrowprops={'arrowstyle': 'simple', 'fc': 'k', 'ec': 'k'}, zorder=order)


def createFig_subs(fig_size=(5, 3)):
    fig_split = plt.figure(figsize=fig_size)

    ax_lo = fig_split.add_subplot(2, 1, 2, label='lower')
    ax_up = fig_split.add_subplot(2, 1, 1, label='upper')

    for ax in [ax_lo, ax_up]:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)

    return fig_split


def createFig_split(fig_size=(5, 3)):
    fig_split = plt.figure(figsize=fig_size)

    ax_lo = fig_split.add_subplot(2, 1, 2, label='lower')
    ax_up = fig_split.add_subplot(2, 1, 1, label='upper')

    for ax in [ax_lo, ax_up]:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)

    return fig_split


def defaultNetwork_demo():
    """ load static routes """
    from routes.route_static import get_route_example_0

    vehs_R0, vehs_M1, vehs_M2, vehs_All = get_route_example_0()

    ''' show the simulation scenario '''
    # r = Road()
    # r.x_lim = [-600, 50]
    # r.plot(FLAG_fillRoad=True)
    #
    # for veh in vehs_All.values():
    #     veh.plot(r.ax_road, FLAG_speedColor=False)

    """ create the vehicles network """
    # whether the third group lane change
    # veh_LC = vehs_M1[list(vehs_M1.keys())[5]]
    # veh_LC.route = 'M2'
    # veh_LC.lat_p = Road.LANE_WIDTH / 2 + 2

    network = VehsNetwork(vehs_R0, vehs_M1, vehs_M2,
                          rootVehL_M1=None, rootVehL_M2=None,
                          LC_max_numP=2, LC_max_sizeP_CAVs=2)

    return network


if __name__ == '__main__':
    ''' 1.1 load vehicle network '''
    vn = defaultNetwork_demo()

    ''' 1.2 show the whole DCG (Dynamic conflict graph), for paper IV '''
    # ax_whole = vn.showDCG_whole()
    # vn.addSequence_wholeDCG_forDemo(LC_vehPos_M1=[2, 4], pos_R0=[2, 3, 5, 6], pos_M1_LC=[2, 5])

    ''' 1.3 show two split network '''
    # LC_vehsPos_M1 = [2, 4]
    # vn.showDCG_subs(LC_vehsPos_M1=LC_vehsPos_M1)
    # pos_R0, pos_M1_LC = [2, 3, 5, 6], [2, 5]
    # vn.addSequences_subsDCG(pos_R0=pos_R0, pos_M1_LC=pos_M1_LC, opt_J1=1, opt_J2=2)

    ''' 1.4 show all LC cases '''
    # print(f'** all LC cases num: {vn.LC_cases_num}, \n'
    #       f'** LC_M1_cases:{vn.LC_M1_cases} \n')
    # for LC_numP in vn.LC_M1_cases.keys():
    #     for LC_vehs_M1 in vn.LC_M1_cases[LC_numP]:
    #         LC_vehsPos_M1 = []
    #         for veh_name in LC_vehs_M1:
    #             LC_vehsPos_M1.append(vn.vehsL_M1.index(veh_name))
    #
    #         ax_lo, ax_up = vn.showDCG_subs(LC_vehsPos_M1=LC_vehsPos_M1, fig_size=(4, 3))
    #         ax_lo.set_xlim(-0.02, 0.98)
    #         ax_up.set_xlim(-0.02, 0.98)
    #         plt.pause(1)

    """ 2. get optimal schedule scheme """
    vn.get_optimalSchedule()

    ''' 2.1 get optimal trajectories '''
    vn.optSchedule.get_optimalTrajectories()
    vn.optSchedule.show_vehsNetwork()
    show_trajs_threeLanes(vn.vehs_All)

    plt.show()
