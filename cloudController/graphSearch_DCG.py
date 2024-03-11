"""
    Description: heuristic search algorithm for dynamic conflict graph (DCG), considering Lane-change behavior
    Author: Tenplus
    Create-time: 2022-11-17
    Update-time: 2022-02-20
    Note: # V3.0, 对这一部分有比较大的调整，主要是单纯只对图进行搜索，以和子图作为决策变量；
"""

import sys
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from math import factorial
from cloudController.statePhase import Long_StatePhase
from cloudController.safe_laneChange import StatePhase_safeLC
from objects.Road import Road
from objects.Vehicle import Vehicle
from itertools import combinations

""" Global parameters """
dt_safe_same = 1.2
dt_safe_diff = 1.5 + 0.3
alpha_dict = {'R0': 1, 'M1': 1.2, 'M2': 1.3}
alpha_M1_LC = 0.8


class TreeNode:
    Colors = {'R0': 'r', 'M1': 'b', 'M2': 'purple'}
    Markers = {'R0': 'o', 'M1': 's', 'M2': 'p'}
    MarkerSize = {'o': 30, 's': 28, 'p': 42, 'D': 35}

    def __init__(self, order, id_inLayer, veh_name, caseRange=None, caseMean=None):
        self.order = order
        self.id_inLayer = id_inLayer
        self.veh_name = veh_name

        self.color = self.Colors[veh_name[0:2]] if veh_name != 'root' else 'w'
        self.marker = self.Markers[veh_name[0:2]] if veh_name != 'root' else 'D'
        self.markerSize = self.MarkerSize[self.marker]

        self.caseRange = caseRange
        self.caseMean = caseMean if caseMean is not None else np.mean(caseRange)

        self.parent = None
        self.child_R0 = None
        self.child_M1 = None
        self.child_M2 = None

        self.tfChosen = None
        self.overlap_FLAG = 'sweet'
        self.LC_midT_optional = None  # for vehs_M1_LC, the lane-change time window
        self.j = None
        self.J1, self.J2 = 0, 0

        self.FLAG_searched = False
        self.FLAG_prunedBelow = False


class Schedule_R0M1:
    """ load constant parameters """
    dt_safe_same = dt_safe_same
    dt_safe_diff = dt_safe_diff
    alpha_dict = alpha_dict

    def __init__(self, network, LC_vehsName_M1=[]):
        self.network = network

        """ load vehicle-groups information """
        self.vehs_All = network.vehs_All
        self.vehsL_R0 = network.vehsL_R0
        self.vehsL_M1 = [veh_name for veh_name in network.vehsL_M1 if veh_name not in LC_vehsName_M1]

        self.LC_vehsName_M1 = LC_vehsName_M1

        self.vehsNum_R0 = len(self.vehsL_R0)  # only leader
        self.vehsNum_M1 = len(self.vehsL_M1)
        self.vehsNum_All = self.vehsNum_R0 + self.vehsNum_M1

        """ get sequences_dict and dividers_dict && generate all tree nodes with link """
        self.seqCases_num = factorial(self.vehsNum_All) // (factorial(self.vehsNum_R0) * factorial(self.vehsNum_M1))
        self.sequences_All, self.dividers_List = self.__get_all_seqCases()
        self.treeNodes = self.__generate_treeNodes() if self.vehsNum_All else None

        """ sequence tree R0-M1 """
        self.fig_tree = None
        self.optimal_J1 = [np.inf, None]  # [value, case_id]

        self.optSeq_Lane1_nodes = []  # [treeNode]: list
        self.optSeq_Lane1_nameL = []  # [veh_name]: list
        self.optSeq_pos_R0 = []

        self.vehs_long_tfChosen = {}  # {veh_name: tfChosen}
        self.vehs_long_trajPlan = {}  # {veh_name: trajPlan}

        """ global FLAGs about ***pause&save*** figures """
        self.FLAG_figPause, self.t_figPause = False, 0.1
        self.FLAG_figSave = False

    def __get_all_seqCases(self):
        sequences_All, dividers_List = [], []
        """ first, get all sequences """
        string = [i for i in range(self.vehsNum_All)]
        pos_R0_Dict = list(combinations(string, self.vehsNum_R0))

        for pos_R0 in pos_R0_Dict:
            sequence = np.empty(self.vehsNum_All, dtype=object)
            sequence[list(pos_R0)] = self.vehsL_R0
            sequence[list(set(string) - set(pos_R0))] = self.vehsL_M1
            sequences_All.append(sequence)
        sequences_All = np.array(sequences_All)[::-1]  # when searching: M1 first, R0 then

        """ then, get the dividers """
        dividers_former = [0]
        for order in range(self.vehsNum_All):
            dividers = []
            for seqCase_id in range(1, self.seqCases_num):
                if sum([sequences_All[seqCase_id - 1][order][0] == 'M',
                        sequences_All[seqCase_id][order][0] == 'M']) == 1:
                    dividers.append(seqCase_id)
            dividers = list(set(dividers).union(set(dividers_former)))
            dividers.sort()  # bug, here
            dividers_former = dividers
            dividers_List.append(dividers)

        return sequences_All, dividers_List

    def __generate_treeNodes(self):
        treeNodes = [{} for _ in range(self.vehsNum_All + 1)]  # first: root
        treeNodes[0][0] = TreeNode(0, 0, 'root', caseRange=[0, self.seqCases_num - 1],
                                   caseMean=(self.seqCases_num + 2 * self.dividers_List[0][-1] - 2) / 4)

        """ generate nodes layer by layer, width-first """
        for order in range(self.vehsNum_All):
            dividers = self.dividers_List[order]
            for divider in dividers:
                id_inLayer = dividers.index(divider)
                if divider != dividers[-1]:
                    caseRange = [divider, dividers[id_inLayer + 1] - 1]
                else:
                    caseRange = [divider, self.seqCases_num - 1]
                veh_name = self.sequences_All[divider][order]
                treeNode = TreeNode(order + 1, id_inLayer, veh_name, caseRange)  # note here: order + 1
                treeNodes[order + 1][id_inLayer] = treeNode

                """ create link between nodes """
                if order == 0:
                    parent_idInLayer = 0
                else:
                    parent_idInLayer = np.where(np.array(self.dividers_List[order - 1]) <= divider)[0][-1]
                treeNode.parent = treeNodes[order][parent_idInLayer]
                if veh_name[0] == 'R':
                    treeNode.parent.child_R0 = treeNode
                else:
                    treeNode.parent.child_M1 = treeNode

        return treeNodes

    def searchTree_depth(self, FLAG_figPause=False, FLAG_figSave=False):
        """ set global FLAGs """
        self.FLAG_figPause = FLAG_figPause
        self.FLAG_figSave = FLAG_figSave

        """ recursive search """
        if self.treeNodes:
            self.__searchNode_next(self.treeNodes[0][0])  # search from nodes

    def __searchNode_next(self, node):  # TODO, most importantly in this function
        if node.veh_name == 'root':
            pass
        else:
            node.FLAG_searched = True
            veh = self.vehs_All[node.veh_name]
            veh_statePhase = self.network.statePhases_All[veh.name]  # change, 2022-12-29

            """ decided desired_tf for each node, three cases """
            if node.parent.veh_name == 'root' and self.network.rootVehL_M1 is None:
                desired_tf = max(veh.long_tfChosen, veh_statePhase.tf_Range[0])
            elif node.parent.veh_name == 'root' and self.network.rootVehL_M1 is not None:  # add, 2022-12-29
                rootVehL = self.network.rootVehL_M1
                rootVehL_dt_inner = rootVehL.CFM.get_dt_inner(velocity=Vehicle.v_des['M1'])
                dt_safe = self.dt_safe_diff if sum(
                    [veh.route == 'M1', rootVehL.route == 'M1']) == 1 else self.dt_safe_same
                desired_tf = rootVehL.long_tfChosen + (rootVehL.sizeP - 1) * rootVehL_dt_inner + dt_safe
            else:  # order > 1
                veh_parent = self.vehs_All[node.parent.veh_name]
                parent_dt_inner = veh_parent.CFM.get_dt_inner(velocity=Vehicle.v_des['M1'])  # dt_inner: v_des_M1
                dt_safe = self.dt_safe_diff if sum(
                    [veh.route == 'M1', veh_parent.route == 'M1']) == 1 else self.dt_safe_same
                desired_tf = node.parent.tfChosen + (veh_parent.sizeP - 1) * parent_dt_inner + dt_safe

            """ calculate node loss and prune """
            if desired_tf <= veh_statePhase.tf_Range[1]:
                node.tfChosen = max(desired_tf, veh_statePhase.tf_Range[0])
                alpha = alpha_dict[veh.route]
                node.j = (node.tfChosen - veh_statePhase.tf_Range[0]) * alpha * veh.sizeP
                node.J1 = node.j + node.parent.J1
                if node.order != self.vehsNum_All and node.J1 > self.optimal_J1[0]:
                    node.FLAG_prunedBelow = True
                    return  # pruning rule 2
            else:
                node.tfChosen = np.inf
                node.j, node.J1 = np.inf, np.inf
                node.FLAG_prunedBelow = True
                return  # pruning rule 1

        """ update the optimal sequence """
        if node.order == self.vehsNum_All and node.J1 < self.optimal_J1[0]:
            self.optimal_J1 = [node.J1, node.id_inLayer]

        """ iteration, recursion """
        if node.child_M1:
            self.__searchNode_next(node.child_M1)
        if node.child_R0:
            self.__searchNode_next(node.child_R0)

    def showTree(self, ax=None, fig_size=(10, 4), FLAG_figSave=False, FLAG_J=True, FLAG_showPruned=True):
        self.FLAG_figSave = FLAG_figSave

        if ax is None:
            """ new ax_road on the self.fig_tree """
            if not self.fig_tree:
                self.fig_tree = plt.figure()
                self.fig_tree.set_size_inches(fig_size)
                ax = self.fig_tree.add_subplot(label='tree_R0M1')  # only create ax_road

        ax.tick_params(axis='x', labelsize=11)
        ax.tick_params(axis='y', labelsize=11)
        ax.set_xlabel('(b) Lane M1: cases', fontsize=12)
        ax.set_ylabel('order', fontsize=12)

        # set x_ticks
        xtick_interval = np.ceil(self.seqCases_num / 5)
        xticks = np.arange(0, self.seqCases_num, xtick_interval)
        if xticks[-1] != self.seqCases_num - 1:
            xticks = np.append(xticks, self.seqCases_num - 1)
        xticklabels = [int(tick + 1) for tick in xticks]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)

        # set y_ticks
        orderID_root = self.network.rootVehL_M1.orderID_M1 if self.network.rootVehL_M1 else 0  # add, 2023-01-08
        ax.set_yticks(-np.array(range(self.vehsNum_All + 1)))
        ax.set_yticklabels(np.array(range(self.vehsNum_All + 1)) + orderID_root)

        """ show nodes layer by layer """
        root = self.treeNodes[0][0]
        ax.scatter(root.caseMean, 0, marker=root.marker, fc=root.color, ec='k', s=root.markerSize)

        for layer in self.treeNodes[1:]:
            for node in layer.values():
                if self.vehsNum_All <= 6:
                    node.markerSize += 15

                ''' show nodes that searched '''
                if node.FLAG_searched:
                    ax.scatter(node.caseMean, -node.order,
                               marker=node.marker, fc=node.color, ec=node.color, s=node.markerSize)
                    ax.plot([node.caseMean, node.parent.caseMean], [-node.order, -node.order + 1], 'gray', ls='-',
                            zorder=0, alpha=.6)

                ''' show nodes that pruned below '''
                if node.FLAG_prunedBelow and node.order != self.vehsNum_All:
                    for child in [node.child_M1, node.child_R0]:
                        if child:
                            ax.scatter(child.caseMean, -child.order,
                                       marker=node.marker, fc='gray', ec='gray', s=node.markerSize)
                            ax.plot([child.caseMean, child.parent.caseMean], [-child.order, -child.order + 1],
                                    'gray', ls='--', zorder=0, alpha=0.2)
                            ax.scatter(np.mean([node.caseMean, child.caseMean]), -node.order - 0.5,
                                       marker='x', c='k', s=20)

                ''' show nodes that not searched if FLAG_showPruned==True '''
                if FLAG_showPruned and not node.FLAG_searched and not node.parent.FLAG_prunedBelow:
                    ax.scatter(node.caseMean, -node.order, marker=node.marker, fc='gray', ec='gray', s=node.markerSize)
                    ax.plot([node.caseMean, node.parent.caseMean], [-node.order, -node.order + 1], 'gray', ls='--',
                            zorder=0, alpha=.2)

        """ show optimal search-path """
        if self.optimal_J1[1] is not None:
            temp_node = self.treeNodes[self.vehsNum_All][self.optimal_J1[1]]
            while temp_node.parent:
                ''' plot optimal sequence and arrow '''
                ax.plot([temp_node.caseMean, temp_node.parent.caseMean], [-temp_node.order, -temp_node.parent.order],
                        c='k', lw=2.2, zorder=0)
                # arrow
                center = ((temp_node.caseMean + temp_node.parent.caseMean) / 2, (-temp_node.order * 2 + 1) / 2)
                angle = [temp_node.caseMean - temp_node.parent.caseMean, -1]
                angle = np.array(angle) / np.linalg.norm(angle) / 10  # normalize
                ax.annotate('', xytext=center, xy=(center[0] + angle[0], center[1] + angle[1]),
                            arrowprops={'headwidth': 5, 'headlength': 5, 'fc': 'k', 'ec': 'k'})

                temp_node = temp_node.parent

            ''' text the optimal J '''
            if FLAG_J:
                ax.text(.75 * self.seqCases_num, -self.vehsNum_All + .3,
                        r'$J_1^*=%.1f~\mathrm{s}$' % self.optimal_J1[0],
                        size=13, bbox={'fc': 'w', 'ec': 'w', 'alpha': 0.8})

        plt.tight_layout()

    def get_optimalSequence(self):
        if self.optimal_J1[1] is not None:  # found the optimal sequence
            ''' get the optimal sequence in Lane 1 '''
            last_node = self.treeNodes[-1][self.optimal_J1[1]]
            while last_node.parent:
                self.optSeq_Lane1_nodes = [last_node] + self.optSeq_Lane1_nodes
                self.optSeq_Lane1_nameL = [last_node.veh_name] + self.optSeq_Lane1_nameL
                last_node = last_node.parent

            self.optSeq_pos_R0 = [self.optSeq_Lane1_nameL.index(veh_name) for veh_name in self.optSeq_Lane1_nameL
                                  if veh_name in self.vehsL_R0]
        elif self.treeNodes is None:  # add, 2023-02-21
            pass
        else:
            print(f'Warning: no solution when get_optimalSequence at schedule_R0M1')
            print()
            return None, None, None     # add, 23-07-05
            # sys.exit()      # TODO: adjust here

        return self.optSeq_Lane1_nodes, self.optSeq_Lane1_nameL, self.optSeq_pos_R0

    def get_trajectories(self):
        """ save trajectories in self.vehs_long_trajPlan """
        if not self.optSeq_Lane1_nodes:
            self.get_optimalSequence()

        for node in self.optSeq_Lane1_nodes:
            self.vehs_long_tfChosen[node.veh_name] = node.tfChosen
            self.vehs_long_trajPlan[node.veh_name] = self.network.statePhases_All[node.veh_name].get_trajectory_withPF(
                node.tfChosen, overlap_FLAG=node.overlap_FLAG)

            if self.vehs_All[node.veh_name].followers:
                self.add_followerTrajectories(self.vehs_long_trajPlan, self.vehs_All, node.veh_name,
                                              v_des=Vehicle.v_des['M1'])

    @staticmethod
    def add_followerTrajectories(trajs, vehs_all, name_L, v_des):
        leader_traj = trajs[name_L]

        """ extend leader trajectory """
        t, planned_a, planned_v, planned_p = zip(*leader_traj)
        t_ex = 2 * (vehs_all[name_L].sizeP - 1)  # 2s for the inner dt

        t_ = np.arange(1, t_ex * 10 + 1) / 10
        planned_a = np.append(planned_a, np.zeros(np.shape(t_)))
        planned_v = np.append(planned_v, np.ones(np.shape(t_)) * v_des)
        planned_p = np.append(planned_p, planned_p[-1] + t_ * v_des)
        t = np.append(t, t[-1] + t_)

        leader_traj = list(zip(t, planned_a, planned_v, planned_p))

        """ add trajectory for each follower """
        for veh_F in vehs_all[name_L].followers:
            front_traj = leader_traj if veh_F.front.name == name_L else trajs[veh_F.front.name]
            t, a_front, v_front, p_front = zip(*front_traj)
            veh_stateNow = [veh_F.time_c, veh_F.long_a, veh_F.long_v, veh_F.long_p]
            front_stateNow = [veh_F.front.time_c, veh_F.front.long_a, veh_F.front.long_v, veh_F.front.long_p]
            leader_stateNow = [veh_F.leader.time_c, veh_F.leader.long_a, veh_F.leader.long_v, veh_F.leader.long_p]

            a_follower = [veh_F.CFM.get_next_a(veh_stateNow, front_stateNow, leader_stateNow)]
            v_follower = [veh_F.long_v + a_follower[-1] * 0.1]
            p_follower = [veh_F.long_p + np.mean([veh_F.long_v, v_follower[-1]]) * 0.1]

            for i in range(len(front_traj) - 1):
                a_ = veh_F.CFM.get_next_a([t[i], a_follower[-1], v_follower[-1], p_follower[-1]],
                                          front_traj[i], leader_traj[i])
                v_ = v_follower[-1] + a_ * 0.1
                p_ = p_follower[-1] + (v_ + v_follower[-1]) / 2 * 0.1
                a_ = 0 if abs(a_) < 1e-4 else a_
                a_follower.append(a_)
                v_follower.append(v_)
                p_follower.append(p_)

            trajs[veh_F.name] = list(zip(t, a_follower, v_follower, p_follower))


class Schedule_M1M2:
    """ load constant parameters """
    dt_safe_same = dt_safe_same
    dt_safe_diff = dt_safe_diff + 0.6  # keep LC enough safe
    alpha_dict = alpha_dict
    alpha_M1_LC = alpha_M1_LC

    def __init__(self, network, LC_vehsName_M1=[], s_01=None):
        self.network = network
        self.s_01 = s_01

        """ load vehicle-groups information """
        self.vehs_All = network.vehs_All
        self.vehsL_M1 = network.vehsL_M1
        self.vehsL_M2 = network.vehsL_M2

        self.vehsL_M1_LC = LC_vehsName_M1

        self.vehsNum_M1_LC = len(LC_vehsName_M1)  # only leader
        self.vehsNum_M2 = len(self.vehsL_M2)
        self.vehsNum_All = self.vehsNum_M1_LC + self.vehsNum_M2

        """ get sequences_dict and dividers_dict && generate all tree nodes with link """
        self.seqCases_num = factorial(self.vehsNum_All) // (factorial(self.vehsNum_M1_LC) * factorial(self.vehsNum_M2))
        self.sequences_All, self.dividers_List = self.__get_all_seqCases()
        self.treeNodes = self.__generate_treeNodes() if self.vehsNum_All else None  # add, 2023-02-22

        """ sequence tree M1_LC-M2 """
        self.fig_tree = None
        self.optimal_J2 = [np.inf, None]  # [value, case_id]

        self.optSeq_Lane2_nodes = []  # [treeNode]: list
        self.optSeq_Lane2_nameL = []  # [veh_name]: list
        self.optSeq_pos_M1_LC = []

        self.vehs_long_tfChosen = {}  # {veh_name: tfChosen}
        self.vehs_long_trajPlan = {}  # {veh_name: trajPlan}

        """ global FLAGs about ***pause&save*** figures """
        self.FLAG_figPause, self.t_figPause = False, 0.1
        self.FLAG_figSave = False

    def __get_all_seqCases(self):
        sequences_All, dividers_List = [], []
        """ first, get all sequences """
        string = [i for i in range(self.vehsNum_All)]
        pos_M1_LC_Dict = list(combinations(string, self.vehsNum_M1_LC))

        for pos_M1_LC in pos_M1_LC_Dict:
            sequence = np.empty(self.vehsNum_All, dtype=object)
            sequence[list(pos_M1_LC)] = self.vehsL_M1_LC
            sequence[list(set(string) - set(pos_M1_LC))] = self.vehsL_M2
            sequences_All.append(sequence)
        sequences_All = np.array(sequences_All)[::-1]  # when searching: M2 first, M1_LC then

        """ then, get the dividers """
        dividers_former = [0]
        for order in range(self.vehsNum_All):
            dividers = []
            for seqCase_id in range(1, self.seqCases_num):
                if sum([sequences_All[seqCase_id - 1][order] in self.vehsL_M2,
                        sequences_All[seqCase_id][order] in self.vehsL_M2]) == 1:
                    dividers.append(seqCase_id)
            dividers = list(set(dividers).union(set(dividers_former)))
            dividers.sort()  # bug, here
            dividers_former = dividers
            dividers_List.append(dividers)

        return sequences_All, dividers_List

    def __generate_treeNodes(self):
        treeNodes = [{} for _ in range(self.vehsNum_All + 1)]  # first: root
        treeNodes[0][0] = TreeNode(0, 0, 'root', caseRange=[0, self.seqCases_num - 1],
                                   caseMean=(self.seqCases_num + 2 * self.dividers_List[0][-1] - 2) / 4)

        """ generate nodes layer by layer, width-first """
        for order in range(self.vehsNum_All):
            dividers = self.dividers_List[order]
            for divider in dividers:
                id_inLayer = dividers.index(divider)
                if divider != dividers[-1]:
                    caseRange = [divider, dividers[id_inLayer + 1] - 1]
                else:
                    caseRange = [divider, self.seqCases_num - 1]
                veh_name = self.sequences_All[divider][order]
                treeNode = TreeNode(order + 1, id_inLayer, veh_name, caseRange)  # note here: order + 1
                treeNodes[order + 1][id_inLayer] = treeNode

                """ create link between nodes """
                if order == 0:
                    parent_idInLayer = 0
                else:
                    parent_idInLayer = np.where(np.array(self.dividers_List[order - 1]) <= divider)[0][-1]
                treeNode.parent = treeNodes[order][parent_idInLayer]
                if veh_name in self.vehsL_M1_LC:
                    treeNode.parent.child_M1 = treeNode
                else:
                    treeNode.parent.child_M2 = treeNode

        return treeNodes

    def searchTree_depth(self, FLAG_figPause=False, FLAG_figSave=False):
        """ set global FLAGs """
        self.FLAG_figPause = FLAG_figPause
        self.FLAG_figSave = FLAG_figSave

        """ recursive search """
        if self.treeNodes:
            self.__searchNode_next(self.treeNodes[0][0])  # search from nodes

    def __searchNode_next(self, node):  # TODO, most importantly in this function
        if node.veh_name == 'root':
            pass
        else:
            node.FLAG_searched = True
            veh = self.vehs_All[node.veh_name]
            veh_statePhase = self.network.statePhases_All[veh.name] if veh.name not in self.vehsL_M1_LC \
                else self.network.statePhases_LC[veh.name]  # change, 2022-12-29

            """ decide desired_tf for each node """
            if node.parent.veh_name == 'root' and self.network.rootVehL_M2 is None:
                desired_tf = max(veh.long_tfChosen, veh_statePhase.tf_Range[0])
            elif node.parent.veh_name == 'root' and self.network.rootVehL_M2 is not None:  # add, 2022-12-29
                rootVehL = self.network.rootVehL_M2
                rootVehL_dt_inner = rootVehL.CFM.get_dt_inner(velocity=Vehicle.v_des['M2'])
                dt_safe = self.dt_safe_diff if sum(
                    [veh.route == 'M2', rootVehL.route == 'M2']) == 1 else self.dt_safe_same
                desired_tf = rootVehL.long_tfChosen + (rootVehL.sizeP - 1) * rootVehL_dt_inner + dt_safe
            else:  # order > 1
                veh_parent = self.vehs_All[node.parent.veh_name]
                parent_dt_inner = veh_parent.CFM.get_dt_inner(velocity=Vehicle.v_des['M2'])  # dt_inner: v_des_M2
                dt_safe = self.dt_safe_diff if sum(
                    [veh.route == 'M2', veh_parent.route == 'M2']) == 1 else self.dt_safe_same
                ref_tf = node.parent.tfChosen + (veh_parent.sizeP - 1) * parent_dt_inner + dt_safe

                """ calculate desired_tf for LC vehs """
                if veh.route == 'M2' and veh_parent.route == 'M2':
                    desired_tf = ref_tf
                else:  # at least one vehicle on Lane M1
                    ''' get the parent vehicle-group trajectory '''
                    parent_statePhase = self.network.statePhases_LC[veh_parent.name] if \
                        veh_parent.name in self.vehsL_M1_LC else self.network.statePhases_All[veh_parent.name]
                    self.vehs_long_trajPlan[veh_parent.name] = parent_statePhase.get_trajectory_withPF(
                        node.parent.tfChosen)
                    self.add_followerTrajectories(self.vehs_long_trajPlan, self.vehs_All, veh_parent.name,
                                                  v_des=Vehicle.v_des['M2'])
                    vehParent_traj = self.vehs_long_trajPlan[veh_parent.followers[-1].name] if veh_parent.followers \
                        else self.vehs_long_trajPlan[veh_parent.name]

                    ''' get the front and back vehicle-group trajectory '''
                    if veh.route == 'M1':  # veh_parent = 'M1' or 'M2'
                        vehsName_M1 = [veh.name for veh in self.vehs_All.values() if
                                       veh.name[-1] == 'L' and veh.route == 'M1']
                        veh_index = vehsName_M1.index(veh.name)

                        ''' get the front vehicle-group trajectory '''
                        if veh_index > 0:
                            vehName_front = vehsName_M1[veh_index - 1]
                            if vehName_front in self.vehsL_M1_LC:  # M1_LC -> M1_LC, parent = front
                                vehFormer_traj = vehParent_traj  # parent, already the former
                            else:  # M2 -> M1_LC
                                vehFront = self.vehs_All[vehName_front]
                                vehName_former = vehFront.followers[-1].name if vehFront.followers else vehFront.name
                                if vehName_former in self.s_01.vehs_long_trajPlan:  # debug, 23-07-06
                                    vehFormer_traj = self.s_01.vehs_long_trajPlan[vehName_former]
                                else:
                                    vehFormer_traj = None
                        else:
                            vehFormer_traj = None

                        ''' get the back vehicle-group trajectory '''
                        if veh_index < len(vehsName_M1) - 1:
                            vehName_Back = vehsName_M1[veh_index + 1]
                            if vehName_Back in self.vehsL_M1_LC:
                                vehBack_traj = None
                            else:
                                if vehName_Back in self.s_01.vehs_long_trajPlan:    # debug, 23-07-06
                                    vehBack_traj = self.s_01.vehs_long_trajPlan[vehName_Back]  # the leader
                                else:
                                    vehBack_traj = None
                        else:
                            vehBack_traj = None

                    else:  # veh.route = 'M2' and veh_parent.route = 'M1'
                        vehFormer_traj, vehBack_traj = None, None

                    ''' already get vehParent & vehFront & vehBack trajectories, calculate the desired'''
                    sp_safeLC = StatePhase_safeLC(veh, veh_statePhase,
                                                  vehParent_traj, vehFormer_traj, vehBack_traj,
                                                  ref_tf=ref_tf)
                    desired_tf, node.overlap_FLAG, node.LC_midT_optional = sp_safeLC.get_desiredTF_andMidT()
                    if not node.LC_midT_optional:  # prune in advance
                        node.tfChosen, node.j, node.J2 = np.inf, np.inf, np.inf
                        node.FLAG_prunedBelow = True
                        return  # no solution, pruning rule 1

            """ calculate node loss and prune """
            if desired_tf <= veh_statePhase.tf_Range[1]:
                node.tfChosen = max(desired_tf, veh_statePhase.tf_Range[0])
                alpha = alpha_dict['M2'] if veh.route == 'M2' else self.alpha_M1_LC
                node.j = (node.tfChosen - veh_statePhase.tf_Range[0]) * alpha * veh.sizeP
                node.J2 = node.j + node.parent.J2
                if node.order != self.vehsNum_All and node.J2 > self.optimal_J2[0]:
                    node.FLAG_prunedBelow = True
                    return  # pruning rule 2
            else:
                node.tfChosen = np.inf
                node.j, node.J2 = np.inf, np.inf
                node.FLAG_prunedBelow = True
                return  # pruning rule 1

        """ update the optimal sequence """
        if node.order == self.vehsNum_All and node.J2 < self.optimal_J2[0]:
            self.optimal_J2 = [node.J2, node.id_inLayer]

        """ iteration, recursion """
        if node.child_M2:
            self.__searchNode_next(node.child_M2)
        if node.child_M1:
            self.__searchNode_next(node.child_M1)

    def showTree(self, ax=None, fig_size=(6, 3), FLAG_figSave=False, FLAG_J=True, FLAG_showPruned=True):
        self.FLAG_figSave = FLAG_figSave

        if ax is None:
            """ new ax_road on the self.fig_tree """
            if not self.fig_tree:
                self.fig_tree = plt.figure()
                self.fig_tree.set_size_inches(fig_size)
                ax = self.fig_tree.add_subplot(label='tree_M1M2')  # only create ax_road

        ax.tick_params(axis='x', labelsize=11)
        ax.tick_params(axis='y', labelsize=11)
        ax.set_xlabel('(a) Lane M2: cases', fontsize=12)
        ax.set_ylabel('order', fontsize=12)

        # set x_ticks
        xtick_interval = np.ceil(self.seqCases_num / 5)
        xticks = np.arange(0, self.seqCases_num, xtick_interval)
        if xticks[-1] != self.seqCases_num - 1:
            xticks = np.append(xticks, self.seqCases_num - 1)
        xticklabels = [int(tick + 1) for tick in xticks]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)

        # set y_ticks
        orderID_root = self.network.rootVehL_M2.orderID_M2 if self.network.rootVehL_M2 else 0  # add, 2023-01-08
        ax.set_yticks(-np.array(range(self.vehsNum_All + 1)))
        ax.set_yticklabels(np.array(range(self.vehsNum_All + 1)) + orderID_root)

        """ show nodes layer by layer """
        root = self.treeNodes[0][0]
        ax.scatter(root.caseMean, 0, marker=root.marker, fc=root.color, ec='k', s=root.markerSize)

        for layer in self.treeNodes[1:]:
            for node in layer.values():
                if self.vehsNum_All <= 6:
                    node.markerSize += 15

                ''' show nodes that searched '''
                if node.FLAG_searched:
                    ax.scatter(node.caseMean, -node.order,
                               marker=node.marker, fc=node.color, ec=node.color, s=node.markerSize)
                    ax.plot([node.caseMean, node.parent.caseMean], [-node.order, -node.order + 1], 'gray', ls='-',
                            zorder=0, alpha=.6)

                ''' show nodes that pruned below '''
                if node.FLAG_prunedBelow and node.order != self.vehsNum_All:
                    for child in [node.child_M2, node.child_M1]:
                        if child:
                            ax.scatter(child.caseMean, -child.order,
                                       marker=node.marker, fc='gray', ec='gray', s=node.markerSize)
                            ax.plot([child.caseMean, child.parent.caseMean], [-child.order, -child.order + 1],
                                    'gray', ls='--', zorder=0, alpha=0.2)
                            ax.scatter(np.mean([node.caseMean, child.caseMean]), -node.order - 0.5,
                                       marker='x', c='k', s=20)

                ''' show nodes that not searched if FLAG_showPruned==True '''
                if FLAG_showPruned and not node.FLAG_searched and not node.parent.FLAG_prunedBelow:
                    ax.scatter(node.caseMean, -node.order, marker=node.marker, fc='gray', ec='gray', s=node.markerSize)
                    ax.plot([node.caseMean, node.parent.caseMean], [-node.order, -node.order + 1], 'gray', ls='--',
                            zorder=0, alpha=.2)

        """ show optimal search-path """
        if self.optimal_J2[1] is not None:
            temp_node = self.treeNodes[self.vehsNum_All][self.optimal_J2[1]]
            while temp_node.parent:
                ''' plot optimal sequence and arrow '''
                ax.plot([temp_node.caseMean, temp_node.parent.caseMean], [-temp_node.order, -temp_node.parent.order],
                        c='k', lw=2.2, zorder=0)
                # arrow
                center = ((temp_node.caseMean + temp_node.parent.caseMean) / 2, (-temp_node.order * 2 + 1) / 2)
                angle = [temp_node.caseMean - temp_node.parent.caseMean, -1]
                angle = np.array(angle) / np.linalg.norm(angle) / 10  # normalize
                ax.annotate('', xytext=center, xy=(center[0] + angle[0], center[1] + angle[1]),
                            arrowprops={'headwidth': 4.5, 'headlength': 4.5, 'fc': 'k', 'ec': 'k'})

                temp_node = temp_node.parent
            ''' text the optimal J '''
            if FLAG_J:
                ax.text(.75 * self.seqCases_num, -self.vehsNum_All + .3,
                        r'$J_2^*=%.1f~\mathrm{s}$' % self.optimal_J2[0],
                        size=13, bbox={'fc': 'w', 'ec': 'w', 'alpha': 0.8})

        plt.tight_layout()

    def get_optimalSequence(self):
        if self.optimal_J2[1] is not None:  # found the optimal sequence
            ''' get the optimal sequence in Lane 2 '''
            last_node = self.treeNodes[-1][self.optimal_J2[1]]
            while last_node.parent:
                self.optSeq_Lane2_nodes = [last_node] + self.optSeq_Lane2_nodes
                self.optSeq_Lane2_nameL = [last_node.veh_name] + self.optSeq_Lane2_nameL
                last_node = last_node.parent

            self.optSeq_pos_M1_LC = [self.optSeq_Lane2_nameL.index(veh_name) for veh_name in self.optSeq_Lane2_nameL
                                     if veh_name in self.vehsL_M1_LC]
        else:
            self.optimal_J2[0] = np.inf
            return None, None, None

        return self.optSeq_Lane2_nodes, self.optSeq_Lane2_nameL, self.optSeq_pos_M1_LC

    @staticmethod
    def add_followerTrajectories(trajs, vehs_all, name_L, v_des):
        leader_traj = trajs[name_L]

        """ extend leader trajectory """
        t, planned_a, planned_v, planned_p = zip(*leader_traj)
        t_ex = 2 * (vehs_all[name_L].sizeP - 1)  # 2s for the inner dt

        t_ = np.arange(1, t_ex * 10 + 1) / 10
        planned_a = np.append(planned_a, np.zeros(np.shape(t_)))
        planned_v = np.append(planned_v, np.ones(np.shape(t_)) * v_des)
        planned_p = np.append(planned_p, planned_p[-1] + t_ * v_des)
        t = np.append(t, t[-1] + t_)

        leader_traj = list(zip(t, planned_a, planned_v, planned_p))

        """ add trajectory for each follower """
        for veh_F in vehs_all[name_L].followers:
            front_traj = leader_traj if veh_F.front.name == name_L else trajs[veh_F.front.name]
            t, a_front, v_front, p_front = zip(*front_traj)
            veh_stateNow = [veh_F.time_c, veh_F.long_a, veh_F.long_v, veh_F.long_p]
            front_stateNow = [veh_F.front.time_c, veh_F.front.long_a, veh_F.front.long_v, veh_F.front.long_p]
            leader_stateNow = [veh_F.leader.time_c, veh_F.leader.long_a, veh_F.leader.long_v, veh_F.leader.long_p]

            a_follower = [veh_F.CFM.get_next_a(veh_stateNow, front_stateNow, leader_stateNow)]
            v_follower = [veh_F.long_v + a_follower[-1] * 0.1]
            p_follower = [veh_F.long_p + np.mean([veh_F.long_v, v_follower[-1]]) * 0.1]

            for i in range(len(front_traj) - 1):
                a_ = veh_F.CFM.get_next_a([t[i], a_follower[-1], v_follower[-1], p_follower[-1]],
                                          front_traj[i], leader_traj[i])
                v_ = v_follower[-1] + a_ * 0.1
                p_ = p_follower[-1] + (v_ + v_follower[-1]) / 2 * 0.1
                a_ = 0 if abs(a_) < 1e-4 else a_
                a_follower.append(a_)
                v_follower.append(v_)
                p_follower.append(p_)

            trajs[veh_F.name] = list(zip(t, a_follower, v_follower, p_follower))


class Search_DCG:

    def __init__(self, network, LC_vehsName_M1):
        self.network = network
        self.LC_vehsName_M1 = LC_vehsName_M1
        self.LC_vehsPos_M1 = [network.vehsL_M1.index(veh_name) for veh_name in self.LC_vehsName_M1]

        time_0 = time.time()
        """ create scheduler s_01 """
        self.s_01 = Schedule_R0M1(network, LC_vehsName_M1)
        self.s_01.searchTree_depth()
        # self.s_01.showTree(FLAG_showPruned=False)
        self.optSeq_Lane1_nodes, self.optSeq_Lane1_nameL, self.optSeq_pos_R0 = self.s_01.get_optimalSequence()
        if self.optSeq_Lane1_nodes:
            self.s_01.get_trajectories()  # must keep here, for s_12
        self.t_search_01 = time.time() - time_0

        time_1 = time.time()
        """ create scheduler s_12 """
        if self.optSeq_Lane1_nodes:
            self.s_12 = Schedule_M1M2(network, LC_vehsName_M1, self.s_01)
            self.s_12.searchTree_depth()
            # self.s_12.showTree(FLAG_showPruned=True)
            self.optSeq_Lane2_nodes, self.optSeq_Lane2_nameL, self.optSeq_pos_M1_LC = self.s_12.get_optimalSequence()
        else:
            self.s_12 = None
            self.optSeq_Lane2_nodes, self.optSeq_Lane2_nameL, self.optSeq_pos_M1_LC = None, None, None
        self.t_search_12 = time.time() - time_1

        """ search result """
        if self.optSeq_Lane1_nodes:
            self.J_total = self.s_01.optimal_J1[0] + self.s_12.optimal_J2[0]
        else:
            self.J_total = np.inf
        self.t_search_all = self.t_search_01 + self.t_search_12

        """ figs, vehsNetwork & sequence tree """
        self.fig_vehsNetwork = None
        self.fig_sequenceTree = None

    def get_optimalTrajectories(self):
        """ get and assign the optimal trajectories for each vehicle """
        if self.optSeq_Lane1_nodes:     # add, 2023-02-22
            for node in self.optSeq_Lane1_nodes:
                veh = self.network.vehs_All[node.veh_name]
                veh.long_tfChosen = node.tfChosen
                veh.long_statePhase = self.network.statePhases_All[veh.name]
                veh.long_trajPlan = self.s_01.vehs_long_trajPlan[veh.name]
                veh.get_followerTrajectories(v_des=Vehicle.v_des['M1'])

                ''' plan LC trajectory for Ramp vehicles '''  # TODO, lateral LC planning here
                if veh.name[0] == 'R':
                    veh.get_LC_Trajectories(lat_p0=-0.5 * Road.LANE_WIDTH, lat_pf=0.5 * Road.LANE_WIDTH,
                                            LC_duration=4, t0_LC='default')

        if self.optSeq_Lane2_nodes:
            for node in self.optSeq_Lane2_nodes:
                veh = self.network.vehs_All[node.veh_name]
                veh.long_statePhase = self.network.statePhases_All[veh.name] if veh.route == 'M2' else \
                    self.network.statePhases_LC[veh.name]

                veh.long_tfChosen = node.tfChosen
                veh.long_trajPlan = veh.long_statePhase.get_trajectory_withPF(node.tfChosen, overlap_FLAG=node.overlap_FLAG)
                veh.get_followerTrajectories(v_des=Vehicle.v_des['M2'])

                ''' plan LC trajectory for M1_LC vehicles '''
                if veh.name in self.LC_vehsName_M1:
                    veh.route = 'M2'
                    veh.v_des, veh.v_min = Vehicle.v_des['M2'], Vehicle.v_min['M2']  # debug, 2022-01-02
                    # veh.LC_midT = random.choice(node.LC_midT_optional)  # random LC point
                    # relative fixed LC point
                    if node.LC_midT_optional:  # debug here, 2023-01-29
                        LC_index = int(len(node.LC_midT_optional) / 5)
                        veh.LC_midT = node.LC_midT_optional[LC_index]

                        veh.get_LC_Trajectories(lat_p0=0.5 * Road.LANE_WIDTH, lat_pf=1.5 * Road.LANE_WIDTH,
                                                LC_duration=Vehicle.LC_duration,
                                                t0_LC=veh.LC_midT - Vehicle.LC_duration / 2)

    def show_vehsNetwork(self, fig_network=None, fig_size=(5, 3), FLAG_J=True):
        if self.J_total is not np.inf:
            if not fig_network:
                if not self.fig_vehsNetwork:
                    self.fig_vehsNetwork = plt.figure(figsize=fig_size)
                fig_network = self.fig_vehsNetwork

            opt_J1 = self.s_01.optimal_J1[0] if FLAG_J else None
            opt_J2 = self.s_12.optimal_J2[0] if FLAG_J else None

            self.network.showDCG_subs(fig_subs=fig_network, LC_vehsPos_M1=self.LC_vehsPos_M1)
            self.network.addSequences_subsDCG(pos_R0=self.optSeq_pos_R0, opt_J1=opt_J1,
                                              pos_M1_LC=self.optSeq_pos_M1_LC, opt_J2=opt_J2)

    def show_sequenceTree(self, fig_tree=None, fig_size=(5.5, 5.5), FLAG_J=True):
        if not fig_tree:
            if not self.fig_sequenceTree:
                self.fig_sequenceTree = plt.figure(figsize=fig_size)
            fig_tree = self.fig_sequenceTree

        for ax in fig_tree.get_axes():
            ax.remove()

        ax_tree_R0M1 = fig_tree.add_subplot(2, 1, 2, label='tree_R0M1')
        ax_tree_M1M2 = fig_tree.add_subplot(2, 1, 1, label='tree_M1M2')

        self.s_01.showTree(ax_tree_R0M1, FLAG_showPruned=False, FLAG_J=FLAG_J)
        self.s_12.showTree(ax_tree_M1M2, FLAG_showPruned=False, FLAG_J=FLAG_J)
        plt.tight_layout()


if __name__ == "__main__":
    from cloudController.vehsNetwork import defaultNetwork_demo
    from utilities.plotResults import show_trajs_threeLanes

    network = defaultNetwork_demo()
    s = Search_DCG(network, network.LC_M1_cases[2][0])  # adjust LC_vehsName_M1 here
    s.show_vehsNetwork()
    s.show_sequenceTree()
    s.get_optimalTrajectories()

    show_trajs_threeLanes(network.vehs_All)

    plt.show()
