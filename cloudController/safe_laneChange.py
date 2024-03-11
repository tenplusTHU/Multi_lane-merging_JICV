"""
    Description: safe lane-change, decide the desired_tf
    Author: Tenplus
    Create-time: 2022-12-29
"""

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from objects.Vehicle import Vehicle
from objects.Road import Road
from cloudController.statePhase import Long_StatePhase
from utilities.plotResults import show_severalTrajs


class StatePhase_safeLC:
    """ for vehs_M1_LC """
    LC_duration = Vehicle.LC_duration
    safeGap = 1 * Vehicle.v_des['M2']  # 25 m

    def __init__(self, veh: Vehicle, veh_statePhase,
                 vehParent_traj, vehFormer_traj, vehBack_traj, safeGap=safeGap, ref_tf=None):
        self.veh = veh
        self.veh_sp = veh_statePhase

        self.vehParent_traj = vehParent_traj
        self.vehFormer_traj = vehFormer_traj
        self.vehBack_traj = vehBack_traj

        self.safeGap = safeGap
        self.ref_tf = ref_tf if ref_tf else self.veh_sp.tf_Range[0]  # add 2022-12-29, to accelerate calculating

        ''' outputs '''
        self.region = None
        self.desired_tf, self.overlap_FLAG, self.LC_midT = None, 'sweet', None

        self.FLAG_print = False

    def get_desiredTF_andMidT(self):
        sp = self.veh_sp

        """ 1. check the pole first """
        if self.ref_tf <= sp.tf_Range[0]:
            LC_midT_lapped = self.__spTF_to_lappedMidT(sp.tf_Range[0], 'pole')  # parent & front
            if LC_midT_lapped:
                if self.FLAG_print:
                    print(f'** pole has solution ** LC_midT: {LC_midT_lapped}')
                self.region, self.desired_tf, self.LC_midT = 'pole', sp.tf_Range[0], LC_midT_lapped
                return self.desired_tf, self.overlap_FLAG, self.LC_midT

        """ 2. check opt_tf then """
        if self.ref_tf <= sp.adjustD_tf[-1]:
            # check the right first
            LC_midT_lapped_end = self.__spTF_to_lappedMidT(sp.adjustD_tf[-1], 'sweet')
            if LC_midT_lapped_end:
                spTF_range = [max(self.ref_tf, sp.tf_Range[0]), sp.adjustD_tf[-1]]
                self.desired_tf, self.LC_midT = self.__get_opt_spTF_feasible('opt_tf', spTF_range)
                self.region = 'opt_tf'
                if self.FLAG_print:
                    print(f'** opt_tf has solution ** tf: {self.desired_tf}, LC_midT: {self.LC_midT}')
                return self.desired_tf, self.overlap_FLAG, self.LC_midT
            else:
                pass  # no solution in opt_tf

        """ 3. check opt_pf next """
        if self.ref_tf <= sp.boundP_tf[-1]:
            # get the spTF_backEnd, and check feasible
            spTF_backEnd = self.__get_max_spTF_with_vehBack()
            if spTF_backEnd > max(self.ref_tf, sp.tf_Range[0]):
                LC_midT_lapped_end = self.__spTF_to_lappedMidT(spTF_backEnd, 'normal')
                if LC_midT_lapped_end:
                    spTF_range = [max(self.ref_tf, sp.tf_Range[0]), spTF_backEnd]
                    self.desired_tf, self.LC_midT = self.__get_opt_spTF_feasible('opt_pf', spTF_range)

                    if self.LC_midT:    # add, 2023-02-19
                        self.region = 'opt_pf'
                        if self.FLAG_print:
                            print(f'** opt_pf has solution ** tf: {self.desired_tf}, LC_midT: {self.LC_midT}')
                        return self.desired_tf, self.overlap_FLAG, self.LC_midT

        """ other case """
        return None, None, None

    def __spTF_to_lappedMidT(self, tf, overlap_FLAG):
        """ return the lapped midT between vehParent and vehFront """  # modified, 2023-02-17
        _, LC_midT = self.__inputTF_to_gapsAndMidT(tf, overlap_FLAG, None)      # no vehParent or vehFront

        if self.vehParent_traj:
            _, LC_midT_parent = self.__inputTF_to_gapsAndMidT(tf, overlap_FLAG, 'parent')
        else:
            LC_midT_parent = LC_midT

        if self.vehFormer_traj:
            _, LC_midT_front = self.__inputTF_to_gapsAndMidT(tf, overlap_FLAG, 'front')
        else:
            LC_midT_front = LC_midT

        ''' get midT intersection '''
        LC_midT_lapped = [midT for midT in LC_midT if midT in LC_midT_parent and midT in LC_midT_front]

        # print for debugging
        # print(f'LC_midT: {LC_midT}')
        # print(f'LC_midT_parent: {LC_midT_parent}')
        # print(f'LC_midT_front: {LC_midT_front}')
        # print(f'LC_midT_lapped: {LC_midT_lapped}')
        return LC_midT_lapped

    def __inputTF_to_gapsAndMidT(self, tf, overlap_FLAG, other_pos):
        sp = self.veh_sp
        self_traj = sp.get_trajectory_withPF(tf=tf, pf=Road.MERGE_ZONE_LENGTH, overlap_FLAG=overlap_FLAG)

        ''' the possible LC midT list '''
        LC_midT_maxRange = [self.veh.time_c + self.LC_duration / 2, tf - self.LC_duration / 2]  # widest
        LC_midT_optional = np.arange(np.ceil(LC_midT_maxRange[0] * 10), LC_midT_maxRange[1] * 10, 1) / 10
        LC_midT_optional = list(LC_midT_optional)

        ''' calculate gaps and check LC_midT for case other_pos '''
        # case 0
        if other_pos is None:
            return None, LC_midT_optional

        # case 1
        if other_pos == 'parent':
            Gaps = self.trajs_toGaps(self_traj, self.vehParent_traj, 'parent')

            for Gap in Gaps[::-1]:  # check from right to left
                if Gap[0] > LC_midT_optional[-1] and Gap[1] < self.safeGap:
                    LC_midT_optional = []  # no solution
                    break
                if Gap[0] < LC_midT_optional[-1]:  # time-saving
                    break
            for LC_midT in LC_midT_optional[::-1]:
                if np.interp(LC_midT, Gaps[:, 0], Gaps[:, 1]) < self.safeGap:
                    index = LC_midT_optional.index(LC_midT)
                    del LC_midT_optional[:index + 1]  # delete the left part
                    break
        # case 2 & case 3
        if other_pos in ['front', 'back']:
            if other_pos == 'front':
                Gaps = self.trajs_toGaps(self_traj, self.vehFormer_traj, 'front')
            else:
                self.veh.long_trajPlan = self_traj
                self.veh.get_followerTrajectories(v_des=Vehicle.v_des['M2'])
                selfTail_traj = self.veh.followers[-1].long_trajPlan if self.veh.followers else self_traj  # 2022-12-31
                Gaps = self.trajs_toGaps(selfTail_traj, self.vehBack_traj, 'back')

            for Gap in Gaps:
                if Gap[0] < LC_midT_optional[0] and Gap[1] < self.safeGap:
                    LC_midT_optional = []  # no solution
                    break
                if Gap[0] > LC_midT_optional[0]:  # time-saving
                    break
            for LC_midT in LC_midT_optional:
                if np.interp(LC_midT, Gaps[:, 0], Gaps[:, 1]) < self.safeGap:
                    index = LC_midT_optional.index(LC_midT)
                    del LC_midT_optional[index:]  # delete the right part
                    break

        """ debug here """
        # plt.figure()
        # plt.plot(Gaps[:, 0], Gaps[:, 1], 'b', lw=2)
        # show_severalTrajs({'g': self.vehFormer_traj, 'k': self_traj})
        # print(f'** Debug, LC_midT_optional: {LC_midT_optional}')
        # plt.show()

        return Gaps, LC_midT_optional

    def __get_opt_spTF_feasible(self, sp_region, spTF_range):
        """ Dichotomy when solution exist """
        sp = self.veh_sp

        """ check the left first """
        midT_lapped_left = self.__spTF_to_lappedMidT(spTF_range[0], sp_region)
        if midT_lapped_left:  # solution at the left-bound
            return spTF_range[0], midT_lapped_left
        else:
            # pinch method
            while spTF_range[1] - spTF_range[0] >= 0.1:
                spTF_middle = np.mean(spTF_range)
                midT_lapped_middle = self.__spTF_to_lappedMidT(spTF_middle, sp_region)  # vehParent & vehFront
                if midT_lapped_middle:
                    spTF_range[1] = spTF_middle
                else:
                    spTF_range[0] = spTF_middle

                if self.FLAG_print:
                    print(f'spTF_range at {sp_region}: {spTF_range}')

            """ get the result """
            LC_midT_lapped = self.__spTF_to_lappedMidT(spTF_range[1], sp_region)
            # print(f'optimal spTF in {sp_region}, tf: {spTF_range[1]}, LC_midT: {LC_midT_lapped}')
            return spTF_range[1], LC_midT_lapped

    def __get_max_spTF_with_vehBack(self):
        """ get the available spTF range in opt_pf """
        sp = self.veh_sp
        spTF_range = [max(self.ref_tf, sp.tf_Range[0]), sp.boundP_tf[-1]]

        """ if vehBack_traj do not exist """
        if not self.vehBack_traj:
            return spTF_range[1]

        """ check the bounds of spTF_range """
        GapsBack_left, _ = self.__inputTF_to_gapsAndMidT(spTF_range[0], 'normal', 'back')
        GapsBack_right, _ = self.__inputTF_to_gapsAndMidT(spTF_range[1], 'normal', 'back')

        if min(GapsBack_left[:, 1]) < self.safeGap:
            return -np.inf  # debug, 2023-01-02
        if min(GapsBack_right[:, 1]) > self.safeGap:
            return spTF_range[-1]  # all the range goes well

        ''' else, Pinch method '''
        while spTF_range[1] - spTF_range[0] >= 0.1:
            spTF_middle = np.mean(spTF_range)
            GapsBack_middle, _ = self.__inputTF_to_gapsAndMidT(spTF_middle, 'normal', 'back')
            if min(GapsBack_middle[:, 1]) > self.safeGap:
                spTF_range[0] = spTF_middle
            else:
                spTF_range[1] = spTF_middle
            # print(f'spTF_range: {spTF_range}')

        """ debug """
        # minGap_back = {}
        # tf_list = np.linspace(sp.tf_Range[0], sp.boundP_tf[-1], 100, endpoint=True)
        # for tf in tf_list:
        #     Gaps_back, LC_midT_back = self.__inputTF_to_gapsAndMidT(tf, 'normal', 'back')
        #     if LC_midT_back:
        #         minGap_back[tf] = min(Gaps_back[:, 1])
        #
        # plt.figure()
        # plt.plot(minGap_back.keys(), minGap_back.values(), 'k', lw=2, label='back')
        # plt.scatter(spTF_range[0], self.safeGap, marker='o', color='r', zorder=9)
        # plt.legend()
        # plt.show()

        return spTF_range[0]

    def debug_check_sp(self, sp_region):
        """ point-by-point calculation """
        sp = self.veh_sp

        overlap_FLAG = 'sweet' if sp_region == 'opt_tf' else 'normal'
        tf_end = sp.adjustD_tf[-1] if sp_region == 'opt_tf' else sp.boundP_tf[-1]

        tf_list = np.linspace(sp.tf_Range[0], tf_end, 100, endpoint=True)
        parent_midT_max, parent_midT_min = {}, {}
        front_midT_max, front_midT_min = {}, {}
        back_midT_max, back_midT_min = {}, {}

        """ calculate """
        for tf in tf_list:
            # parent
            _, LC_midT_parent = self.__inputTF_to_gapsAndMidT(tf, overlap_FLAG, 'parent')
            if LC_midT_parent:
                parent_midT_max[tf] = LC_midT_parent[-1]
                parent_midT_min[tf] = LC_midT_parent[0]
            # front
            if self.vehFormer_traj:
                _, LC_midT_front = self.__inputTF_to_gapsAndMidT(tf, overlap_FLAG, 'front')
                if LC_midT_front:
                    front_midT_max[tf] = LC_midT_front[-1]
                    front_midT_min[tf] = LC_midT_front[0]
            # back
            if self.vehBack_traj:
                _, LC_midT_back = self.__inputTF_to_gapsAndMidT(tf, overlap_FLAG, 'back')
                if LC_midT_back:
                    back_midT_max[tf] = LC_midT_back[-1]
                    back_midT_min[tf] = LC_midT_back[0]

        """ draw the figure """
        plt.figure()
        plt.plot(parent_midT_max.keys(), parent_midT_max.values(), 'k')
        plt.plot(parent_midT_min.keys(), parent_midT_min.values(), 'k', lw=4, label='parent')
        if self.vehFormer_traj:
            plt.plot(front_midT_max.keys(), front_midT_max.values(), 'b', lw=4, label='front')
            plt.plot(front_midT_min.keys(), front_midT_min.values(), 'b')
        if self.vehBack_traj:
            plt.plot(back_midT_max.keys(), back_midT_max.values(), 'g', lw=4, label='back')
            plt.plot(back_midT_min.keys(), back_midT_min.values(), 'g')

        plt.legend()
        plt.show()

    @staticmethod
    def trajs_toGaps(self_traj, other_traj, other_pos):
        t_self, a_self, v_self, p_self = zip(*self_traj)
        t_oth, a_oth, v_oth, p_oth = zip(*other_traj)

        """ aligning """
        if t_oth[-1] > t_self[-1]:
            t = t_oth
            delta_t = np.arange(1, len(t_oth) - len(t_self) + 1) / 10
            p_self = np.append(p_self, delta_t * v_self[-1] + p_self[-1])
        else:
            t = t_self
            delta_t = np.arange(1, len(t_self) - len(t_oth) + 1) / 10
            p_oth = np.append(p_oth, delta_t * v_oth[-1] + p_oth[-1])

        """ calculate Gaps """
        if other_pos in ['parent', 'front']:
            gaps = np.array(p_oth) - np.array(p_self) - Vehicle.length
        elif other_pos == 'back':
            gaps = np.array(p_self) - np.array(p_oth) - Vehicle.length
        else:
            gaps = None

        Gaps = list(zip(t, gaps))
        Gaps = np.array(Gaps)
        return Gaps


def vehsInfo_demo():
    from routes.route_static import Route_staticInput

    """ generate vehicles """
    R0_info = []
    M1_info = [['HDVs', 3, 2],
               ['CAVs', 2, 2],
               ['CAVs', 2, 2.5]]
    M2_info = [['CAVs', 2, 4]]

    route = Route_staticInput(R0_info, M1_info, M2_info)
    vehs_R0, vehs_M1, vehs_M2, vehs_All = route.generate_vehs()

    ''' show vehicles on road '''
    r = Road()
    r.x_lim = [-250, 50]
    r.plot(FLAG_fillRoad=True)

    for vehs_group in [vehs_M1, vehs_M2]:
        for veh in vehs_group.values():
            veh.plot(r.ax_road, FLAG_speedColor=False)

    """ define parent/front/back vehicles """
    vehsL_M1 = [veh_name for veh_name in vehs_M1 if veh_name[-1] == 'L']
    vehFront = vehs_M1[vehsL_M1[0]]
    vehSelf = vehs_M1[vehsL_M1[1]]
    vehBack = vehs_M1[vehsL_M1[2]]

    # Note: the former vehicle in Front and Parent group
    vehFront = vehFront.followers[-1] if vehFront.followers else vehFront
    vehParent = route.vehs_M2.get(list(route.vehs_M2.keys())[-1])

    """ generate trajectories for parent/front/back vehicles """
    for veh in [vehFront, vehBack]:
        veh.long_statePhase = Long_StatePhase(veh.time_c, veh.long_p, veh.long_v, veh.long_a, route='M1',
                                              v_des=Vehicle.v_des['M1'], v_min=veh.v_min)

    for veh in [vehParent, vehSelf]:
        veh.long_statePhase = Long_StatePhase(veh.time_c, veh.long_p, veh.long_v, veh.long_a, route='M2',
                                              v_des=Vehicle.v_des['M2'], v_min=veh.v_min)

    vehParent.long_trajPlan = vehParent.long_statePhase.get_trajectory_withPF(
        vehParent.long_statePhase.tf_Range[0] + 0.2)
    vehFront.long_trajPlan = vehFront.long_statePhase.get_trajectory_withPF(vehFront.long_statePhase.tf_Range[0] + 0.6)
    vehBack.long_trajPlan = vehBack.long_statePhase.get_trajectory_withPF(vehBack.long_statePhase.tf_Range[0] + 0.4)

    return vehSelf, vehParent, vehFront, vehBack


if __name__ == "__main__":
    """ 0. load vehicles """
    vehSelf, vehParent, vehFront, vehBack = vehsInfo_demo()
    # show_severalTrajs({'b': vehFront.long_trajPlan, 'k': vehBack.long_trajPlan, 'purple': vehParent.long_trajPlan})

    """ 1. safe_LC """
    sp_safeLC = StatePhase_safeLC(vehSelf, vehSelf.long_statePhase,
                                  vehParent.long_trajPlan, vehFront.long_trajPlan, vehBack.long_trajPlan)
    print(f'*** ref_tf: {sp_safeLC.ref_tf}, tf_Range[0]: {sp_safeLC.veh.long_statePhase.tf_Range[0]}, \n'
          f'*** adjustD_tf[-1]: {sp_safeLC.veh.long_statePhase.adjustD_tf[-1]}, \n'
          f'*** boundP_tf[-1]: {sp_safeLC.veh.long_statePhase.boundP_tf[-1]} \n')

    sp_safeLC.FLAG_print = True
    sp_safeLC.get_desiredTF_andMidT()

    sp_safeLC.debug_check_sp('opt_pf')

    plt.show()
