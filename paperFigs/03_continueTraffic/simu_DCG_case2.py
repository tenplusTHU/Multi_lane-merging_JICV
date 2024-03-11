"""
    Description: Receding Horizon Optimization Algorithm based on Dynamic conflict graph
    Author: Tenplus
    Create-time: 2022-11-15
    Update-time: 2023-02-22
"""

import math
import time
import os
import matplotlib.pyplot as plt
import numpy as np
from objects.Road import Road
from cloudController.vehsNetwork import createFig_split, VehsNetwork
from utilities.plotScenarios import show_scenario_simu
from utilities.plotResults import show_trajs_threeLanes, show_severalTrajs


plt.rcParams['font.family'] = 'Arial'


class Simulation_DCG:
    def __init__(self, vehs_R0, vehs_M1, vehs_M2,
                 FLAG_scene_pause=False, FLAG_scene_save=False, FLAG_network_save=False, FLAG_traj_save=False):
        """ initialize, load vehicles """
        self.vehs_R0 = vehs_R0
        self.vehs_M1 = vehs_M1
        self.vehs_M2 = vehs_M2
        self.vehs_All = dict(self.vehs_R0, **self.vehs_M1, **vehs_M2)

        """ get vehicle-groups depart TimeTabel """
        self.departTimetable = self.__get_departTimetable()
        # # **** debug, print ****
        # departTime = list(self.departTimetable.keys())
        # departTime.sort()
        # print(f'Depart Timetable: {departTime}')
        # sys.exit()

        """ scenario """
        self.road = Road()
        self.fig_scene, self.fig_network = None, None
        self.FLAG_scene_pause = FLAG_scene_pause
        self.FLAG_scene_save = FLAG_scene_save
        self.FLAG_network_save = FLAG_network_save
        self.FLAG_traj_save = FLAG_traj_save

        """ create folder to save data and figures """
        if self.FLAG_scene_save or self.FLAG_network_save or self.FLAG_traj_save:
            folder_name = time.strftime('%Y%m%d_%H%M_%S', time.localtime(time.time()))
            self.savePath = '../../../04 Data saving/movingProcess/' + folder_name
            os.makedirs(self.savePath)
            # scenario
            if self.FLAG_scene_save:
                os.makedirs(self.savePath + '/scene')
            # network
            if self.FLAG_network_save:
                os.makedirs(self.savePath + '/network')
                ''' create the initial network '''
                self.fig_network = createFig_split(fig_size=(5, 3))
                plt.savefig(f'{self.savePath}/network/0.jpg')
            # trajs
            if self.FLAG_traj_save:
                os.makedirs(self.savePath + '/trajs')

        """ states parameters """
        self.vehs_departed, self.vehs_InMap = [], []  # [veh_name], all vehicles
        self.vehs_sche_R0, self.vehs_sche_M1, self.vehs_sche_M2 = {}, {}, {}  # {veh_name: veh}, vehName
        self.rootVehL_M1, self.rootVehL_M2 = None, None
        self.orderID_M1, self.orderID_M2 = 0, 0

        """ other parameters """
        self.FLAG_newSchedule = False  # when new vehicle-group enters
        self.vehsNetwork = None
        self.step = 0

    def run(self):
        while not (len(self.vehs_departed) == len(self.vehs_All) and
                   len(self.vehs_InMap) == 0) and self.step < 100000:  # terminal condition
            """ run step-by-step """
            self.__vehDepart()  # 1. vehicle depart
            if self.step % 10 == 0:  # TODO: change Debug simulation speed here
                self.__vehVisualization()  # 2. pause and save figs
            self.__newSchedule_three()
            self.__driveAndUpdate()

            if self.step % 40 == 0:
                print(f'step: {self.step}')
            self.step += 1

        """ save trajectories """
        if self.FLAG_traj_save:
            self.__save_trajectories()

    def __get_departTimetable(self):
        departTimetable = {}  # {depart: [veh_name]}, one decimal
        for veh in self.vehs_All.values():
            if veh.depart not in departTimetable.keys():
                departTimetable[veh.depart] = [veh.name]
            else:
                departTimetable[veh.depart].append(veh.name)

        return departTimetable

    def __vehDepart(self):
        t_step = self.step / 10
        if t_step in self.departTimetable.keys():  # new vehicle group enters
            for veh_name in self.departTimetable[t_step]:
                self.vehs_departed.append(veh_name)
                self.vehs_InMap.append(veh_name)

    def __vehVisualization(self):
        if self.FLAG_scene_pause or self.FLAG_scene_save:
            """ show scenario """
            if not self.fig_scene:
                self.fig_scene = plt.figure(figsize=(10, 2))

            show_scenario_simu(self, FLAG_speedColor=False)

            """ save and pause figure """
            if self.FLAG_scene_save:
                plt.savefig(f'{self.savePath}/scene/{self.step}.jpg')
            if self.FLAG_scene_pause:
                plt.pause(1e-4)

    def __newSchedule_three(self):
        if self.FLAG_newSchedule:
            print(f'*** new schedule, step: {self.step}')
            """ new schedules_three """
            if self.step in [135]:
                print(f'*** debug here at newSchedule_three, step: {self.step} ***')
            self.vehsNetwork = VehsNetwork(self.vehs_sche_R0, self.vehs_sche_M1, self.vehs_sche_M2,
                                           rootVehL_M1=self.rootVehL_M1, rootVehL_M2=self.rootVehL_M2,
                                           LC_max_numP=2, LC_max_sizeP_CAVs=2)
            optSche = self.vehsNetwork.get_optimalSchedule()
            optSche.get_optimalTrajectories()

            """ save vehsNetwork """
            if self.FLAG_network_save and self.vehsNetwork:
                self.fig_network = createFig_split(fig_size=(5, 3))  # add, 2023-01-08
                optSche.show_optVehsNetwork(fig_network=self.fig_network, fig_size=(5, 3), FLAG_J=False)
                plt.savefig(f'{self.savePath}/network/{self.step}.jpg')
                plt.close()

            """ show sequence-Tree search process, for debug """
            # if self.step == 75:
            #     fig_tree = plt.figure(figsize=(5.5, 5.5))
            #     optSche.show_sequenceTree(fig_tree=fig_tree, FLAG_J=False)
            #     plt.show()
            #     # plt.savefig()
            #     plt.close()

            self.FLAG_newSchedule = False

    def __driveAndUpdate(self):
        offMap_popList = []

        """ drive a step """
        for veh_name in self.vehs_InMap:
            veh = self.vehs_All[veh_name]
            veh.states_log.append(veh.states_current)  # save current states
            ''' longitudinal movement '''
            if veh.long_trajPlan:
                veh.time_c, veh.long_a, veh.long_v, veh.long_p = list(veh.long_trajPlan[0])
                veh.long_trajPlan.pop(0)
            else:
                veh.time_c, veh.long_p = (self.step + 1) / 10, veh.long_p + veh.long_v * 0.1  # a, v: constant

            ''' lateral movement '''
            if veh.lat_trajPlan and abs(self.step + 1 - veh.lat_trajPlan[0][0] * 10) < 1e-3:
                _, veh.lat_a, veh.lat_v, veh.lat_p = list(veh.lat_trajPlan[0])
                veh.yaw = math.degrees(math.atan(veh.lat_v / veh.long_v))
                veh.lat_trajPlan.pop(0)

            ''' states summary, next step '''
            # veh.states_current = [veh.time_c, veh.long_p, veh.long_v, veh.long_a,
            #                       veh.lat_p, veh.lat_v, veh.lat_a, veh.yaw, veh.route]

            veh.states_current = [veh.time_c, veh.long_p, veh.long_v, veh.long_a,
                                  veh.lat_p, veh.lat_v, veh.lat_a, veh.yaw]

            ''' whether offMap? '''
            if veh.long_p >= Road.MAIN_RANGE[1]:  # whether offMap
                offMap_popList.append(veh.name)

            ''' update the rootVehL_M1 and rootVehL_M2 '''  # add, 2023-01-02
            if (veh.name in self.vehs_sche_M1.keys() and veh.long_p > Road.M1_SCH_RANGE[1] and veh.route == 'M1') or \
                    (veh.name in self.vehs_sche_R0.keys() and veh.long_p > Road.R0_SCH_RANGE[1]):
                if veh.name[-1] == 'L':
                    self.rootVehL_M1 = veh
                    self.orderID_M1 += 1
                    veh.orderID_M1 = self.orderID_M1
            if (veh.name in self.vehs_sche_M1.keys() and veh.long_p > Road.M2_SCH_RANGE[1] and veh.route == 'M2') or \
                    (veh.name in self.vehs_sche_M2.keys() and veh.long_p > Road.M2_SCH_RANGE[1]):
                if veh.name[-1] == 'L':
                    self.rootVehL_M2 = veh
                    self.orderID_M2 += 1
                    veh.orderID_M2 = self.orderID_M2

            ''' whether in schedule zone: R0? '''
            if veh.name[-1] == 'L' and veh.name[0:2] == 'R0' and veh.name not in self.vehs_sche_R0.keys() and \
                    -350 <= veh.long_p <= Road.R0_SCH_RANGE[1]:   # modified here, 23-07-05
                self.FLAG_newSchedule = True
                self.vehs_sche_R0[veh.name] = veh
                if veh.followers:
                    for vehFollower in veh.followers:
                        self.vehs_sche_R0[vehFollower.name] = vehFollower
            if veh.name in self.vehs_sche_R0.keys() and veh.long_p > Road.R0_SCH_RANGE[1]:
                self.vehs_sche_R0.pop(veh.name)

            ''' whether in schedule zone: M1? '''
            if veh.name[-1] == 'L' and veh.name[0:2] == 'M1' and veh.name not in self.vehs_sche_M1.keys() and \
                    Road.M1_SCH_RANGE[0] <= veh.long_p <= Road.M1_SCH_RANGE[1]:
                self.FLAG_newSchedule = True
                self.vehs_sche_M1[veh.name] = veh
                if veh.followers:
                    for vehFollower in veh.followers:
                        self.vehs_sche_M1[vehFollower.name] = vehFollower
            if veh.name in self.vehs_sche_M1.keys() and veh.long_p > Road.M1_SCH_RANGE[1]:
                self.vehs_sche_M1.pop(veh.name)

            ''' whether in schedule zone: M2? '''
            if veh.name[-1] == 'L' and veh.name[0:2] == 'M2' and veh.name not in self.vehs_sche_M2.keys() and \
                    Road.M2_SCH_RANGE[0] <= veh.long_p <= Road.M2_SCH_RANGE[1]:
                self.FLAG_newSchedule = True
                self.vehs_sche_M2[veh.name] = veh
                if veh.followers:
                    for vehFollower in veh.followers:
                        self.vehs_sche_M2[vehFollower.name] = vehFollower
            if veh.name in self.vehs_sche_M2.keys() and veh.long_p > Road.M2_SCH_RANGE[1]:
                self.vehs_sche_M2.pop(veh.name)

        """ whether offMap """
        self.vehs_InMap = list(set(self.vehs_InMap) ^ set(offMap_popList))

    def __save_trajectories(self):
        for veh in self.vehs_All.values():
            np.save(self.savePath + '/trajs/%s.npy' % veh.name, np.array(veh.states_log))


if __name__ == "__main__":
    """ generate routes """
    from routes.route_newPossion import Route_LaneNewPoisson

    t_simu = 10
    R0_route = Route_LaneNewPoisson(t_simu, 'R0', vehP_lamb=0.25,
                                    p_typeDistri={'CAVs': 0.2, 'HDVs': 0.8}, p_sizeDistri=[0.3, 0.4, 0.2, 0.1], seed=72,
                                    FLAG_slack=True, slack_inter=3, slack_range=[0.6, 1.2])
    M1_route = Route_LaneNewPoisson(t_simu, 'M1', vehP_lamb=0.52,
                                    p_typeDistri={'CAVs': 0.75, 'HDVs': 0.25}, p_sizeDistri=[0.35, 0.4, 0.25], seed=98,
                                    FLAG_slack=True, slack_inter=3, slack_range=[0.2, 0.6])
    M2_route = Route_LaneNewPoisson(t_simu, 'M2', vehP_lamb=0.20,
                                    p_typeDistri={'CAVs': 0.5, 'HDVs': 0.5}, p_sizeDistri=[0.2, 0.3, 0.3, 0.2],
                                    seed=183,
                                    FLAG_slack=True, slack_inter=3, slack_range=[0.2, 0.4])
    R0_vehs = R0_route.get_vehs()
    M1_vehs = M1_route.get_vehs()
    M2_vehs = M2_route.get_vehs()

    print(f'**** flow rate **** \n'
          f'** R0: {round(R0_route.flowRate, 2)} veh/s, {int(R0_route.flowRate*3600)} veh/hour \n'
          f'** M1: {round(M1_route.flowRate, 2)} veh/s, {int(M1_route.flowRate*3600)} veh/hour \n'
          f'** M2: {round(M2_route.flowRate, 2)} veh/s, {int(M2_route.flowRate*3600)} veh/hour \n')

    """ create the simulation with OFPD_multiLanes algorithm """
    simu = Simulation_DCG(R0_vehs, M1_vehs, M2_vehs,
                          FLAG_scene_pause=False, FLAG_scene_save=False,
                          FLAG_network_save=False, FLAG_traj_save=True)
    simu.run()

    plt.show()
