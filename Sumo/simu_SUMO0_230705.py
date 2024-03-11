"""
    The contrast experiment: SUMO
    Author: Tenplus
    Create-time: 2023-07-04
    Update-time: 2022-07-05  # V1.3, 重写sumo仿真算法，集成，简化
"""

import sys, os, time
import traci
import numpy as np
import matplotlib.pyplot as plt
import warnings

from sumolib import checkBinary


# from routes.routes_caseStudy import Continue_uniform_routes
# from paper_ex2.route_ex2 import Poisson_routes
# from cloudController.schedules import Schedule
# from plotFun.plotResults import show_single_traj, show_trajectories, show_delays, show_delay_stack
# from objects.Config import GlobalParams
# from cloudController.statePhase import StatePhase

class Simulation_SUMO:
    colors = {'purple': (128, 0, 128), 'gray': (128, 128, 128), 'r': (255, 0, 0), 'b': (0, 0, 255)}

    def __init__(self, vehs_R0, vehs_M1, vehs_M2, FLAG_GUI=False, FLAG_traj_save=False):
        """ initialize, load vehicles """
        self.vehs_R0 = vehs_R0
        self.vehs_M1 = vehs_M1
        self.vehs_M2 = vehs_M2
        self.vehs_All = dict(self.vehs_R0, **self.vehs_M1, **vehs_M2)

        """ create folder to save data and figures """
        if FLAG_traj_save:
            folder_name = time.strftime('%Y%m%d_%H%M_%S_SUMO', time.localtime(time.time()))
            self.savePath = '../../04 Data saving/movingProcess/' + folder_name
            os.makedirs(self.savePath)

        """ sumo config """
        self.FLAG_GUI = FLAG_GUI
        sumoBinary = checkBinary('sumo-gui' if self.FLAG_GUI else 'sumo')
        config_dir = os.path.abspath(os.path.dirname(__file__)) + '\\config'

        traci.start([sumoBinary, '--net-file', config_dir + '\\ramp.net.xml',
                     '--route-files', config_dir + '\\ramp.rou.xml',
                     '--gui-settings-file', config_dir + '\\ramp.gui.xml',
                     '--step-length', '0.1', '--delay', '100',
                     '--start', 'false', '--quit-on-end', 'false', '--no-warnings'])  # no-warnings

        """ add vehicles """
        self.add_vehicles()

        self.step = 0

        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()

            self.__setMaxSpeed(traci)
            self.__dataLog(traci)

            self.step += 1

            """ print """
            if self.step % 40 == 0:
                print(f'step: {self.step}')

        traci.close()

        """ save trajectories """
        if FLAG_traj_save:
            self.__save_trajectories()

    def __setMaxSpeed(self, traci):
        for veh_name in traci.vehicle.getIDList():
            if veh_name[:2] == 'M1':
                if traci.vehicle.getLaneID(veh_name) in ['Main_start_1', 'Main_1', 'Merge_2', 'Converge_1']:
                    traci.vehicle.setMaxSpeed(veh_name, 25)
                else:
                    traci.vehicle.setMaxSpeed(veh_name, 20)

            if veh_name[:2] == 'R0':
                if traci.vehicle.getLaneID(veh_name) in ['Ramp_start_0', 'Ramp_limited_0']:
                    traci.vehicle.setMaxSpeed(veh_name, 10)
                else:
                    traci.vehicle.setMaxSpeed(veh_name, 20)

    def __dataLog(self, traci):
        for veh_name in traci.vehicle.getIDList():
            if veh_name[0] == 'M':
                self.vehs_All[veh_name].states_log.append([self.step / 10,
                                                           traci.vehicle.getPosition(veh_name)[0],
                                                           traci.vehicle.getSpeed(veh_name),
                                                           traci.vehicle.getPosition(veh_name)[1]])
            elif veh_name[0] == 'R':
                if traci.vehicle.getLaneID(veh_name) in ['Ramp_start_0', ':J4_0_0', 'Ramp_limited_0', ':J3_0_0',
                                                         'Ramp_0', ':gneJ6_0_0']:
                    long_p = - traci.vehicle.getDrivingDistance(veh_name, 'Merge', 0) - 15.5   # calibration
                else:
                    long_p = traci.vehicle.getPosition(veh_name)[0]

                self.vehs_All[veh_name].states_log.append([self.step / 10,
                                                           long_p,
                                                           traci.vehicle.getSpeed(veh_name),
                                                           traci.vehicle.getPosition(veh_name)[1]])
                # if veh_name == 'R0_0_HDVs_0_2L':
                #     print(self.step, long_p)

    def __save_trajectories(self):
        for veh in self.vehs_All.values():
            if veh.states_log:
                np.save(self.savePath + '/%s.npy' % veh.name, np.array(veh.states_log))

    def add_vehicles(self):

        MAIN_LENGTH = traci.lane.getLength('Main_start_0') + traci.lane.getLength('Main_0')
        RAMP_LENGTH = traci.lane.getLength('Ramp_start_0') + traci.lane.getLength('Ramp_limited_0') + \
                      traci.lane.getLength('Ramp_0')

        """ vehicles in M2 """
        for veh in self.vehs_M2.values():
            departPos = MAIN_LENGTH + veh.long_p
            traci.vehicle.add(veh.name, 'Main', typeID=veh.typeV, depart=veh.depart, departPos=departPos,
                              departSpeed=veh.long_v - 0.1, departLane=1, arrivalLane=1)
            traci.vehicle.setColor(veh.name, self.colors[veh.color])
            traci.vehicle.setSpeedMode(veh.name, 31)
            traci.vehicle.setMaxSpeed(veh.name, 25)
            traci.vehicle.setLaneChangeMode(veh.name, 512)  # no lane change

        """ vehicles in M1 """
        for veh in self.vehs_M1.values():
            departPos = MAIN_LENGTH + veh.long_p
            traci.vehicle.add(veh.name, 'Main', typeID=veh.typeV, depart=veh.depart, departPos=departPos,
                              departSpeed=veh.long_v - 4, departLane=0)  # TODO: assume here
            traci.vehicle.setColor(veh.name, self.colors[veh.color])
            traci.vehicle.setSpeedMode(veh.name, 31)
            traci.vehicle.setMaxSpeed(veh.name, 20)
            # traci.vehicle.setLaneChangeMode(veh.name, 16)   # only left lane-change allowed

        """ vehicles in R0 """
        for veh in self.vehs_R0.values():
            departPos = RAMP_LENGTH + veh.long_p
            traci.vehicle.add(veh.name, 'Ramp', typeID=veh.typeV, depart=veh.depart, departPos=departPos,
                              departSpeed=veh.long_v - 0.1, departLane=0, arrivalLane=0)
            traci.vehicle.setColor(veh.name, self.colors[veh.color])
            traci.vehicle.setSpeedMode(veh.name, 31)
            traci.vehicle.setParameter(veh.name, 'lcStrategic', str(np.random.uniform(0.5, 5)))


if __name__ == '__main__':
    """ generate routes """
    from routes.route_newPossion import Route_LaneNewPoisson

    t_simu = 300
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

    """ create the simulation with OFPD_multiLanes algorithm """
    simu_sumo = Simulation_SUMO(R0_vehs, M1_vehs, M2_vehs, FLAG_GUI=True, FLAG_traj_save=True)
