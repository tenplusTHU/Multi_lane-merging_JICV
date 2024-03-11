"""
    Description: different
    Author: Tenplus
    Create-time: 2023-07-10
    Update-time: 2023-07-10
    Note: # v1.0 不同流量下的仿真
"""

import math
import time
import os, sys
import matplotlib.pyplot as plt
import numpy as np
from objects.Road import Road
from cloudController.vehsNetwork import createFig_split, VehsNetwork
from utilities.plotScenarios import show_scenario_simu

from simulations.simu_DCG import Simulation_DCG
from Sumo.simu_SUMO0_230705 import Simulation_SUMO

from routes.route_newPossion import Route_LaneNewPoisson
from paperFigs.dataAnalysis.trajsAnalysis import TrajsAnalysis


def generate_routes(t_simu, lamb_R0, lamb_M1, lamb_M2,
                    slack_R0=[0.6, 1.2], slack_M1=[0.2, 0.6], c=[0.2, 0.4]):
    R0_route = Route_LaneNewPoisson(t_simu, 'R0', vehP_lamb=lamb_R0,
                                    p_typeDistri={'CAVs': 0.2, 'HDVs': 0.8}, p_sizeDistri=[0.3, 0.4, 0.2, 0.1], seed=72,
                                    FLAG_slack=True, slack_inter=3, slack_range=slack_R0)
    M1_route = Route_LaneNewPoisson(t_simu, 'M1', vehP_lamb=lamb_M1,
                                    p_typeDistri={'CAVs': 0.75, 'HDVs': 0.25}, p_sizeDistri=[0.35, 0.4, 0.25], seed=98,
                                    FLAG_slack=True, slack_inter=3, slack_range=slack_M1)
    M2_route = Route_LaneNewPoisson(t_simu, 'M2', vehP_lamb=lamb_M2,
                                    p_typeDistri={'CAVs': 0.5, 'HDVs': 0.5}, p_sizeDistri=[0.2, 0.3, 0.3, 0.2],
                                    seed=183,
                                    FLAG_slack=True, slack_inter=3, slack_range=slack_M1)
    R0_vehs = R0_route.get_vehs()
    M1_vehs = M1_route.get_vehs()
    M2_vehs = M2_route.get_vehs()

    print(f'**** flow rate **** \n'
          f'** R0: {round(R0_route.flowRate, 2)} veh/s, {int(R0_route.flowRate * 3600)} veh/hour \n'
          f'** M1: {round(M1_route.flowRate, 2)} veh/s, {int(M1_route.flowRate * 3600)} veh/hour \n'
          f'** M2: {round(M2_route.flowRate, 2)} veh/s, {int(M2_route.flowRate * 3600)} veh/hour \n')

    return R0_vehs, M1_vehs, M2_vehs


if __name__ == "__main__":
    """ generate routes """
    t_simu = 300

    # # lamb_R0=0.35, lamb_M1=0.54
    # lamb_R0, lamb_M1, lamb_M2 = 0.25, 0.4, 0.20

    # lamb_R0=0.35, lamb_M1=0.38
    # lamb_R0, lamb_M1, lamb_M2 = 0.25, 0.25, 0.20

    # lamb_R0=0.25, lamb_M1=0.68
    # lamb_R0, lamb_M1, lamb_M2 = 0.17, 0.52, 0.20    # 0.19 for DCG

    # lamb_R0=0.25, lamb_M1=0.54
    # lamb_R0, lamb_M1, lamb_M2 = 0.17, 0.4, 0.20

    # lamb_R0=0.25, lamb_M1=0.28
    lamb_R0, lamb_M1, lamb_M2 = 0.17, 0.25, 0.20

    """ DCG """
    # R0_vehs, M1_vehs, M2_vehs = generate_routes(t_simu, lamb_R0, lamb_M1, lamb_M2)
    # simu_dcg = Simulation_DCG(R0_vehs, M1_vehs, M2_vehs,
    #                       FLAG_scene_pause=False, FLAG_scene_save=False,
    #                       FLAG_network_save=False, FLAG_traj_save=True)
    # simu_dcg.run()
    #
    # trajsAna_dcg = TrajsAnalysis(simu_dcg.savePath + '/trajs')
    # delays_dcg = trajsAna_dcg.calculate_Delays()
    # meanV_R0_dcg, meanV_M1_dcg, meanV_M2_dcg = trajsAna_dcg.calculate_AverageV()
    #
    # print(f'*** DCG performance *** \n'
    #       f'* total J: {delays_dcg[-1]} \n'
    #       f'* average V, R0: {meanV_R0_dcg}, M1: {meanV_M1_dcg}, M2: {meanV_M2_dcg} \n')

    """ SUMO """
    R0_vehs, M1_vehs, M2_vehs = generate_routes(t_simu, lamb_R0, lamb_M1, lamb_M2)
    simu_sumo = Simulation_SUMO(R0_vehs, M1_vehs, M2_vehs, FLAG_GUI=False, FLAG_traj_save=True)

    trajsAna_sumo = TrajsAnalysis(simu_sumo.savePath)
    delays_sumo = trajsAna_sumo.calculate_Delays()
    meanV_R0_sumo, meanV_M1_sumo, meanV_M2_sumo = trajsAna_sumo.calculate_AverageV()

    print(f'*** SUMO performance *** \n'
          f'* total J: {delays_sumo[-1]} \n'
          f'* average V, R0: {meanV_R0_sumo}, M1: {meanV_M1_sumo}, M2: {meanV_M2_sumo}')

    plt.show()
