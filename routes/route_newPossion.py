"""
    Description: New Poisson Distribution
    Author: Tenplus
    Create-time: 2022-08-30
    Update-time: 2022-11-03
    Note: # v1.0 多车道的研究，对车流的重新定义

    Update-time: 2023-07-04
    Note: # v2.0 新的泊松分布，创建总的车辆生成时间

"""

import sys
from objects.Vehicle import Vehicle, CACC, OVM
from objects.Road import Road
import numpy as np


class Route_LaneNewPoisson:
    seeds_dict = {'R0': 1998, 'M1': 2015, 'M2': 2019}
    lane_list = {'R0': 0, 'M1': 1, 'M2': 2}
    departPos_dict = {'R0': Road.RAMP_STRAIGHT_RANGE[0], 'M1': Road.MAIN_RANGE[0], 'M2': Road.MAIN_RANGE[0]}
    initSpeed_dict = {'R0': Vehicle.v_R0, 'M1': Vehicle.v_des['M1'], 'M2': Vehicle.v_des['M2']}

    def __init__(self, t_simu=100, route='R0', vehP_lamb=0.2, p_typeDistri=None, p_sizeDistri=None, seed=None,
                 FLAG_slack=True, slack_inter=4, slack_range=[0.1, 0.3]):
        self.route = route
        seed = seed if seed else self.seeds_dict[route]
        np.random.seed(seed)  # TODO: random off

        self.p_num = 0
        self.v_num = 0

        ''' return '''
        self.depart_final = 0
        self.vehs = {}

        """ define platoon information """
        while self.depart_final <= t_simu:
            ''' decision platoon depart time '''
            if self.p_num == 0:
                p_depart = self.depart_final
            else:
                delta_t_poisson = np.random.poisson(lam=1 / vehP_lamb * 10, size=1)[0] / 10
                p_depart = self.depart_final + delta_t_poisson

            # add slack
            if FLAG_slack and self.p_num and self.p_num % slack_inter == 0:
                depart_slack = np.around(np.random.uniform(slack_range[0], slack_range[1]), 1)
                p_depart += depart_slack

            ''' decision platoon type and size'''
            p_typeDistri = p_typeDistri if p_typeDistri else {'CAVs': 0.6, 'HDVs': 0.4}  # default platoons type distri
            p_sizeDistri = p_sizeDistri if p_sizeDistri else [0.2, 0.3, 0.3, 0.2]  # default platoon size distribution

            p_type = np.random.choice(list(p_typeDistri.keys()), p=list(p_typeDistri.values()))
            p_size = np.random.choice(np.arange(len(p_sizeDistri)) + 1, p=p_sizeDistri)

            p_initSpeed = self.initSpeed_dict[route]
            inner_dp = CACC.t_c * p_initSpeed + CACC.s0 + Vehicle.length if p_type == 'CAVs' else \
                OVM.vel2spacing(p_initSpeed)

            """ new vehicle group """
            self.__generate_vehsGroup(self.route, self.p_num, self.v_num, p_type, p_size, p_depart, p_initSpeed, inner_dp)

            ''' update info '''
            p_inner_dt = np.around((p_size - 1) * inner_dp / p_initSpeed, 1)

            self.depart_final = p_depart + p_inner_dt
            self.p_num += 1
            self.v_num += p_size

        """ calculate the flow rate """
        self.flowRate = self.v_num / self.depart_final

    def __generate_vehsGroup(self, route, p_id, vehL_id, p_type, p_size, p_depart, p_departSpeed, inner_dp):
        leader_name = '%s_%d_%s_%d_%dL' % (route, vehL_id, p_type, p_id, p_size)
        p_depart = np.around(p_depart, 1)

        for i in range(p_size):
            v_id_ = vehL_id + i
            v_id_ = vehL_id + i
            name = '%s_%d_%s_%d_%s' % (route, v_id_, p_type, p_id, i + 1)
            name = leader_name if i == 0 else name
            typeV = 'HDV' if p_type == 'HDVs' and i != 0 else 'CAV'
            laneID = self.lane_list[route]
            departPos = self.departPos_dict[route] - i * inner_dp
            front_veh = self.vehs.get(list(self.vehs.keys())[-1]) if self.vehs else None
            leader_veh = self.vehs[leader_name] if i != 0 else None
            veh = Vehicle(name, typeV, route, laneID=laneID,
                          depart=p_depart, departPos=departPos, departSpeed=p_departSpeed,
                          typeP=p_type, sizeP=p_size, front=front_veh, leader=leader_veh)

            if i != 0:
                self.vehs[leader_name].followers.append(veh)
            self.vehs[name] = veh

        #     print(f'** new vehicle: {name}, depart: {p_depart}')
        # print()

    def get_vehs(self):
        return self.vehs


if __name__ == '__main__':
    """ define lane route """

    r1 = Route_LaneNewPoisson()
    r1_vehs = r1.get_vehs()

    # for veh in r1_vehs.values():
    #     print(veh.name, veh.depart)
