"""
    Description: Generate continue routes.
    Author: Tenplus
    Create-time: 2022-08-30
    Update-time: 2022-11-03
    Note: # v1.0 多车道的研究，对车流的重新定义
"""

import sys
from objects.Vehicle import Vehicle, CACC, OVM
from objects.Road import Road
import numpy as np


class Route_LanePoisson:
    seeds_dict = {'R0': 1998, 'M1': 2015, 'M2': 2019}
    lane_list = {'R0': 0, 'M1': 1, 'M2': 2}
    departPos_dict = {'R0': Road.RAMP_STRAIGHT_RANGE[0], 'M1': Road.MAIN_RANGE[0], 'M2': Road.MAIN_RANGE[0]}
    initSpeed_dict = {'R0': Vehicle.v_R0, 'M1': Vehicle.v_des['M1'], 'M2': Vehicle.v_des['M2']}

    def __init__(self, route='R0', p_num=6, veh_lamb=0.2, p_typeDistri=None, p_sizeDistri=None, seed=None):
        self.route = route

        """ define platoon information """
        p_typeDistri = p_typeDistri if p_typeDistri else {'CAVs': 0.6, 'HDVs': 0.4}  # default platoons type distri
        p_sizeDistri = p_sizeDistri if p_sizeDistri else [0.2, 0.3, 0.3, 0.2]  # default platoon size distribution
        seed = seed if seed else self.seeds_dict[route]
        np.random.seed(seed)    # TODO: random off

        p_typeList = np.random.choice(list(p_typeDistri.keys()), p=list(p_typeDistri.values()), size=p_num)
        p_sizeList = np.random.choice(np.arange(len(p_sizeDistri)) + 1, p=p_sizeDistri, size=p_num)
        self.p_sizeList = p_sizeList
        p_initSpeedList = [self.initSpeed_dict[route]] * p_num  # TODO: temp, the same speed at first

        """ generate depart time according to Poisson Distribution """
        veh_departInterval = np.random.poisson(lam=1 / veh_lamb * 10, size=sum(p_sizeList)) / 10
        veh_departList = np.cumsum(veh_departInterval) - veh_departInterval[0]
        veh_departList = np.around(veh_departList, 1)
        leader_numList = [0] + list(np.cumsum(p_sizeList)[:-1])
        p_departList = [veh_departList[v_id] for v_id in leader_numList]

        self.p_infoList = list(zip(p_typeList, p_sizeList, p_departList, p_initSpeedList))

        """ return """
        self.vehs = {}

    def generate_vehs(self):
        veh_id = 0

        for p_info in self.p_infoList:
            p_id = self.p_infoList.index(p_info)
            p_type = p_info[0]
            p_size = p_info[1]
            p_depart = p_info[2]
            p_departSpeed = p_info[3]

            inner_dp = CACC.t_c * p_departSpeed + CACC.s0 + Vehicle.length if p_type == 'CAVs' else \
                OVM.vel2spacing(p_departSpeed)
            leader_name = '%s_%d_%s_%d_L' % (self.route, veh_id + 1, p_type, p_id)

            for i in range(p_size):
                veh_id += 1
                name = '%s_%d_%s_%d_%s' % (self.route, veh_id, p_type, p_id, i + 1)
                name = leader_name if i == 0 else name
                typeV = 'HDV' if p_type == 'HDVs' and i != 0 else 'CAV'
                laneID = self.lane_list[self.route]
                departPos = self.departPos_dict[self.route] - i * inner_dp
                front_veh = self.vehs.get(list(self.vehs.keys())[-1]) if i != 0 else None
                leader_veh = self.vehs[leader_name] if i != 0 else None
                veh = Vehicle(name, typeV, self.route, laneID=laneID,
                              depart=p_depart, departPos=departPos, departSpeed=p_departSpeed,
                              typeP=p_type, sizeP=p_size, front=front_veh, leader=leader_veh)

                if i != 0:
                    self.vehs[leader_name].followers.append(veh)
                self.vehs[name] = veh

        return self.vehs


if __name__ == '__main__':
    """ define lane route """
    R0_info = Route_LanePoisson(route='R0')
    R0_route = R0_info.generate_vehs()
    for veh in R0_route.values():
        print(veh.name, veh.depart, veh.long_p)

    M1_info = Route_LanePoisson(route='M1', p_num=10, seed=1988)
    M1_route = M1_info.generate_vehs()

    print(R0_info.p_sizeList)
    print(M1_info.p_sizeList)