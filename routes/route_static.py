"""
    Description: Generate static_input routes for case study (demo description)
    Author: Tenplus
    Create-time: 2022-08-28
    Update-time: 2023-02-15
    Note: # v1.4 新学期的更新
"""

from objects.Vehicle import Vehicle, CACC, OVM
from objects.Road import Road
import matplotlib.pyplot as plt


class Route_staticInput:
    def __init__(self, R0_info=None, M1_info=None, M2_info=None):
        self.R0_lastDepartPos, self.M1_lastDepartPos, self.M2_lastDepartPos = 0, 0, 0
        self.R0_vehID, self.M1_vehID, self.M2_vehID = 0, 0, 0

        """ default info """  # [typeP, sizeP, delta_t]
        R0_info_default = [['HDVs', 3, 17],
                           ['CAVs', 1, 5]]
        M1_info_default = [['CAVs', 2, 10],
                           ['CAVs', 1, 2],
                           ['HDVs', 3, 2],
                           ['CAVs', 4, 2.5],
                           ['HDVs', 2, 3]]
        M2_info_default = [['CAVs', 2, 10.5],
                           ['CAVs', 3, 3.4],
                           ['HDVs', 2, 3.5]]

        self.R0_info = R0_info if R0_info else R0_info_default
        self.M1_info = M1_info if M1_info else M1_info_default
        self.M2_info = M2_info if M2_info else M2_info_default

        """ dicts to save the routes """
        self.vehs_R0, self.vehs_M1, self.vehs_M2 = {}, {}, {}

    def generate_vehs(self):
        info_list = [self.R0_info, self.M1_info, self.M2_info]  # input info
        route_list = ['R0', 'M1', 'M2']
        lane_list = [0, 1, 2]
        vehID_list = [self.R0_vehID, self.M1_vehID, self.M2_vehID]
        lastDepartPos_list = [self.R0_lastDepartPos, self.M1_lastDepartPos, self.M2_lastDepartPos]

        vehs_list = [self.vehs_R0, self.vehs_M1, self.vehs_M2]  # output vehicles

        for case in range(len(info_list)):
            for p_info in info_list[case]:  # p: platoon
                p_id = info_list[case].index(p_info)
                p_type = p_info[0]
                p_size = p_info[1]
                route = route_list[case]
                p_departSpeed = Vehicle.v_R0 if route[0] == 'R' else Vehicle.v_des[route]  # TODO: initial speed
                p_departPos = lastDepartPos_list[case] - p_departSpeed * p_info[2]
                inner_dp = CACC.t_c * p_departSpeed + CACC.s0 + Vehicle.length if p_type == 'CAVs' else \
                    OVM.vel2spacing(p_departSpeed)
                leader_name = '%s_%d_%s_%d_L' % (route, vehID_list[case] + 1, p_type, p_id)

                for i in range(p_size):
                    vehID_list[case] += 1
                    name = '%s_%d_%s_%d_%s' % (route, vehID_list[case], p_type, p_id, i + 1)
                    name = leader_name if i == 0 else name
                    departPos = p_departPos - i * inner_dp
                    front_veh = vehs_list[case].get(list(vehs_list[case].keys())[-1]) if i != 0 else None
                    leader_veh = vehs_list[case][leader_name] if i != 0 else None
                    typeV = 'HDV' if p_type == 'HDVs' and i != 0 else 'CAV'
                    laneID = lane_list[case]
                    laneID = -1 if laneID == 0 and departPos < 0 else laneID
                    veh = Vehicle(name, typeV, route, laneID=laneID, depart=0, departPos=departPos,
                                  departSpeed=p_departSpeed,
                                  typeP=p_type, sizeP=p_size, front=front_veh, leader=leader_veh)

                    if i != 0:
                        vehs_list[case][leader_name].followers.append(veh)
                    vehs_list[case][name] = veh
                    lastDepartPos_list[case] = departPos

        vehs_All = dict(self.vehs_R0, **self.vehs_M1, **self.vehs_M2)
        return self.vehs_R0, self.vehs_M1, self.vehs_M2, vehs_All


def get_route_example_0():
    """ example route for schedule_multiLanes.py and vehNetwork.py """
    R0_info = [['HDVs', 3, 15],
               ['CAVs', 1, 3],
               ['HDVs', 2, 4],
               ['HDVs', 2, 3.5]]
    M1_info = [['CAVs', 3, 7],
               ['HDVs', 2, 3],
               ['CAVs', 1, 3],
               ['CAVs', 4, 2.5],
               ['CAVs', 2, 2.5],
               ['HDVs', 3, 2.8]]
    M2_info = [['CAVs', 2, 5],
               ['CAVs', 3, 3],
               ['HDVs', 2, 4],
               ['CAVs', 1, 3],
               ['HDVs', 2, 3]]

    route = Route_staticInput(R0_info, M1_info, M2_info)

    vehs_R0, vehs_M1, vehs_M2, vehs_All = route.generate_vehs()
    return vehs_R0, vehs_M1, vehs_M2, vehs_All


if __name__ == '__main__':
    """ generate the simulation scenario """
    r = Road()
    r.x_lim = [-600, 50]
    r.plot(FLAG_fillRoad=True)

    """ generate default route """
    route = Route_staticInput()
    route.generate_vehs()

    """ show vehs on the scenario """
    for vehs_group in [route.vehs_R0, route.vehs_M1, route.vehs_M2]:
        for veh in vehs_group.values():
            veh.plot(r.ax_road, FLAG_speedColor=False)

    """ show single vehicle with LC trajs """
    from utilities.plotResults import show_LCTraj_demo
    veh_M1 = route.vehs_M1.get(list(route.vehs_M1.keys())[2])
    show_LCTraj_demo(r.ax_road, veh_M1, duration_list=[6.5, 5, 3.5])

    # plt.savefig('../../03 Data saving/routes/' + 'static_1.svg', dpi=600)
    plt.show()
