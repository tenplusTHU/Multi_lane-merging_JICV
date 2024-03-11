"""
    Description: Define the properties and functions for Vehicles
    Author: Tenplus
    Create-time: 2022-02-10
    Update-time: 2023-02-15  # V3.3, 准备 TIV
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from objects.Road import Road

""" Global Parameters  """
veh_length = 4
veh_width = 1.5

v_max = 25  # m/s, used in OVM
v_M2 = 25
v_M1 = 20
v_R0 = 10

v_min_M2 = 12  # m/s
v_min_M1 = 9
v_min_R0 = 4

LC_duration = 4  # s, for vehs_M1_LC and vehs_R0


class CACC:
    """
        PF topology, linear cloudController:
        u_i = kp_p * (s_i - t_c * v_i - s0) + kv_p * (v_p - v_i) + kv_l * (v_l - v_i)
    """

    s0 = 2  # m, minimal headway distance
    t_c = 0.6  # s, time headway
    kp_2p = 0.45  # kp to proceeding vehicle
    kv_2p = 0.50  # kv to proceeding vehicle
    kv_2l = 0.50  # kv to leading vehicle  # TODO: adjust here

    def __init__(self, veh=None):
        self.veh = veh

    def get_next_a(self, self_state, front_state, leader_state=None):  # states: [t, a, v, p]
        next_a = self.kp_2p * (front_state[3] - self_state[3] - veh_length - self.s0 - self.t_c * front_state[2]) + \
                 self.kv_2p * (front_state[2] - self_state[2]) + self.kv_2l * (leader_state[2] - self_state[2])
        next_a = 0 if abs(next_a) < 1e-4 else next_a
        return next_a

    @staticmethod
    def get_dt_inner(velocity):
        dt_inner = (CACC.t_c * velocity + CACC.s0 + veh_length) / velocity
        return dt_inner


class OVM:
    """
        OVM for HDV.
    """
    alpha = 0.6  # velocity sensitivity parameter
    beta = 0.9  # relative velocity sensitivity parameter
    s_st = 5  # m, safety distance when standing
    s_go = 25  # m, safety distance at maximal velocity

    def __init__(self, veh=None):
        self.veh = veh

    def get_next_a(self, self_state, front_state, leader_state=None):
        spacing = front_state[3] - self_state[3] - veh_length
        if self.s_st <= spacing <= self.s_go:
            v_desired = v_max / 2 * (1 - np.cos(np.pi * (spacing - self.s_st) / (self.s_go - self.s_st)))
        elif spacing < self.s_st:
            v_desired = 0
        else:
            v_desired = v_max

        next_a = self.alpha * (v_desired - self_state[2]) + self.beta * (front_state[2] - self_state[2])
        return next_a

    @staticmethod
    def vel2spacing(vel):  # stable, used when generate routes
        s = np.arccos(1 - 2 * vel / v_max) / np.pi * (OVM.s_go - OVM.s_st) + OVM.s_st + veh_length
        return s

    @staticmethod
    def get_dt_inner(velocity):
        dt_inner = OVM.vel2spacing(velocity) / velocity
        return dt_inner


class Vehicle:
    """ basic parameters """
    length = veh_length
    width = veh_width

    v_R0 = v_R0  # Initial ramp speed
    v_des = {'R0': v_M1, 'M1': v_M1, 'M2': v_M2}
    v_min = {'R0': v_min_R0, 'M1': v_min_M1, 'M2': v_min_M2}

    color_dict = {'R0': 'r', 'M1': 'b', 'M2': 'purple'}
    color_map = plt.cm.get_cmap(name='brg')
    color_range = [0.5, 0.9]  # corresponding to v: [0, v_max]

    LC_duration = LC_duration

    def __init__(self, name='default', typeV='CAV', route='M1', color=None, laneID=-np.inf,
                 depart=0, departPos=Road.RAMP_STRAIGHT_RANGE[0], departSpeed=v_M1, departAcc=0,
                 typeP='CAVs', sizeP=1, front=None, leader=None):
        self.name = name
        self.typeV = typeV
        self.route = route
        self.v_des = self.v_des[route]  # change with route
        self.v_min = self.v_min[route]

        if color:
            self.color = color
        else:
            color = self.color_dict[route]
            self.color = 'gray' if self.typeV == 'HDV' else color

        self.typeP = typeP
        self.sizeP = sizeP
        self.CFM = CACC(self) if self.typeP == 'CAVs' else OVM(self)

        self.front = front
        self.leader = leader
        self.followers = []

        self.depart = depart  # depart time
        """ longitudinal information """
        self.time_c = depart  # current time
        self.long_p = departPos
        self.long_v = departSpeed
        self.long_a = departAcc

        self.long_statePhase = None
        self.long_tfChosen = -np.inf
        self.long_trajPlan = None  # [list: [t, a, v, p]]

        """ lateral information """
        self.laneID = laneID
        self.lat_p = (laneID - 1 / 2) * Road.LANE_WIDTH
        self.lat_v = 0
        self.lat_a = 0
        self.yaw = 0

        self.lat_trajPlan = None  # [list: [t, Y_a, Y_v, Y_p]]
        self.LC_midT = None  # middle time of the LC process

        """ states summary """  # [t, long_p, long_v, long_a, lat_p, lat_v, lat_a, yaw, route]
        self.states_current = [depart, departPos, departSpeed, departAcc,
                               self.lat_p, self.lat_v, self.lat_a, self.yaw]
        self.states_log = []
        
        self.orderID_M1, self.orderID_M2 = 0, 0
        """ other FLAGs """
        self.FLAG_LC_M1 = False

    def plot(self, ax, FLAG_speedColor=False):
        if self.name[0] == 'R' and self.long_p < 0:
            x, y, yaw = Road.dis2XY_ramp(self.long_p)  # x, y: geodetic coordinate system; yaw: degree
        else:
            x, y, yaw = self.long_p, self.lat_p, self.yaw

        if FLAG_speedColor:
            norm_v = np.interp(self.long_v, [0, v_M2], self.color_range)
            color = self.color_map(norm_v)
        else:
            color = self.color

        veh = Rectangle((x, (y - self.width / 2) * Road.Y_scale), -self.length, self.width * Road.Y_scale,
                        angle=yaw * Road.Y_scale, fc=color, ec=color, lw=2)  # Note: Y_scale, including angle
        veh.set_zorder(9)

        if ax:
            ax.add_patch(veh)

    def get_followerTrajectories(self, v_des):
        """ extend self trajectory for followers by CFM """
        vehL_trajExtend = self.extend_longTrajectory(self, v_des=v_des)

        for veh in self.followers:
            front_traj = vehL_trajExtend if veh.front.name == self.name else veh.front.long_trajPlan
            t, a_front, v_front, p_front = zip(*front_traj)
            veh_stateNow = [veh.time_c, veh.long_a, veh.long_v, veh.long_p]
            front_stateNow = [veh.front.time_c, veh.front.long_a, veh.front.long_v, veh.front.long_p]
            leader_stateNow = [veh.leader.time_c, veh.leader.long_a, veh.leader.long_v, veh.leader.long_p]

            a_follower = [veh.CFM.get_next_a(veh_stateNow, front_stateNow, leader_stateNow)]
            v_follower = [veh.long_v + a_follower[-1] * 0.1]
            p_follower = [veh.long_p + np.mean([veh.long_v, v_follower[-1]]) * 0.1]

            for i in range(len(front_traj) - 1):
                a_ = veh.CFM.get_next_a([t[i], a_follower[-1], v_follower[-1], p_follower[-1]],
                                        front_traj[i], vehL_trajExtend[i])
                v_ = v_follower[-1] + a_ * 0.1
                p_ = p_follower[-1] + (v_ + v_follower[-1]) / 2 * 0.1
                a_ = 0 if abs(a_) < 1e-4 else a_
                a_follower.append(a_)
                v_follower.append(v_)
                p_follower.append(p_)

            veh.long_trajPlan = list(zip(t, a_follower, v_follower, p_follower))

    @staticmethod
    def extend_longTrajectory(veh, v_des):
        if veh.long_trajPlan:
            t, planned_a, planned_v, planned_p = zip(*veh.long_trajPlan)
        else:
            t, planned_a, planned_v, planned_p = [], [], [], []

        t_ex = 2 * (veh.sizeP - 1)  # 2s for the inner dt

        t_ = np.arange(1, t_ex * 10 + 1) / 10
        planned_a = np.append(planned_a, np.zeros(np.shape(t_)))
        planned_v = np.append(planned_v, np.ones(np.shape(t_)) * v_des)
        last_p = planned_p[-1] if planned_p else veh.long_p
        planned_p = np.append(planned_p, last_p + t_ * v_des)
        t = np.append(t, t[-1] + t_)

        return list(zip(t, planned_a, planned_v, planned_p))

    def get_LC_Trajectories(self, lat_p0, lat_pf, LC_duration=LC_duration, t0_LC='default'):
        """ plan self LC trajectory first """
        if t0_LC == 'default':
            pf_LC_done = max(self.long_statePhase.intersection[1], LC_duration * self.v_des + 5)  # s0: 5 m
            tf_LC_done = (pf_LC_done - self.long_statePhase.intersection[1]) / self.v_des + \
                         self.long_statePhase.intersection[0]
            t0_LC = int((tf_LC_done - LC_duration) * 10) / 10

        dt_inner = self.CFM.get_dt_inner(velocity=self.v_des)
        self.lat_trajPlan = self.lateralPlan_quinticPoly(t0=t0_LC, duration=LC_duration, lat_p0=lat_p0, lat_pf=lat_pf)

        """ plan LC trajs for followers """
        for veh in self.followers:
            t0_LC_F = t0_LC + dt_inner * (int(veh.name[-1]) - 1)  # debug, 2023-02-15
            t0_LC_F = int(t0_LC_F * 10) / 10
            veh.lat_trajPlan = veh.lateralPlan_quinticPoly(t0=t0_LC_F, duration=LC_duration,
                                                           lat_p0=lat_p0, lat_pf=lat_pf)

    @staticmethod
    def lateralPlan_quinticPoly(t0=0.0, duration=4.0, lat_p0=0.5 * Road.LANE_WIDTH, lat_v0=0, lat_a0=0,
                                lat_pf=1.5 * Road.LANE_WIDTH, lat_vf=0, lat_af=0):
        t1 = t0 + duration
        Y_TM = np.array([[t0 ** 5, t0 ** 4, t0 ** 3, t0 ** 2, t0 ** 1, 1],
                         [5 * t0 ** 4, 4 * t0 ** 3, 3 * t0 ** 2, 2 * t0 ** 1, 1, 0],
                         [20 * t0 ** 3, 12 * t0 ** 2, 6 * t0, 2, 0, 0],
                         [t1 ** 5, t1 ** 4, t1 ** 3, t1 ** 2, t1 ** 1, 1],
                         [5 * t1 ** 4, 4 * t1 ** 3, 3 * t1 ** 2, 2 * t1 ** 1, 1, 0],
                         [20 * t1 ** 3, 12 * t1 ** 2, 6 * t1, 2, 0, 0]]
                        )  # the reconstructed time matrix
        Y0 = np.array([[lat_p0, lat_v0, lat_a0, lat_pf, lat_vf, lat_af]]).T
        B_TM = np.dot(np.linalg.inv(Y_TM), Y0)

        t = t0 + np.arange(1, int(duration * 10) + 1) / 10
        TM_p = np.array([[t_ ** 5, t_ ** 4, t_ ** 3, t_ ** 2, t_, 1] for t_ in t])
        Y_p = np.dot(TM_p, B_TM)[:, 0]
        TM_v = np.array([[5 * t_ ** 4, 4 * t_ ** 3, 3 * t_ ** 2, 2 * t_, 1, 0] for t_ in t])
        Y_v = np.dot(TM_v, B_TM)[:, 0]
        TM_a = np.array([[20 * t_ ** 3, 12 * t_ ** 2, 6 * t_, 2, 0, 0] for t_ in t])
        Y_a = np.dot(TM_a, B_TM)[:, 0]

        lat_trajPlan = list(zip(t, Y_a, Y_v, Y_p))
        return lat_trajPlan


if __name__ == '__main__':
    r = Road()
    r.plot()

    """ create a new vehicle """
    veh = Vehicle(name='R1', typeV='CAV', route='R0', laneID=-1, departPos=-100, typeP='HDVs', color=None)
    veh.plot(r.ax_road)

    plt.show()
