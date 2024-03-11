"""
    Description: Functions to draw the figures of Simulations Scenarios ..
    Author: Tenplus
    Create-time: 2022-02-21
    Update-time: 2023-02-22, # V1.0 新学期 Check 代码
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from objects.Road import Road

import sys
import math
from matplotlib.patches import Rectangle


def show_scenario_simu(simu, FLAG_speedColor=False):
    """ show road """
    for ax in simu.fig_scene.get_axes():
        ax.remove()

    ax_road = simu.fig_scene.add_subplot(1, 1, 1, label='road')
    simu.road.plot(ax_road)
    ax_road.tick_params(axis='x', labelsize=10)
    ax_road.text(100, -50, f't = %.1f s' % (simu.step / 10), size=15)
    ax_road.set_title('Multi-lane Highway On-ramp Merging Scenario', fontsize=13)

    """ show vehicle on in the Map """
    vehs_plot = [simu.vehs_All[veh_name] for veh_name in simu.vehs_InMap]

    for veh in vehs_plot:
        x, y, yaw = veh.long_p, veh.lat_p, veh.yaw

        if veh.name[0] == 'R' and veh.long_p <= 0:
            x, y, yaw = Road.dis2XY_ramp(veh.long_p)
            yaw *= Road.Y_scale

        if FLAG_speedColor:
            from objects.Vehicle import Vehicle
            color_range = [0.5, 0.9]
            speed_range = [Vehicle.v_min['R0'], Vehicle.v_des['M2']]
            norm_v = np.interp(veh.long_v, speed_range, color_range)
            color = veh.color_map(norm_v)
        else:
            color = veh.color

        veh_plot = Rectangle((x, (y - veh.width / 2) * Road.Y_scale), -veh.length, veh.width * Road.Y_scale,
                             angle=yaw, fc=color, ec=color, lw=2)
        veh_plot.set_order = 12
        ax_road.add_patch(veh_plot)

    plt.tight_layout()
    return ax_road