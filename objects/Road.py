"""
    Description: Define the Roads for multi-lane on-ramp merging scenario.
    Author: Tenplus
    Create-time: 2022-03-29
    Update-time: 2022-12-04, V1.2: 细化道路细节
"""

import numpy as np
import matplotlib.pyplot as plt

""" Global Parameters  """
# longitudinal
MAIN_RANGE = [-600, 300]
M2_SCH_RANGE = [-600, 0]  # debug, 2022-12-19
M1_SCH_RANGE = [-500, 0]
R0_SCH_RANGE = [-500, 0]

MERGE_ZONE_LENGTH = 200
RAMP_LIMITED = -150  # TODO: adjust here

# lateral
LANE_WIDTH = 4
Y_scale = 3.8


class Road:
    """ load parameters """
    # longitudinal
    MAIN_RANGE = MAIN_RANGE
    M2_SCH_RANGE = M2_SCH_RANGE
    M1_SCH_RANGE = M1_SCH_RANGE
    R0_SCH_RANGE = R0_SCH_RANGE

    MERGE_ZONE_LENGTH = MERGE_ZONE_LENGTH
    RAMP_LIMITED = RAMP_LIMITED

    # lateral
    MAIN_LANES_N = 2
    LANE_WIDTH = LANE_WIDTH
    Y_scale = Y_scale

    """ set parameters """
    RAMP_R = 1000
    RAMP_ANGLE = 6  # degree
    RAMP_RAD = RAMP_ANGLE / 180 * np.pi
    RAMP_STRAIGHT_RANGE = [R0_SCH_RANGE[0], - RAMP_R * np.sin(RAMP_RAD) * 2]

    def __init__(self):
        self.ax_road = None

        ''' axes range setting '''
        self.x_lim = self.MAIN_RANGE
        yLim_up = (self.MAIN_LANES_N + 1) * self.LANE_WIDTH * Y_scale
        yLim_down = - 4 * self.RAMP_R * (1 - np.cos(self.RAMP_RAD)) * Y_scale
        self.y_lim = [yLim_down, yLim_up]

    def plot(self, ax=None, FLAG_fillRoad=True, fig_size=(11, 2.2)):
        if ax is not None:
            self.ax_road = ax
        else:
            fig = plt.figure(figsize=fig_size)
            self.ax_road = fig.add_subplot(label='road')

        self.ax_road.set_aspect(1)  # set the scale of x and y axes
        self.ax_road.set_xlim(self.x_lim)
        self.ax_road.set_ylim(self.y_lim)
        self.ax_road.spines['right'].set_visible(False)
        self.ax_road.spines['left'].set_visible(False)
        self.ax_road.spines['top'].set_visible(False)
        self.ax_road.set_yticks([])

        lw_bound = 1.5
        """ Draw the main road """
        self.ax_road.hlines(self.MAIN_LANES_N * self.LANE_WIDTH * self.Y_scale, self.MAIN_RANGE[0], self.MAIN_RANGE[1],
                            colors='k', lw=lw_bound, zorder=0)
        self.ax_road.hlines(0, self.MAIN_RANGE[0], 0, colors='k', lw=lw_bound, zorder=0)
        smooth_x = 30
        self.ax_road.hlines(0, self.MERGE_ZONE_LENGTH + smooth_x, self.MAIN_RANGE[1], colors='k', lw=lw_bound, zorder=0)
        self.ax_road.hlines(-self.LANE_WIDTH * self.Y_scale, 0, self.MERGE_ZONE_LENGTH,
                            colors='k', lw=lw_bound, zorder=0)
        connect_x = np.linspace(self.MERGE_ZONE_LENGTH, self.MERGE_ZONE_LENGTH + smooth_x, 100, endpoint=True)
        connect_y = - self.LANE_WIDTH / 2 * np.cos(
            np.pi * (connect_x - self.MERGE_ZONE_LENGTH) / smooth_x) - 1 / 2 * self.LANE_WIDTH
        self.ax_road.plot(connect_x, connect_y * self.Y_scale, lw=lw_bound, color='k', zorder=0)

        ''' Draw the ramp road '''
        curve_upper = self.__curveLane_upper()
        curve_lower = self.__curveLane_lower()
        self.ax_road.plot(curve_upper[0], curve_upper[1] * self.Y_scale, lw=lw_bound, color='k', zorder=0)
        self.ax_road.plot(curve_lower[0], curve_lower[1] * self.Y_scale, lw=lw_bound, color='k', zorder=0)
        self.ax_road.hlines(np.array([curve_upper[1][0], curve_lower[1][0]]) * self.Y_scale,
                            self.RAMP_STRAIGHT_RANGE[0],
                            self.RAMP_STRAIGHT_RANGE[1], colors='k', lw=lw_bound, zorder=0)

        """ Draw lanes """
        lw_lanes = 1
        line1, = self.ax_road.plot(self.MAIN_RANGE, [self.LANE_WIDTH * self.Y_scale] * 2,
                                   lw=lw_lanes, c='grey', ls='-', zorder=0)
        line2, = self.ax_road.plot([10, self.MERGE_ZONE_LENGTH + smooth_x / 2], [0] * 2,
                                   lw=lw_lanes, color='grey', linestyle='-', zorder=0)
        line1.set_dashes((10, 10))
        line2.set_dashes((10, 10))

        """ draw schedule zones """
        if FLAG_fillRoad:
            self.ax_road.fill_between(self.M1_SCH_RANGE, 0, self.Y_scale * self.LANE_WIDTH * self.MAIN_LANES_N,
                                      fc='gray', alpha=.1, zorder=0, ec='w')
            self.ax_road.fill_between([self.M2_SCH_RANGE[0], self.M1_SCH_RANGE[0]], self.Y_scale * self.LANE_WIDTH,
                                      2 * self.Y_scale * self.LANE_WIDTH, fc='gray', alpha=.1, zorder=0, ec='w')
            self.ax_road.fill_between([self.M2_SCH_RANGE[0], self.M1_SCH_RANGE[0]], 0, self.Y_scale * self.LANE_WIDTH,
                                      fc='gray', alpha=.05, zorder=0, ec='w')

            curve_x = np.insert(curve_lower[0], 0, self.RAMP_STRAIGHT_RANGE[0])
            curve_y_down = np.insert(curve_lower[1], 0, curve_lower[1][0]) * self.Y_scale
            curve_y_up = np.interp(curve_x, curve_upper[0], curve_upper[1] * self.Y_scale)
            self.ax_road.fill_between(curve_x, curve_y_down, curve_y_up, fc='gray', alpha=.1, zorder=0, ec='w')

            connect_x = np.insert(connect_x, 0, 0)
            connect_y = np.insert(connect_y, 0, connect_y[0])
            self.ax_road.fill_between(connect_x, connect_y * self.Y_scale, self.LANE_WIDTH * self.Y_scale * 2,
                                      fc='r', ec='w', alpha=.1, zorder=0)

            self.ax_road.fill_between([connect_x[-1], self.MAIN_RANGE[-1]], 0, self.LANE_WIDTH * self.Y_scale * 2,
                                      fc='g', ec='w', alpha=.1, zorder=0)

        plt.tight_layout()

        return self.ax_road

    def __curveLane_upper(self):
        theta = np.linspace(0, self.RAMP_RAD, 100)
        # 斜率平滑切换的点，middle_point
        middleP_x = -self.RAMP_R * np.sin(self.RAMP_RAD)
        middleP_y = -self.RAMP_R + self.RAMP_R * np.cos(self.RAMP_RAD) - self.LANE_WIDTH / 2

        x_right = -(self.RAMP_R + self.LANE_WIDTH / 2) * np.sin(theta)
        y_right = -self.RAMP_R + (self.RAMP_R + self.LANE_WIDTH / 2) * np.cos(theta) - self.LANE_WIDTH / 2
        x_right = np.flipud(x_right)
        y_right = np.flipud(y_right)

        x_left = (self.RAMP_R - self.LANE_WIDTH / 2) * np.sin(theta) + 2 * middleP_x
        y_left = - (self.RAMP_R - self.LANE_WIDTH / 2) * np.cos(
            theta) + self.RAMP_R + self.LANE_WIDTH / 2 + 2 * middleP_y

        x = x_left.tolist() + x_right.tolist()
        y = y_left.tolist() + y_right.tolist()

        return np.array(x), np.array(y)

    def __curveLane_lower(self):
        self.LANE_WIDTH = self.LANE_WIDTH  # no use
        theta = np.linspace(0, self.RAMP_RAD, 100)
        middleP_x = -self.RAMP_R * np.sin(self.RAMP_RAD)
        middleP_y = -self.RAMP_R + self.RAMP_R * np.cos(self.RAMP_RAD) - self.LANE_WIDTH / 2

        x_right = -(self.RAMP_R - self.LANE_WIDTH / 2) * np.sin(theta)
        y_right = -self.RAMP_R + (self.RAMP_R - self.LANE_WIDTH / 2) * np.cos(theta) - self.LANE_WIDTH / 2
        x_right = np.flipud(x_right)
        y_right = np.flipud(y_right)

        x_left = (self.RAMP_R + self.LANE_WIDTH / 2) * np.sin(theta) + 2 * middleP_x
        y_left = - (self.RAMP_R + self.LANE_WIDTH / 2) * np.cos(
            theta) + self.RAMP_R + self.LANE_WIDTH / 2 + 2 * middleP_y

        x = x_left.tolist() + x_right.tolist()
        y = y_left.tolist() + y_right.tolist()

        return np.array(x), np.array(y)

    @staticmethod
    def dis2XY_ramp(distance):
        if distance > 0:
            x, y, yaw = distance, -Road.LANE_WIDTH / 2, 0
        else:
            distance = -distance
            theta_rad = distance / Road.RAMP_R
            theta_angle = theta_rad / np.pi * 180
            if theta_rad <= Road.RAMP_RAD:
                yaw = theta_angle
                x = - Road.RAMP_R * np.sin(theta_rad)
                y = Road.RAMP_R * np.cos(theta_rad) - Road.RAMP_R - Road.LANE_WIDTH / 2
            elif theta_rad <= 2 * Road.RAMP_RAD:
                yaw = 2 * Road.RAMP_ANGLE - theta_angle
                middleP_x = - Road.RAMP_R * np.sin(Road.RAMP_RAD)
                middleP_y = Road.RAMP_R * np.cos(Road.RAMP_RAD) - Road.RAMP_R - Road.LANE_WIDTH / 2
                x = Road.RAMP_R * np.sin(2 * Road.RAMP_RAD - theta_rad) + 2 * middleP_x
                y = - Road.RAMP_R * np.cos(
                    2 * Road.RAMP_RAD - theta_rad) + 2 * middleP_y + Road.RAMP_R + Road.LANE_WIDTH / 2
            else:
                yaw = 0
                x = -2 * Road.RAMP_R * np.sin(Road.RAMP_RAD) - (distance - Road.RAMP_R * 2 * Road.RAMP_RAD)
                y = 2 * (-Road.RAMP_R + Road.RAMP_R * np.cos(Road.RAMP_RAD)) - Road.LANE_WIDTH / 2
        return x, y, yaw


if __name__ == "__main__":
    r = Road()
    # r.set_xlim([-400, 200])
    r.plot()

    plt.show()
