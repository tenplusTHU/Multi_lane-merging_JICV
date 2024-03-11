"""
    Description: Functions to draw the figures of vehicle trajectory..
    Author: Tenplus
    Create-time: 2022-02-21
    Update-time: 2023-02-15, # V1.2 新学期 Check 代码
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from objects.Road import Road

import sys
import math
from matplotlib.patches import Rectangle


def show_severalTrajs(Trajs: dict):
    if Trajs:
        """ fig_scene: basic setting """
        fig = plt.figure(figsize=(10, 2.3))
        font_label = {'size': 11}
        ax1 = fig.add_subplot(141)
        ax2 = fig.add_subplot(142)
        ax3 = fig.add_subplot(143)
        ax4 = fig.add_subplot(144)

        for ax_ in [ax1, ax2, ax3]:
            ax_.set_xlabel(r'$t~\mathrm{(s)}$', font_label)

        ax1.set_ylabel(r'$p~\mathrm{(m)}$', font_label)
        ax2.set_ylabel(r'$v~\mathrm{(m/s)}$', font_label)
        ax3.set_ylabel(r'$a~\mathrm{(m/s^2)}$', font_label)

        ax4.set_xlabel(r'$a~\mathrm{(m/s^2)}$', font_label)
        ax4.set_ylabel(r'$v~\mathrm{(m/s)}$', font_label)

        """ plot lines """
        for color, traj in Trajs.items():
            if traj:
                t, a, v, p = zip(*traj)
                lw = 2

                ax1.plot(t, p, lw=lw, c=color)
                ax2.plot(t, v, lw=lw, c=color)
                ax3.plot(t, a, lw=lw, c=color)
                ax4.plot(a, v, lw=lw, c=color)

        plt.tight_layout()


def show_trajs_threeLanes(vehs_all):
    fig_trajs = plt.figure(figsize=(10, 5.5))

    """ axes setting """
    font_label = {'size': 12}
    ax1 = fig_trajs.add_subplot(2, 3, 1)  # position, R0-M1
    ax2 = fig_trajs.add_subplot(2, 3, 2)  # position, M1-M2
    ax3 = fig_trajs.add_subplot(2, 3, 4)  # velocity ~ t
    ax4 = fig_trajs.add_subplot(2, 3, 5)  # acceleration ~ t
    ax5 = fig_trajs.add_subplot(2, 3, 6)  # velocity ~ acceleration

    for ax_ in [ax1, ax2, ax3, ax4, ax5]:
        ax_.tick_params(axis='x', labelsize=11)
        ax_.tick_params(axis='y', labelsize=11)

    for ax_ in [ax1, ax2]:
        ax_.set_ylim([-500, 200])

    for ax_ in [ax1, ax2, ax3, ax4]:
        ax_.set_xlabel(r'$t~\mathrm{(s)}$', font_label)
        ax_.set_xlim([0, 40])

    ax1.set_ylabel(r'$Lane~M1: p~\mathrm{(m)}$', font_label)
    ax2.set_ylabel(r'$Lane~M2: p~\mathrm{(m)}$', font_label)
    ax3.set_ylabel(r'$v~\mathrm{(m/s)}$', font_label)
    ax4.set_ylabel(r'$a~\mathrm{(m/s^2)}$', font_label)

    ax5.set_xlabel(r'$a~\mathrm{(m/s^2)}$', font_label)
    ax5.set_ylabel(r'$v~\mathrm{(m/s)}$', font_label)

    """ lanes setting """
    Orders = {'r': 9, 'b': 8, 'purple': 7, 'gray': 0}
    LineStyles = {'CAVs': '-.', 'HDVs': '--'}

    """ plot trajs """
    for veh in vehs_all.values():
        lw = 1.5 if veh.name[-1] == 'L' else 1
        ls = LineStyles[veh.typeP] if veh.name[-1] != 'L' else '-'
        alpha = 1 if veh.name[-1] == 'L' else 0.7
        order = Orders[veh.color]

        t, long_a, long_v, long_p = zip(*veh.long_trajPlan)
        # velocity and acceleration
        ax3.plot(t, long_v, veh.color, lw=lw, ls=ls, zorder=order, alpha=alpha)
        ax4.plot(t, long_a, veh.color, lw=lw, ls=ls, zorder=order, alpha=alpha)
        if veh.name[-1] == 'L':
            ax5.plot(long_a, long_v, veh.color, lw=lw, ls=ls, zorder=order, alpha=alpha)

        # position
        if veh.name[0:2] == 'R0':
            LC_end = veh.lat_trajPlan[-1][0]
            LC_end = int(LC_end * 10) / 10
            LC_end_index = np.where(np.array(t) <= LC_end)[0][-1]

            ax1.plot(t[LC_end_index:], long_p[LC_end_index:], veh.color, lw=lw, ls=ls, zorder=order, alpha=alpha)
            ax1.plot(t[:LC_end_index], long_p[:LC_end_index], veh.color, lw=lw, ls=ls, zorder=order, alpha=alpha * .6)
            ax1.plot(t[LC_end_index], long_p[LC_end_index], color=veh.color, marker='o', markersize=3)

        elif veh.name[0:2] == 'M2':
            ax2.plot(t, long_p, veh.color, lw=lw, ls=ls, zorder=order, alpha=alpha)
        elif veh.name[0:2] == 'M1':
            if not veh.lat_trajPlan:
                ax1.plot(t, long_p, veh.color, lw=lw, ls=ls, zorder=order, alpha=alpha)
            else:
                LC_end = veh.lat_trajPlan[-1][0]
                LC_end = int(LC_end * 10) / 10
                LC_end_index = np.where(np.array(t) <= LC_end)[0][-1]

                ax2.plot(t[LC_end_index:], long_p[LC_end_index:], veh.color, lw=lw, ls=ls, zorder=order, alpha=alpha)
                ax2.plot(t[:LC_end_index], long_p[:LC_end_index], veh.color, lw=lw, ls=ls, zorder=order,
                         alpha=alpha * .6)
                ax2.plot(t[LC_end_index], long_p[LC_end_index], color=veh.color, marker='o', markersize=3)

    plt.tight_layout()


def show_LCTraj_demo(ax, veh, duration_list=[2, 4], c=None):
    """ used in the demo plotting """
    color = c if c else veh.color
    alpha_list = np.linspace(0.3, 1, len(duration_list), endpoint=True) if len(duration_list) > 1 else [1]
    lat_p0 = - 0.5 * Road.LANE_WIDTH if veh.route == 'R0' else 0.5 * Road.LANE_WIDTH
    lat_pf = 0.5 * Road.LANE_WIDTH if veh.route == 'R0' else 1.5 * Road.LANE_WIDTH
    for duration in duration_list:
        lat_plan = veh.lateralPlan_quinticPoly(duration=duration, lat_p0=lat_p0, lat_pf=lat_pf)
        t = np.array([veh.depart] + [k[0] for k in lat_plan])
        lat_p = np.array([veh.lat_p] + [k[3] for k in lat_plan])
        long_p = (t - veh.depart) * veh.long_v + veh.long_p  # TODO: longitudinal, constant speed
        ax.plot(long_p, lat_p * Road.Y_scale, '--', c=color, alpha=alpha_list[duration_list.index(duration)])
