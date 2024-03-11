"""
    Description: Bang-off-bang controller, analytical profiles.
    Author: Tenplus
    Create-time: 2022-03-24
"""

import sys
import random
import numpy as np
from scipy.optimize import fsolve
from objects.Vehicle import Vehicle
from utilities.plotResults import show_severalTrajs
import matplotlib.pyplot as plt

""" Global Parameters  """
j_min = -0.6  # m/s^3, the minimal jerk
j_max = 0.4  # m/s^3, the maximal jerk


class Stage2_max2min:
    def __init__(self, bbc):
        self.bbc = bbc
        self.t0, self.p0, self.v0, self.a0 = bbc.t0, bbc.p0, bbc.v0, bbc.a0

    def fun(self, X):
        t1 = X[0]
        t2 = X[1]
        pf = X[2]

        eq1 = self.a0 + j_max * t1 + j_min * t2
        eq2 = self.v0 + (j_min * t2 ** 2) / 2 - (j_max * t1 ** 2) / 2 + (t1 + t2) * (
                self.a0 + j_max * t1) - self.bbc.v_des
        eq3 = self.p0 + (t1 + t2) * (self.v0 - (j_max * t1 ** 2) / 2) + (t1 + t2) ** 2 * (
                self.a0 / 2 + (j_max * t1) / 2) + (j_min * t2 ** 3) / 6 + (j_max * t1 ** 3) / 6 - pf

        return [eq1, eq2, eq3]

    def get_element(self):
        if self.bbc.stage0_FLAG:
            return 0, 0, self.p0
        elif self.bbc.stage1_FLAG:
            return 0, self.bbc.stage1_t_one, self.bbc.stage1_pf_one

        search_n, t1, t2 = 0, -1, -1
        while not (t1 > 0 and t2 > 0) and (search_n < 100):
            random.seed(search_n)
            seed = [random.randint(1, 20), random.randint(1, 20), random.randint(10, 1000)]
            t1, t2, pf = fsolve(self.fun, seed)
            search_n += 1
        if t1 > 0 and t2 > 0:
            return t1, t2, pf
        else:
            print(f'*** error at get_element in Stage2_max2min')
            print(f'*** (t0, p0, v0, a0): {self.t0, self.p0, self.v0, self.a0}')
            print(f'*** invalid (t1, t2): {t1, t2}')
            sys.exit()

    def get_traj(self):
        t1, t2, pf = self.get_element()
        planned_a, planned_v, planned_p = [], [], []

        t = np.arange(1, (t1 + t2) * 10 + 1) / 10
        for t_ in t:
            if t_ < t1:
                a_ = j_max * t_ + self.a0
                v_ = 1 / 2 * j_max * t_ ** 2 + self.a0 * t_ + self.v0
                p_ = 1 / 6 * j_max * t_ ** 3 + 1 / 2 * self.a0 * t_ ** 2 + self.v0 * t_ + self.p0
            else:
                a_ = j_min * (t_ - t1) + j_max * t1 + self.a0
                v_ = 1 / 2 * j_min * (t_ - t1) ** 2 + (j_max * t1 + self.a0) * t_ - \
                     1 / 2 * j_max * t1 ** 2 + self.v0
                p_ = 1 / 6 * j_min * (t_ - t1) ** 3 + 1 / 2 * (j_max * t1 + self.a0) * t_ ** 2 + \
                     (- 1 / 2 * j_max * t1 ** 2 + self.v0) * t_ + (j_max * t1 ** 3) / 6 + self.p0

            planned_a.append(a_)
            planned_v.append(v_)
            planned_p.append(p_)

        t = t + self.t0
        return list(zip(t, planned_a, planned_v, planned_p))


class Stage2_zero2min:
    def __init__(self, bbc):
        self.bbc = bbc
        self.t0, self.p0, self.v0, self.a0 = bbc.t0, bbc.p0, bbc.v0, bbc.a0

    def fun(self, X):
        t1 = X[0]
        t2 = X[1]
        pf = X[2]

        eq1 = self.a0 + j_min * t2
        eq2 = self.v0 + self.a0 * (t1 + t2) + (j_min * t2 ** 2) / 2 - self.bbc.v_des
        eq3 = self.p0 + self.v0 * (t1 + t2) + (self.a0 * (t1 + t2) ** 2) / 2 + j_min * t2 ** 3 / 6 - pf

        return [eq1, eq2, eq3]

    def get_element(self):
        if self.bbc.stage1_FLAG:
            return 0, self.bbc.stage1_t_one, self.bbc.stage1_pf_one

        if self.a0 > 0 and not self.bbc.stage1_FLAG:
            search_n, t1, t2 = 0, -1, -1
            while not (t1 > 0 and t2 > 0) and (search_n < 100):
                random.seed(search_n + 11)
                seed = [random.randint(1, 20), random.randint(1, 20), random.randint(10, 1000)]
                t1, t2, pf = fsolve(self.fun, seed)
                search_n += 1
            if t1 > 0 and t2 > 0:
                return t1, t2, pf
            else:
                print(f'*** error at get_element in Stage2_zero2min')
                print(f'*** (t0, p0, v0, a0): {self.t0, self.p0, self.v0, self.a0}')
                print(f'*** invalid (t1, t2): {t1, t2}')
                sys.exit()
        else:
            print(f'*** error at get_element in Stage2_zero2min')
            print(f'*** (t0, p0, v0, a0): {self.t0, self.p0, self.v0, self.a0}')
            sys.exit(2)

    def get_traj(self):
        t1, t2, pf = self.get_element()
        planned_a, planned_v, planned_p = [], [], []

        t = np.arange(1, (t1 + t2) / 0.1 + 1) / 10
        for t_ in t:
            if t_ < t1:
                a_ = self.a0
                v_ = self.a0 * t_ + self.v0
                p_ = 1 / 2 * self.a0 * t_ ** 2 + self.v0 * t_ + self.p0
            else:
                a_ = j_min * (t_ - t1) + self.a0
                v_ = 1 / 2 * j_min * (t_ - t1) ** 2 + self.a0 * t_ + self.v0
                p_ = 1 / 6 * j_min * (t_ - t1) ** 3 + 1 / 2 * self.a0 * t_ ** 2 + self.v0 * t_ + self.p0

            planned_a.append(a_)
            planned_v.append(v_)
            planned_p.append(p_)

        t = t + self.t0
        return list(zip(t, planned_a, planned_v, planned_p))


class Stage3_max2zero2min:
    def __init__(self, bbc):
        self.bbc = bbc
        self.t0, self.p0, self.v0, self.a0 = bbc.t0, bbc.p0, bbc.v0, bbc.a0

    def fun(self, X, pf):
        t1 = X[0]
        t2 = X[1]
        t3 = X[2]

        eq1 = self.a0 + j_min * t3 + j_max * t1
        eq2 = self.v0 + (self.a0 + j_max * t1) * (t1 + t2 + t3) + (j_min * t3 ** 2) / 2 - (
                j_max * t1 ** 2) / 2 - self.bbc.v_des
        eq3 = self.p0 + (self.v0 - (j_max * t1 ** 2) / 2) * (t1 + t2 + t3) + (j_min * t3 ** 3) / 6 + (
                j_max * t1 ** 3) / 6 + (self.a0 / 2 + (j_max * t1) / 2) * (t1 + t2 + t3) ** 2 - pf

        return [eq1, eq2, eq3]

    def get_element(self, pf):
        """ check the min_max bound, first """
        t1, t2, pfMin = Stage2_max2min(self.bbc).get_element()
        if abs(pf - pfMin) < 1e-4:
            return t1, 0, t2, pfMin
        elif pf <= pfMin - 1e-4:
            print(f'Error, invalid input pf {pf} at Stage3_max2zero2min, the min input_pf is {pfMin}')
            sys.exit(1)

        if self.bbc.stage1_FLAG:
            return 0, 0, self.bbc.stage1_t_one, self.bbc.stage1_pf_one

        if self.a0 > 0 and not self.bbc.stage1_FLAG:
            t1, t2, pfMax = Stage2_zero2min(self.bbc).get_element()
            if abs(pf - pfMax) < 1e-4:
                return 0, t1, t2, pfMax
            elif pf >= pfMax + 1e-4:
                print(f'Error, invalid input pf {pf} at Stage3_max2zero2min, the max input_pf is {pfMax}')
                sys.exit(2)

        """ solve the equations """
        search_n, t1, t2, t3 = 0, -1, -1, -1
        while not (t1 > 0 and t2 > 0 and t3 > 0) and (search_n < 100):
            random.seed(search_n)
            seed = [random.randint(1, 20), random.randint(1, 20), random.randint(1, 20)]
            t1, t2, t3 = fsolve(self.fun, seed, args=pf)
            search_n += 1
        if t1 > 0 and t2 > 0 and t3 > 0:
            return t1, t2, t3, pf
        else:
            print(f'*** error at get_element in Stage3_max2zero2min when pf is {pf}')
            print(f'*** (t0, p0, v0, a0): {self.t0, self.p0, self.v0, self.a0}, and v_des: {self.bbc.v_des}')
            print(f'*** invalid (t1, t2, t3): {t1, t2, t3}')
            sys.exit(3)

    def get_traj(self, pf):
        t1, t2, t3, _ = self.get_element(pf)
        planned_a, planned_v, planned_p = [], [], []

        t = np.arange(1, (t1 + t2 + t3) / 0.1 + 1) / 10
        for t_ in t:
            if t_ < t1:
                a_ = j_max * t_ + self.a0
                v_ = 1 / 2 * j_max * t_ ** 2 + self.a0 * t_ + self.v0
                p_ = 1 / 6 * j_max * t_ ** 3 + 1 / 2 * self.a0 * t_ ** 2 + self.v0 * t_ + self.p0

            elif t_ < t1 + t2:
                a_ = j_max * t1 + self.a0
                v_ = (j_max * t1 + self.a0) * t_ + (-1 / 2 * j_max * t1 ** 2 + self.v0)
                p_ = 1 / 2 * (j_max * t1 + self.a0) * t_ ** 2 + (-1 / 2 * j_max * t1 ** 2 + self.v0) * t_ + \
                     (1 / 6 * j_max * t1 ** 3 + self.p0)
            else:
                a_ = j_min * (t_ - t1 - t2) + (j_max * t1 + self.a0)
                v_ = 1 / 2 * j_min * (t_ - t1 - t2) ** 2 + (j_max * t1 + self.a0) * t_ + (
                        -1 / 2 * j_max * t1 ** 2 + self.v0)
                p_ = 1 / 6 * j_min * (t_ - t1 - t2) ** 3 + 1 / 2 * (j_max * t1 + self.a0) * t_ ** 2 + (
                        -1 / 2 * j_max * t1 ** 2 + self.v0) * t_ + (1 / 6 * j_max * t1 ** 3 + self.p0)

            planned_a.append(a_)
            planned_v.append(v_)
            planned_p.append(p_)

        t = t + self.t0
        return list(zip(t, planned_a, planned_v, planned_p))


class Stage3_min2max2min:
    def __init__(self, bbc):
        self.bbc = bbc
        self.t0, self.p0, self.v0, self.a0 = bbc.t0, bbc.p0, bbc.v0, bbc.a0

    def fun(self, X, param):
        t1 = X[0]
        t2 = X[1]
        t3 = X[2]

        [variable, value] = param
        if variable == 'pf':
            pf = value
        elif variable == 'tf':
            tf = value
            pf = X[3]
        elif variable == 'vMin_three':
            v_min = value
            pf = X[3]
        elif variable == 'vMin_two':  # debug, 2023-02-16
            v_min = value
            t1 = 0
            pf = X[0]

        eq1 = self.a0 + j_min * t1 + j_max * t2 + j_min * t3
        eq2 = self.v0 + (self.a0 + j_min * t1 + j_max * t2) * (t1 + t2 + t3) - (j_min * t1 ** 2) / 2 + (
                j_min * t3 ** 2) / 2 - (j_max * t2 ** 2) / 2 - j_max * t1 * t2 - self.bbc.v_des
        eq3 = self.p0 + (self.a0 / 2 + (j_min * t1) / 2 + (j_max * t2) / 2) * (t1 + t2 + t3) ** 2 - (
                t1 + t2 + t3) * ((j_min * t1 ** 2) / 2 + j_max * t1 * t2 + (j_max * t2 ** 2) / 2
                                 - self.v0) + (j_min * t1 ** 3) / 6 + (j_min * t3 ** 3) / 6 + (
                      j_max * t2 ** 3) / 6 + (j_max * t1 * t2 ** 2) / 2 + (j_max * t1 ** 2 * t2) / 2 - pf

        if variable == 'pf':
            return [eq1, eq2, eq3]
        elif variable == 'tf':
            eq4 = t1 + t2 + t3 + self.t0 - tf
            return [eq1, eq2, eq3, eq4]
        elif variable == 'vMin_three':
            eq4 = self.v0 + self.a0 * t1 + (j_min * t1 ** 2) / 2 - self.a0 ** 2 / (2 * j_max) - \
                  (j_min ** 2 * t1 ** 2) / (2 * j_max) - (self.a0 * j_min * t1) / j_max - v_min
            return [eq1, eq2, eq3, eq4]
        elif variable == 'vMin_two':
            return [eq1, eq2, eq3]

    def get_element(self, param):
        [variable, value] = param  # debug, here
        t1, t2, pf = Stage2_max2min(self.bbc).get_element()
        if param == ['v_min', self.v0] or (variable == 'tf' and abs(value - self.t0 - t1 - t2) < 0.1):
            return 0, t1, t2, pf  # no min stage

        # debug, 2023-02-16: v_min > vMin_th: two stage; v_min < vMin_th: three stage
        vMin_th = self.v0 + self.a0 * (- self.a0 / j_max) + 1 / 2 * j_max * (- self.a0 / j_max) ** 2

        search_n, t1, t2, t3, pf = 0, -1, -1, -1, value
        while not (t1 >= 0 and t2 > 0 and t3 > 0) and (search_n < 1000):
            random.seed(search_n + 11)
            if variable == 'pf':
                seed = [random.randint(1, 20), random.randint(1, 20), random.randint(1, 20)]
                t1, t2, t3 = fsolve(self.fun, seed, args=['pf', value])
            elif variable == 'tf':
                seed = [random.randint(1, 30), random.randint(1, 30), random.randint(1, 20), random.randint(10, 1000)]
                t1, t2, t3, pf = fsolve(self.fun, seed, args=['tf', value])
            elif variable == 'v_min':
                if value < vMin_th:  # three-stage
                    seed = [random.randint(1, 30), random.randint(1, 30), random.randint(1, 20),
                            random.randint(10, 1000)]
                    t1, t2, t3, pf = fsolve(self.fun, seed, args=['vMin_three', value])
                else:
                    seed = [random.randint(1, 20), random.randint(1, 20), random.randint(1, 20)]
                    t1 = 0
                    pf, t2, t3 = fsolve(self.fun, seed, args=['vMin_two', value])

            search_n += 1

        if t1 >= 0 and t2 >= 0 and t3 > 0:
            return t1, t2, t3, pf
        else:
            print(f'*** error at get_element in Stage3_min2max2min when param is {param}')
            print(f'*** (t0, p0, v0, a0): {self.t0, self.p0, self.v0, self.a0}')
            print(f'*** invalid (t1, t2, t3): {t1, t2, t3}')
            sys.exit()

    def get_traj(self, param):
        t1, t2, t3, pf = self.get_element(param)

        planned_a, planned_v, planned_p = [], [], []
        t = np.arange(1, (t1 + t2 + t3) / 0.1 + 1) / 10
        for t_ in t:
            if t_ < t1:
                a_ = j_min * t_ + self.a0
                v_ = 1 / 2 * j_min * t_ ** 2 + self.a0 * t_ + self.v0
                p_ = 1 / 6 * j_min * t_ ** 3 + 1 / 2 * self.a0 * t_ ** 2 + self.v0 * t_ + self.p0
            elif t_ < t1 + t2:
                a_ = j_max * (t_ - t1) + j_min * t1 + self.a0
                v_ = 1 / 2 * j_max * (t_ - t1) ** 2 + (
                        j_min * t1 + self.a0) * t_ - 1 / 2 * j_min * t1 ** 2 + self.v0
                p_ = 1 / 6 * j_max * (t_ - t1) ** 3 + 1 / 2 * (j_min * t1 + self.a0) * t_ ** 2 + (
                        - 1 / 2 * j_min * t1 ** 2 + self.v0) * t_ + (j_min * t1 ** 3) / 6 + self.p0
            else:
                a_ = j_min * (t_ - t1 - t2) + (j_min * t1 + j_max * t2 + self.a0)
                v_ = 1 / 2 * j_min * (t_ - t1 - t2) ** 2 + (j_min * t1 + j_max * t2 + self.a0) * t_ + \
                     (self.v0 - j_min * 1 / 2 * t1 ** 2 - j_max * t1 * t2 - j_max * 1 / 2 * t2 ** 2)
                p_ = 1 / 6 * j_min * (t_ - t1 - t2) ** 3 + 1 / 2 * (
                        j_min * t1 + j_max * t2 + self.a0) * t_ ** 2 + \
                     (self.v0 - j_min * 1 / 2 * t1 ** 2 -
                      j_max * t1 * t2 - j_max * 1 / 2 * t2 ** 2) * t_ + \
                     (1 / 6 * j_min * t1 ** 3 + 1 / 2 * j_max * t1 ** 2 * t2 +
                      1 / 2 * j_max * t1 * t2 ** 2 + 1 / 6 * j_max * t2 ** 3 + self.p0)

            planned_a.append(a_)
            planned_v.append(v_)
            planned_p.append(p_)

        t = t + self.t0
        return list(zip(t, planned_a, planned_v, planned_p))


class Stage5_min2max2zero2max2min:
    def __init__(self, bbc):
        self.bbc = bbc
        self.t0, self.p0, self.v0, self.a0 = bbc.t0, bbc.p0, bbc.v0, bbc.a0
        self.v_des = bbc.v_des
        self.v_min = bbc.v_min

    def get_element(self, tf=np.inf):
        t1, t2, t3, pf = Stage3_min2max2min(self.bbc).get_element(['v_min', self.v_min])
        if tf < self.t0 + t1 + t2 + t3:
            print(f'Error, invalid input_tf {tf} at Stage5_min2max2zero2max2min, '
                  f'the minimal value tf is {self.t0 + t1 + t2 + t3}, '
                  f't, p, v, a: {self.t0, self.p0, self.v0, self.t0}, v_des, v_min = {self.v_des, self.v_min}')
            sys.exit(1)
        else:
            t1 = t1
            t2 = -(self.a0 + j_min * t1) / j_max
            t4 = np.sqrt(2 * (self.v_des - self.v_min) * j_min / (j_max * j_min - j_max ** 2))
            t5 = - j_max / j_min * t4
            t3 = tf - self.t0 - t1 - t2 - t4 - t5  # debug here, 2022-10-22

            B_p = j_min * (1 / 6 * t1 ** 3 + 1 / 2 * t1 ** 2 * t2 + 1 / 2 * t1 * t2 ** 2) + j_max * (
                    1 / 6 * t2 ** 3) + 1 / 2 * self.a0 * (t1 + t2) ** 2 + self.v0 * (t1 + t2) + self.p0
            C_p = B_p + t3 * self.v_min
            E_p = j_max * (1 / 6 * t4 ** 3 + 1 / 2 * t4 ** 2 * t5 + 1 / 2 * t4 * t5 ** 2) + j_min * (
                    1 / 6 * t5 ** 3) + self.v_min * (t4 + t5) + C_p

            return t1, t2, t3, t4, t5, B_p, E_p

    def get_traj(self, tf=np.inf):
        t1, t2, t3, t4, t5, B_p, E_p = self.get_element(tf)
        C_p = B_p + t3 * self.v_min
        t123 = t1 + t2 + t3

        planned_a, planned_v, planned_p = [], [], []
        t = np.arange(1, (t1 + t2 + t3 + t4 + t5) / 0.1 + 1) / 10

        for t_ in t:
            if t_ < t1:
                a_ = j_min * t_ + self.a0
                v_ = 1 / 2 * j_min * t_ ** 2 + self.a0 * t_ + self.v0
                p_ = 1 / 6 * j_min * t_ ** 3 + 1 / 2 * self.a0 * t_ ** 2 + self.v0 * t_ + self.p0
            elif t_ < t1 + t2:
                a_ = j_max * (t_ - t1) + j_min * t1 + self.a0
                v_ = 1 / 2 * j_max * (t_ - t1) ** 2 + (
                        j_min * t1 + self.a0) * t_ - 1 / 2 * j_min * t1 ** 2 + self.v0
                p_ = 1 / 6 * j_max * (t_ - t1) ** 3 + 1 / 2 * (j_min * t1 + self.a0) * t_ ** 2 + (
                        - 1 / 2 * j_min * t1 ** 2 + self.v0) * t_ + (j_min * t1 ** 3) / 6 + self.p0
            elif t_ < t1 + t2 + t3:
                a_ = 0
                v_ = self.v_min
                p_ = B_p + (t_ - t1 - t2) * self.v_min
            elif t_ < t1 + t2 + t3 + t4:
                a_ = j_max * (t_ - t123)
                v_ = 1 / 2 * j_max * (t_ - t123) ** 2 + self.v_min
                p_ = 1 / 6 * j_max * (t_ - t123) ** 3 + self.v_min * t_ + C_p - t123 * self.v_min
            else:
                a_ = j_min * (t_ - t123 - t4) + j_max * t4
                v_ = 1 / 2 * j_min * (
                        t_ - t123 - t4) ** 2 + j_max * t4 * t_ - 1 / 2 * j_max * t4 ** 2 - j_max * t123 * t4 + self.v_min
                p_ = 1 / 6 * j_min * (t_ - t123 - t4) ** 3 + 1 / 2 * j_max * t4 * t_ ** 2 + (
                        self.v_min - 1 / 2 * j_max * t4 ** 2 - j_max * t123 * t4) * t_ + j_max * (
                             1 / 6 * t4 ** 3 + 1 / 2 * t4 ** 2 * t123 + 1 / 2 * t4 * t123 ** 2) + C_p - t123 * self.v_min

            planned_a.append(a_)
            planned_v.append(v_)
            planned_p.append(p_)

        t = t + self.t0
        return list(zip(t, planned_a, planned_v, planned_p))


class BBC:
    j_min = j_min  # m/s^3, the minimal jerk
    j_max = j_max  # m/s^3, the maximal jerk

    def __init__(self, t0=0.0, p0=-200.0, v0=Vehicle.v_R0, a0=0.0,
                 v_des=Vehicle.v_des['M1'], v_min=Vehicle.v_min['R0']):
        """ load the input variables """
        self.t0 = t0
        self.p0 = p0
        self.v0 = v0
        self.a0 = a0
        self.v_des = v_des
        self.v_min = v_min

        self.stage0_FLAG = self.exa_stage0_2steady()
        self.stage1_FLAG, self.stage1_t_one, self.stage1_pf_one = self.exa_stage1_2steady()

        self.stage2_max2min = Stage2_max2min(self)
        self.stage2_zero2min = Stage2_zero2min(self)
        self.stage3_max2zero2min = Stage3_max2zero2min(self)
        self.stage3_min2max2min = Stage3_min2max2min(self)
        self.stage5_min2max2zero2max2min = Stage5_min2max2zero2max2min(self)

    def exa_stage0_2steady(self):  # whether already in steady
        if abs(self.v0 - self.v_des) < 1e-3:
            if abs(self.a0) < 1e-1:
                stage0_FLAG = True
            else:
                print(f'Error at exa_stage0_2balance when (v0, a0): {self.v0, self.a0}')
                sys.exit()
        else:
            stage0_FLAG = False
        return stage0_FLAG

    def exa_stage1_2steady(self):  # a0 > 0, constant j_min (t_one)
        if self.stage0_FLAG:
            stage1_FLAG = False
            t_one, pf_one = None, None
        else:
            t_one = - self.a0 / self.j_min
            vf_one = 1 / 2 * self.j_min * t_one ** 2 + self.a0 * t_one + self.v0
            pf_one = 1 / 6 * self.j_min * t_one ** 3 + 1 / 2 * self.a0 * t_one ** 2 + self.v0 * t_one + self.p0
            stage1_FLAG = True if abs(vf_one - self.v_des) < 1e-2 and self.a0 > 0 else False
        return stage1_FLAG, t_one, pf_one


if __name__ == '__main__':
    """ 0. Demo: show BBC and trajectories """
    bbc = BBC()

    traj1, traj2, traj3 = None, None, None
    traj1 = bbc.stage2_max2min.get_traj()
    traj2 = bbc.stage3_max2zero2min.get_traj(pf=52)
    traj3 = bbc.stage3_min2max2min.get_traj(['v_min', 5])

    traj_dict = {'r': traj1, 'b': traj2, 'k': traj3}
    show_severalTrajs(traj_dict)

    """ 1. Debug """
    # # v_min = 13.970597031751657
    t0, p0, v0, a0 = 0, -150, 10, 0
    v_des, v_min = 20, 5
    bbc = BBC(t0=t0, p0=p0, v0=v0, a0=a0, v_des=v_des)
    # traj = bbc.stage3_min2max2min.get_traj(['v_min', v_min])
    traj1 = bbc.stage2_max2min.get_traj()
    # np.save('../paperFigs/dataAnalysis/rampStd.npy', traj1)
    show_severalTrajs({'b': traj1})

    plt.show()
