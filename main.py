import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import shgo
import math
import matplotlib.pyplot as plt

class FourOscillators:
    def __init__(self, paramE1, paramE2, paramE3, paramE4, paramE5, paramX1, paramX2, paramX3, paramX4, paramX5, paramEps):
        self.paramE1 = paramE1
        self.paramE2 = paramE2
        self.paramE3 = paramE3
        self.paramE4 = paramE4
        self.paramE5 = paramE5
        self.paramX1 = paramX1
        self.paramX2 = paramX2
        self.paramX3 = paramX3
        self.paramX4 = paramX4
        self.paramX5 = paramX5
        self.paramEps = paramEps

    def funcG2(self, phi):
        return self.paramE1 * np.cos(phi + self.paramX1) + self.paramE2 * np.cos(2 * phi + self.paramX2)

    def funcG3(self, phi):
        return self.paramE3 * np.cos(phi + self.paramX3)

    def funcG4(self, phi):
        return self.paramE4 * np.cos(phi + self.paramX4)

    def funcG5(self, phi):
        return self.paramE5 * np.cos(phi + self.paramX5)

    def sum1(self, psis, j):
        tmp = 0
        for k in range(3):
            tmp += math.trunc(self.funcG2(psis[k] - psis[j]) - self.funcG2(psis[k]))
        return tmp

    def sum2(self, psis, j):
        tmp = 0
        for k in range(3):
            for l in range(3):
                tmp += math.trunc(self.funcG3(psis[k] + psis[l] - 2 * psis[j]) - self.funcG3(psis[k] + psis[l]))
        return tmp

    def sum3(self, psis, j):
        tmp = 0
        for k in range(3):
            for l in range(3):
                tmp += math.trunc(self.funcG4(2 * psis[k] - psis[l] - psis[j]) - self.funcG4(2 * psis[k] - psis[l]))
        return tmp

    def sum4(self, psis, j):
        tmp = 0
        for k in range(3):
            for l in range(3):
                for m in range(3):
                    tmp += math.trunc(self.funcG5(psis[k] + psis[l] - psis[m] - psis[j]) - self.funcG5(psis[k] + psis[l] - psis[m]))
        return tmp

    def getSystem(self, psis):
        res = list(psis)
        for i in range(3):
            res[i] = self.paramEps / 4 * self.sum1(psis, i) + self.paramEps / 16 * self.sum2(psis, i) + \
                      self.paramEps / 16 * self.sum3(psis, i) + self.paramEps / 64 * self.sum4(psis, i)
        return res

    def funcF(self, psis):
        psis = list(psis)
        sys = self.getSystem(psis)
        return sys[1]**2 + sys[2]**2

system = FourOscillators(2.8, 2.6, 3.7, 4.3, 5.5, 6.1, 7.4, 8.9, 9.2, 10.3, 11.4)
res = shgo(system.funcF, [(0, 0), (0, 2 * math.pi), (0, 2 * math.pi)], n=100, iters=3)
system