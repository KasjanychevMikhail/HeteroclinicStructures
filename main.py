import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import shgo
import math
import matplotlib.pyplot as plt


class FourOscillators:
    def __init__(self, paramE1, paramE2, paramE3, paramE4, paramE5, paramX1, paramX2, paramX3, paramX4, paramX5,
                 paramEps):
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
        for k in range(4):
            tmp += math.trunc(self.funcG2(psis[k] - psis[j]) - self.funcG2(psis[k]))
        return tmp

    def sum2(self, psis, j):
        tmp = 0
        for k in range(4):
            for l in range(4):
                tmp += math.trunc(self.funcG3(psis[k] + psis[l] - 2 * psis[j]) - self.funcG3(psis[k] + psis[l]))
        return tmp

    def sum3(self, psis, j):
        tmp = 0
        for k in range(4):
            for l in range(4):
                tmp += math.trunc(self.funcG4(2 * psis[k] - psis[l] - psis[j]) - self.funcG4(2 * psis[k] - psis[l]))
        return tmp

    def sum4(self, psis, j):
        tmp = 0
        for k in range(4):
            for l in range(4):
                for m in range(4):
                    tmp += math.trunc(
                        self.funcG5(psis[k] + psis[l] - psis[m] - psis[j]) - self.funcG5(psis[k] + psis[l] - psis[m]))
        return tmp

    def getSystem(self, psis):
        res = list(psis)
        for i in range(4):
            if i == 0:
                res[i] = 0
            else:
                res[i] = self.paramEps / 4 * self.sum1(psis, i) + self.paramEps / 16 * self.sum2(psis, i) + \
                         self.paramEps / 16 * self.sum3(psis, i) + self.paramEps / 64 * self.sum4(psis, i)
        return res

    def funcF(self, psis):
        psis = list(psis)
        sys = self.getSystem(psis)
        return sys[2] ** 2 + sys[3] ** 2

    def searchCycles1(self, saddles, param):
        saddles = list(saddles)
        res = list(range(1))
        for i in range(len(saddles)):
            tmp = solve_ivp(self.getSystem, (0, 1000), [saddles[i]])
            for j in range(len(tmp)):
                for k in range(len(saddles)):
                    if abs(tmp[j][2] - saddles[k][2]) <= param:
                        if abs(tmp[j][3] - saddles[k][3]) <= param:
                            res.append((saddles[i], saddles[k]))
        return res

    def searchCycles2(self, cycles, param):
        cycles = list(cycles)
        res = list(range(0))
        for i in range(len(cycles)):
            tmp = solve_ivp(self.getSystem, (0, 1000), [cycles[i][1]])
            for j in range(len(tmp)):
                for l in range(len(cycles)):
                    for k in range(4):
                        if abs(tmp[j][2] - cycles[l][1][3] ** k) <= param:
                            if abs(tmp[j][3] - (2 * math.pi) ** k) <= param:
                                res.append((cycles[i][0], cycles[i][1], [cycles[l][1][3] ** k, (2 * math.pi) ** k]))
        return res


def checkSystemParam(paramE, paramX):
    res = False
    system = FourOscillators(paramE, paramE, paramE, paramE, paramE, paramX, paramX, paramX, paramX, paramX, 0)
    min = shgo(system.funcF, [(0, 0), (0, 0), (0, 2 * math.pi), (0, 2 * math.pi)], iters=3)
    cyc1 = system.searchCycles1(min, 0.5)
    cyc2 = system.searchCycles2(cyc1, 0.5)
    if len(cyc2) >= 1:
        res = True
    return res


#def makePictureParam(startX, stopX):
#    x = []
#    y = []
#
#    for i in list(range(startX, stopX + 1, (stopX - startX) * 10)):
#        for j in list(range(startX, stopX + 1, (stopX - startX) * 10)):
#            if checkSystemParam(i, j):
#                x.append(i)
#                y.append(j)

#    axs.scatter(x, y)

#    axs.set_xlim((startX, stopX))
#    axs.set_ylim((startX, stopX))
#    plt.show()


fig, axs = plt.subplots()
system = FourOscillators(0.8, 0.6, 0.7, 0.3, 0.5, 0.1, 0.4, 0.9, 0.2, 0.3, 0.4)
res = shgo(system.funcF, [(0, 0), (0, 0), (0, 2 * math.pi), (0, 2 * math.pi)], iters=3)
system
