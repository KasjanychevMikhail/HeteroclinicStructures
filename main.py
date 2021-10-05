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
            tmp += self.funcG2(psis[k] - psis[j]) - self.funcG2(psis[k])
        return tmp

    def sum2(self, psis, j):
        tmp = 0
        for k in range(4):
            for l in range(4):
                tmp += self.funcG3(psis[k] + psis[l] - 2 * psis[j]) - self.funcG3(psis[k] + psis[l])
        return tmp

    def sum3(self, psis, j):
        tmp = 0
        for k in range(4):
            for l in range(4):
                tmp += self.funcG4(2 * psis[k] - psis[l] - psis[j]) - self.funcG4(2 * psis[k] - psis[l])
        return tmp

    def sum4(self, psis, j):
        tmp = 0
        for k in range(4):
            for l in range(4):
                for m in range(4):
                    tmp += self.funcG5(psis[k] + psis[l] - psis[m] - psis[j]) - self.funcG5(psis[k] + psis[l] - psis[m])
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
        if psis[2] > psis[3]:
            return 1.0
        sys = self.getSystem(psis)
        Fx = sys[2] ** 2 + sys[3] ** 2
        if Fx < 10**(-10):
            return Fx
        else:
            return 1.0

    def searchABCD(self, M):
        psisx1 = [0., 0., M[0] + 0.001, M[1]]
        psisy1 = [0., 0., M[0], M[1] + 0.001]
        psisx2 = [0., 0., M[0] - 0.001, M[1]]
        psisy2 = [0., 0., M[0], M[1] - 0.001]
        sys1 = self.getSystem(psisx1)
        sys2 = self.getSystem(psisy1)
        sys3 = self.getSystem(psisx2)
        sys4 = self.getSystem(psisy2)

        a = (sys1[2] - sys3[2]) / (2 * 0.001)
        b = (sys2[2] - sys4[2]) / (2 * 0.001)
        c = (sys1[3] - sys3[3]) / (2 * 0.001)
        d = (sys2[3] - sys4[3]) / (2 * 0.001)

        return [a, b, c, d]

    def searchLambdas(self, a, b, c, d):
        D = (a + d)**2 / 4 - np.linalg.det([[a, b], [c, d]])
        if D < 0:
            return [-1000000, -1000000]
        lambda1 = (a + d) / 2 + math.sqrt(D)
        lambda2 = (a + d) / 2 - math.sqrt(D)
        if lambda1 * lambda2 >= 0:
            return [-1000000, -1000000]
        return [lambda1, lambda2]

    def searchGammas(self, lambdas, M):
        gammas = [0., 0.]
        psisx1 = [0., 0., M[0] + 0.001, M[1]]
        psisy1 = [0., 0., M[0], M[1] + 0.001]
        psisx2 = [0., 0., M[0] - 0.001, M[1]]
        psisy2 = [0., 0., M[0], M[1] - 0.001]
        sys1 = self.getSystem(psisx1)
        sys2 = self.getSystem(psisy1)
        sys3 = self.getSystem(psisx2)
        sys4 = self.getSystem(psisy2)

        dPx = (sys1[2] - sys3[2]) / (2 * 0.001)
        dPy = (sys2[2] - sys4[2]) / (2 * 0.001)
        dQx = (sys1[3] - sys3[3]) / (2 * 0.001)
        dQy = (sys2[3] - sys4[3]) / (2 * 0.001)

        for i in range(2):
            if abs(dPy) > 0.001:
                gammas[i] = (lambdas[i] - dPx) / dPy
            elif abs(lambdas[i] - dQy) > 0.001:
                gammas[i] = dQx / (lambdas[i] - dQy)
            else:
                gammas[i] = 10000

        return gammas

    def searchSepStart(self, gammas, M, d0):
        tg1 = math.tan(gammas[0])
        tg2 = math.tan(gammas[1])
        x = [0., 0., 0., 0.]
        y = [0., 0., 0., 0.]

        d = d0
        r1 = d0 + 1
        r2 = r1
        while r1 > d0 and r2 > d0 or d > 10**(-7):
            d = d / 2
            r1 = math.sqrt(d**2 + (d * tg1**2))
            r2 = math.sqrt(d**2 + (d * tg2**2))

        if tg1 > 9999:
            x[0] = M[0]
            x[2] = M[0]
            y[0] = M[1] + d
            y[2] = M[1] - d
        if tg2 > 9999:
            x[1] = M[0]
            x[3] = M[0]
            y[1] = M[1] + d
            y[3] = M[1] - d
        if 0 < tg1 < 9999:
            x[0] = M[0] + d
            x[2] = M[0] - d
        if tg1 < 0:
            x[0] = M[0] - d
            x[2] = M[0] + d
        if 0 < tg2 < 9999:
            x[1] = M[0] + d
            x[3] = M[0] - d
            y[0] = M[1] + abs(tg1)
            y[1] = M[1] + abs(tg2)
            y[2] = M[1] - abs(tg1)
            y[3] = M[1] - abs(tg2)
        if tg2 < 0:
            x[1] = M[0] - d
            x[3] = M[0] + d
            y[0] = M[1] + abs(tg1)
            y[1] = M[1] + abs(tg2)
            y[2] = M[1] - abs(tg1)
            y[3] = M[1] - abs(tg2)
        return [x, y]

    def checkLambdas(self, saddles):
        res = []
        for i in range(len(saddles)):
            M = [0., 0.]
            M[0] = saddles[i][2]
            M[1] = saddles[i][3]
            abcd = self.searchABCD(M)
            lambdas = self.searchLambdas(abcd[0], abcd[1], abcd[2], abcd[3])
            if lambdas[0] != -10000 and lambdas[1] != -10000:
                res.append(saddles[i])
        return res

    def searchCycles1(self, saddles, param):
        saddles = list(saddles)
        res = list(range(1))
        for i in range(len(saddles)):
            M = [0., 0.]
            M[0] = saddles[i][2]
            M[1] = saddles[i][3]
            abcd = self.searchABCD(M)
            lambdas = self.searchLambdas(abcd[0], abcd[1], abcd[2], abcd[3])
            gammas = self.searchGammas(lambdas, M)
            startXY = self.searchSepStart(gammas, M, 0.01)

            for s in range(4):
                tmp = solve_ivp(self.getSystem, (0, 1000), [startXY[0][s], startXY[1][s]])
                for j in range(len(saddles)):
                    for k in range(len(tmp)):
                        if abs(tmp[k][2] - saddles[j][2]) <= param:
                            if abs(tmp[k][3] - saddles[j][3]) <= param:
                                res.append([saddles[i], saddles[j]])
                                break
        return res

    def searchCycles2(self, cycles, param):
        cycles = list(cycles)
        res = list(range(0))
        for i in range(len(cycles)):
            M = [0., 0.]
            M[0] = cycles[i][1][2]
            M[1] = cycles[i][1][3]
            abcd = self.searchABCD(M)
            lambdas = self.searchLambdas(abcd[0], abcd[1], abcd[2], abcd[3])
            gammas = self.searchGammas(lambdas, M)
            startXY = self.searchSepStart(gammas, M, 0.01)
            flag = 0

            for s in range(4):
                tmp = solve_ivp(self.getSystem, (0, 1000), [startXY[0][s], startXY[1][s]])
                for j in range(len(cycles)):
                    for l in range(len(tmp)):
                        for k in range(4):
                            if abs(tmp[l][2] - cycles[j][1][3] ** k) <= param:
                                if abs(tmp[l][3] - (2 * math.pi) ** k) <= param:
                                    res.append([cycles[i][0], cycles[i][1], [cycles[j][1][3] ** k, (2 * math.pi) ** k]])
                                    flag = 1
                                    break
                        if flag == 1:
                            flag = 0
                            break
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
system = FourOscillators(1.8, 0.6, 1.7, 0.3, 1.5, 0.1, 1.4, 0.9, 1.2, 0.3, 2.4)
#res = shgo(system.funcF, [(0, 0), (0, 0), (0, 2 * math.pi), (0, 2 * math.pi)])
system.searchLambdas([1.2, 3.5])
system
