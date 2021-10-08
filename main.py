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

    def getSystemT(self, t, psis):
        res = list(psis)
        for i in range(4):
            if i == 0:
                res[i] = 0
            else:
                res[i] = self.paramEps / 4 * self.sum1(psis, i) + self.paramEps / 16 * self.sum2(psis, i) + \
                         self.paramEps / 16 * self.sum3(psis, i) + self.paramEps / 64 * self.sum4(psis, i)
        return res

    def funcF(self, psis):
        psisx = [0., 0., psis[0], psis[1]]
        sys = self.getSystem(psisx)
        Fx = sys[2] ** 2 + sys[3] ** 2
        return Fx

    def funcT(self, psis):
        res = psis
        res[0] = psis[1]
        res[1] = 2 * math.pi
        return res

    def ineqConstrF(self, psis):
        psisx = [0., 0., psis[0], psis[1]]
        sys = self.getSystem(psisx)
        Fx = sys[2] ** 2 + sys[3] ** 2
        return -Fx + 10**(-10)

    def ineqConstrF2(self, psis):
        psisx = [0., 0., psis[0], psis[1]]
        sys = self.getSystem(psisx)
        Fx = sys[2] ** 2 + sys[3] ** 2
        return Fx

    def searchABCD(self, M):
        psisx1 = [0., 0., M[0], M[1] + 0.001]
        psisy1 = [0., 0., M[0] + 0.001, M[1]]
        psisx2 = [0., 0., M[0], M[1] - 0.001]
        psisy2 = [0., 0., M[0] - 0.001, M[1]]
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
        delta = np.linalg.det([[a, b], [c, d]])
        sigma = -(a + d)
        D = sigma**2 / 4 - delta
        if D < 0:
            return [-1000000, -1000000]
        lambda1 = -sigma / 2 + math.sqrt(D)
        lambda2 = -sigma / 2 - math.sqrt(D)
        if lambda1 * lambda2 >= 0:
            return [-1000000, -1000000]
        return [lambda1, lambda2]

    def searchGammas(self, lambdas, M):
        gammas = [0., 0.]
        psisx1 = [0., 0., M[0], M[1] + 0.001]
        psisy1 = [0., 0., M[0] + 0.001, M[1]]
        psisx2 = [0., 0., M[0], M[1] - 0.001]
        psisy2 = [0., 0., M[0] - 0.001, M[1]]
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
        res = list(range(0))
        for i in range(len(saddles)):
            M = [0., 0.]
            #saddles[i] = [0., 0., saddles[i][0], saddles[i][1]]
            M[0] = saddles[i][0]
            M[1] = saddles[i][1]
            abcd = self.searchABCD(M)
            lambdas = self.searchLambdas(abcd[0], abcd[1], abcd[2], abcd[3])
            gammas = self.searchGammas(lambdas, M)
            startXY = self.searchSepStart(gammas, M, 0.0001)

            for s in range(4):
                #h0 = 0.01
                #lambdasi = 0.
                #if s == 0 or s == 2:
                #    lambdasi = lambdas[0]
                #else:
                #    lambdasi = lambdas[1]
                #h = h0 * np.sign(lambdasi)
                tmp = solve_ivp(self.getSystemT, (0, 50), [0, 0, startXY[0][s], startXY[1][s]], method='RK45')
                                #rtol=1e-8, atol=1e-8)
                for j in range(len(saddles)):
                    if i == j:
                        continue
                    for k in range(len(tmp.y[2])):
                        if math.sqrt((tmp.y[2][k] - saddles[j][0])**2 + (tmp.y[3][k] - saddles[j][1])**2) <= param:
                            res.append([saddles[i], saddles[j]])
                            break
        return res

    def searchCycles2(self, cycles, param):
        cycles = list(cycles)
        res = list(range(0))
        for i in range(len(cycles)):
            M = [0., 0.]

            M[0] = cycles[i][1][0]
            M[1] = cycles[i][1][1]
            abcd = self.searchABCD(M)
            lambdas = self.searchLambdas(abcd[0], abcd[1], abcd[2], abcd[3])
            gammas = self.searchGammas(lambdas, M)
            startXY = self.searchSepStart(gammas, M, 0.01)
            flag = 0
            T = self.funcT([cycles[i][0][0], cycles[i][0][1]])

            for s in range(4):
                tmp = solve_ivp(self.getSystemT, (0, 50), [0., 0., startXY[0][s], startXY[1][s]], method='RK45')
                                #rtol=1e-10, atol=1e-10)
                for j in range(len(cycles)):
                    if i == j:
                        continue
                    for l in range(len(tmp.y[2])):
                        for k in range(4):
                            if math.sqrt(
                                    (tmp.y[2][l] - T[0]**k) ** 2 + (tmp.y[3][l] - T[1]**k) ** 2) <= param:
                                res.append([cycles[i][0], cycles[i][1], [cycles[j][0][1] ** k, (2 * math.pi) ** k]])
                                flag = 1
                                break
                        if flag == 1:
                            flag = 0
                            break
        return res


def checkSystemParam(paramE, paramX):
    res = False
    system = FourOscillators(-0.3, 0.3, 0.02, 0.8, 0.02, paramE, paramX, 0., 1.73, 0., 1.)
    cons = ({'type': 'ineq', 'fun': ineqConstr},
            {'type': 'ineq', 'fun': system.ineqConstrF},
            {'type': 'ineq', 'fun': system.ineqConstrF2})
    pi2 = 2 * math.pi
    bounds = ((0, pi2), (0, pi2))
    minS = shgo(system.funcF, bounds=bounds, n=64, iters=3, constraints=cons, options={'minim_every_item': 'True'})
    if minS.success:
        min = minS.xl
        cyc1 = system.searchCycles1(min, 1.)
        if len(cyc1) >= 1:
            cyc2 = system.searchCycles2(cyc1, 1.)
            if len(cyc2) >= 1:
                res = True
    return res

def ineqConstr(psis):
    psis = list(psis)
    return psis[1] - psis[0]

def funcF(psis, *args):
    psisx = [0., 0., psis[0], psis[1]]
    system = FourOscillators(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10])
    sys = system.getSystem(psisx)
    Fx = sys[2] ** 2 + sys[3] ** 2
    return Fx

def makePictureParam(startX, stopX):
    x = []
    y = []

    for i in range(0, 200 + 1, 10):
        i = i / 100
        for j in range(300, 510 + 1, 10):
            j = j / 100
            if checkSystemParam(i, j):
                x.append(i)
                y.append(j)

    axs.scatter(x, y, s=1, c='black')

    axs.set_xlim((0., 2.))
    axs.set_ylim((3., 5.1))
    axs.set(xlabel='X1')
    axs.set(ylabel='X2')
    plt.show()


fig = plt.figure(figsize=(4, 4))
axs = fig.add_subplot()
args = (1.8, 0.6, 1.7, 0.3, 1.5, 0.1, 1.4, 0.9, 1.2, 0.3, 2.4)
system = FourOscillators(0., -4.6, -4.6, -4.6, -4.6, -4.6, 3.4, 3.4, 3.4, 3.4, 3.4)
constraint1 = {'type': 'ineq', 'fun': ineqConstr}
constraint2 = {'type': 'ineq', 'fun': system.ineqConstrF}
cons = ({'type': 'ineq', 'fun': ineqConstr},
        {'type': 'ineq', 'fun': system.ineqConstrF},
        {'type': 'ineq', 'fun': system.ineqConstrF2})
pi2 = 2 * math.pi
bounds = ((0, pi2), (0, pi2))
#res = shgo(system.funcF, bounds=bounds, n=64, iters=3, constraints=cons, options={'minim_every_item': 'True'})
#ABCD = system.searchABCD(res.xl[3])
#lambdas = system.searchLambdas(ABCD[0], ABCD[1], ABCD[2], ABCD[3])
#gammas = system.searchGammas(lambdas, res.xl[1])
#startXY = system.searchSepStart(gammas, res.xl[1], 0.5)
#cycles1 = system.searchCycles1(res.xl, 0.8)
#cycles2 = system.searchCycles2(cycles1, )

makePictureParam(100, 350)
