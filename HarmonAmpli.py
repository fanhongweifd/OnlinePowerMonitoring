# 谐波放大系数估计
from scipy.io import loadmat
import numpy as np


def harmon_ampli(hr, a=1000, b=7000, delta_n=50):
    LL = b - a + 1 - delta_n
    # HA = np.zeros(3, 99)
    HI = np.zeros(99)

    kkk = 0
    for Num in range(1, 100):
        hh = Num
        Uht2 = hr[:, 0, hh]
        Iht2 = hr[:, 3, hh]
        DeltaUht = np.zeros(LL)
        DeltaIht = np.zeros(LL)
        SumDUh = np.zeros(LL)
        SumDIh = np.zeros(LL)

        for i in range(LL):
            DeltaUht[i] = Uht2[i + delta_n] - Uht2[i]
            DeltaIht[i] = Iht2[i + delta_n] - Iht2[i]
            # SumDUh[i] = sum(DeltaUht[:i])
            # SumDIh[i] = sum(DeltaIht[:i])
        SumDUh = np.cumsum(DeltaUht)
        SumDIh = np.cumsum(DeltaIht)
        A = np.polyfit(SumDIh, SumDUh, 1)
        # z = np.polyval(A, SumDIh)
        HI[kkk] = A[0]
        kkk += 1

    return HI


data = loadmat('data/H1.mat')
HR1 = data['H1']
HI = harmon_ampli(HR1)
HI


