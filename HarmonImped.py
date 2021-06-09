# 谐波阻抗估计
from scipy.io import loadmat
import numpy as np


def harmon_imped(hr, a=1000, b=9000, delta_n=10):
    LL = b - a + 1 - delta_n
    # HA = np.zeros(3, 99)
    HI = np.zeros((3, 99))
    for j in range(3):
        kkk = 0

        for Num in range(1, 100):
            hh = Num
            Uht2 = hr[:, j, hh]
            Iht2 = hr[:, j + 7, hh]
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
            HI[j, kkk] = A[0]
            kkk += 1

    return HI


data = loadmat('data/HR1.mat')
HR1 = data['HR1']
HI = harmon_imped(HR1)
HI


