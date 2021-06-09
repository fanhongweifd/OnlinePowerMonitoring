import numpy as np
import pandas as pd

def ismember(B, A):
    B = np.array(B)
    A = np.array(A)
    B_unique_sorted, B_idx = np.unique(B, return_index=True)
    B_in_A_bool = np.in1d(B_unique_sorted, A, assume_unique=True)

    B_in_A_index = np.zeros(B_in_A_bool.shape)
    for i, value in enumerate(B_in_A_bool):
        if value:
            B_in_A_index[i] = np.where(A == B[i])[0]
        else:
            B_in_A_index[i] = -1
    # B_unique_sorted[B_in_A_bool], B_idx[B_in_A_bool],
    return B_in_A_bool, B_in_A_index

# A = [1,2, 3, 4, 5]
# B =[2, 4, 6, 8, 10]
# ismember(A,B)


def correlation(U_rms, I_rmsT):
    U_rms = (U_rms * 1.732051 - 2.2e5) / 2.2e5

    # 计算电压U_rms提取特征序列点U_cp
    m1 = 0
    Num_cp = np.zeros(len(U_rms))
    U_cp = np.zeros(len(U_rms))
    for i in range(1, len(U_rms) - 1):
        if ((U_rms[i] >= U_rms[i - 1] and U_rms[i] > U_rms[i + 1])\
            or (U_rms[i] > U_rms[i - 1] and U_rms[i] >= U_rms[i + 1]))\
                or ((U_rms[i] <= U_rms[i - 1] and U_rms[i] < U_rms[i + 1])\
                    or (U_rms[i] < U_rms[i - 1] and U_rms[i] <= U_rms[i + 1])):
            U_cp[m1] = U_rms[i]
            Num_cp[m1] = i
            m1 += 1
    U_cp = U_cp[:m1]
    Num_cp = Num_cp[:m1]


    # 计算电流I_rmsT提取特征序列点I_cpT
    m4 = 0
    Num_cpT = np.zeros(len(I_rmsT))
    I_cpT = np.zeros(len(I_rmsT))
    for i in range(1, len(I_rmsT) - 1):
        if ((I_rmsT[i] >= I_rmsT[i - 1] and I_rmsT[i] > I_rmsT[i + 1])\
            or (I_rmsT[i] > I_rmsT[i - 1] and I_rmsT[i] >= I_rmsT[i + 1]))\
                or ((I_rmsT[i] <= I_rmsT[i - 1] and I_rmsT[i] < I_rmsT[i + 1])\
                    or (I_rmsT[i] < I_rmsT[i - 1] and I_rmsT[i] <= I_rmsT[i + 1])):
            I_cpT[m4] = I_rmsT[i]
            Num_cpT[m4] = i
            m4 += 1
    I_cpT = I_cpT[:m4]
    Num_cpT = Num_cpT[:m4]


    # 提取U_cp对应的关键特征序列点U_icp
    U_icp = np.zeros(len(U_cp))
    Num_icp = np.zeros(len(U_cp))
    d1 = np.zeros(len(U_cp))
    t1 = np.zeros(len(U_cp))
    k1 = 0
    for i in range(1, len(U_cp) - 1):
        d1[i] = abs(U_cp[i-1] - U_cp[i] + (U_cp[i+1] - U_cp[i-1]) * (Num_cp[i] - Num_cp[i-1])/
                    (Num_cp[i+1] - Num_cp[i-1]))
        t1[i] = min((Num_cp[i] - Num_cp[i-1]), (Num_cp[i+1] - Num_cp[i]))
        if d1[i] >= 25 or t1[i] >= 3:
            U_icp[k1] = U_cp[i]
            Num_icp[k1] = Num_cp[i]
            k1 += 1
    U_icp = U_icp[:k1]
    Num_icp = Num_icp[:k1]
    U_icp = np.concatenate(([U_rms[0]], U_icp, [U_rms[-1]]))
    Num_icp = np.concatenate(([0], Num_icp, [len(U_rms)-1]))


    # 提取I_cpT对应的关键特征序列点
    I_icpT = np.zeros(I_cpT.shape[0])
    Num_icpT = np.zeros(I_cpT.shape[0])
    d4 = np.zeros(I_cpT.shape[0])
    t4 = np.zeros(I_cpT.shape[0])
    k4 = 0
    for i in range(1, I_cpT.shape[0] - 1):
        d4[i] = abs(I_cpT[i-1] - I_cpT[i] + (I_cpT[i+1] - I_cpT[i-1]) * (Num_cpT[i] - Num_cpT[i-1])/
                    (Num_cpT[i+1] - Num_cpT[i-1]))
        t4[i] = min((Num_cpT[i] - Num_cpT[i-1]), (Num_cpT[i+1] - Num_cpT[i]))
        if d4[i] >= 3 or t4[i] >= 3:
            I_icpT[k4] = I_cpT[i]
            Num_icpT[k4] = Num_cpT[i]
            k4 += 1
    I_icpT = I_icpT[:k4]
    Num_icpT = Num_icpT[:k4]
    I_icpT = np.concatenate(([I_rmsT[0]], I_icpT, [I_rmsT[-1]]))
    Num_icpT = np.concatenate(([0], Num_icpT, [I_rmsT.shape[0]-1]))


    # U_rms 0插进去，方便后续操作
    count = 0
    U_icpfull = np.zeros(len(U_rms))
    for i in range(len(U_rms)):
        if i in Num_icp:
            U_icpfull[i] = U_icp[count]
            count += 1
        else:
            U_icpfull[i] = 0

    # I_rmsT 0插进去，方便后续操作
    count = 0
    I_icpfullT = np.zeros(len(I_rmsT))
    for i in range(len(I_rmsT)):
        if i in Num_icpT:
            I_icpfullT[i] = I_icpT[count]
            count += 1
        else:
            I_icpfullT[i] = 0

    Num_same = np.intersect1d(Num_icp, Num_icpT)
    Num_all = np.union1d(Num_icp, Num_icpT)
    index = False * np.zeros(Num_all.shape)
    for i in range(len(Num_same)):
        res = (Num_all == Num_same[i])
        index = (index + res)
    index = index > 0
    Num_cz = Num_all[~index]

    tf, kk = ismember(Num_same, Num_all)
    a = []
    b = []
    n = []
    t = 0
    # for i in range(1, len(Num_same)):
    #     if kk[i] - kk[i - 1] == 2:
    #         tf, _ = ismember(Num_same, Num_icpT)
    #         if tf:
    #             a[t] = (U_icpfull[Num_all[kk[i]]] - U_icpfull[Num_all[kk[i - 1]]])/\
    #                    (Num_all[kk[i]] - Num_all[kk[i - 1]]) * (
    #                     Num_all[kk[i - 1] + 1] - Num_all[kk[i - 1]]) + U_icpfull[Num_all[kk[i - 1]]];
    #             b[t] = U_icpfull[Num_all[kk[i - 1] + 1]]
    #             n[t] = Num_all[kk[i]] - Num_all[kk[i - 1]]
    #             t = t + 1



    I_rmsT
    Num_cpT
    I_cpT
    Num_icpT
    I_icpT

    rho = 0.5
    delta = np.zeros(len(U_rms))
    for i in range(len(U_rms)):
        delta[i] = U_icpfull[i] - I_icpfullT[i]

    mmax = max(abs(delta))
    mmin = min(abs(delta))
    ksi = (mmin + rho * mmax)/ (abs(delta) + rho * mmax)
    return ksi

    # rho = 0.5
    # delta = np.zeros(len(U_icpfull))

# data = csv.reader('data.csv')
# f = open('data.csv', 'r')
df_csv = pd.read_csv('data/data.csv')
data = df_csv.iloc[:, -7:]
data = data.dropna(axis=0)
U_rmsA=data.iloc[:, 4].values
U_rmsB=data.iloc[:, 5].values
U_rmsC=data.iloc[:, 6].values
I_rmsT1=data.iloc[:, 0].values
I_rmsF1=data.iloc[:, 1].values
I_rmsT2=data.iloc[:, 2].values
I_rmsF2=data.iloc[:, 3].values


ksi = correlation(U_rmsB, I_rmsT1)

data






