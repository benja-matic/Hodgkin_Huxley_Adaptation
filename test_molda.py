import numpy as np
import matplotlib.pyplot as plt

N = 500
IFRAC = 2
k = 50

Aee = .7#float(sys.argv[5])
Aei = 1.5#float(sys.argv[6])
Aie = 2.0#float(sys.argv[7])
Aie_NL = 2.5#float(sys.argv[8])
Aii = 2.5#float(sys.argv[9])


def molda_weights(N, IFRAC, k, Aee, Aei, Aie, Aie_NL, Aii):
    N_local = N/2
    Ni_local = N_local/IFRAC
    Ne_local = N_local - Ni_local
    Ne2 = Ne_local*2
    ks = np.sqrt(k)

    Jee = Aee/ks
    Jei = Aei/ks
    Jie = Aie/ks
    Jie_NL = Aie_NL/ks
    Jii = Aii/ks

    W = np.zeros((N,N))

    for i in range(Ne_local):
        ee1_inds = np.random.randint(0, Ne_local, k)
        ei1_inds = np.random.randint(Ne2, Ne2 + Ni_local, k)
        ee2_inds = np.random.randint(Ne_local, Ne2, k) #try ee2 = ee1 + Ne_local
        ei2_inds = np.random.randint(N_local + Ne_local, N, k)
        for j in range(k):
            W[i, ee1_inds[j]] += Jee
            W[i + Ne_local, ee2_inds[j]] += Jee
            W[i, ei1_inds[j]] += Jei
            W[i+Ne_local, ei2_inds[j]] += Jei


    for i in range(Ne2, Ne2 + Ni_local):
        ie1_inds = np.random.randint(0, Ne_local, k)
        ii1_inds = np.random.randint(Ne2, Ne2+Ni_local, k)
        ie2_inds = np.random.randint(Ne_local, Ne2, k)
        ii2_inds = np.random.randint(Ne2+Ni_local, N, k)
        ieNL1_inds = np.random.randint(Ne_local, Ne2, k)
        ieNL2_inds = np.random.randint(0, Ne_local, k)
        for j in range(k):
            W[i, ie1_inds[j]] += Jie
            W[i, ii1_inds[j]] += Jii
            W[i+Ni_local, ie2_inds[j]] += Jie
            W[i+Ni_local, ii2_inds[j]] += Jii
            W[i, ieNL1_inds[j]] += Jie_NL
            W[i+Ni_local, ieNL2_inds[j]] += Jie_NL

    return W



def molda_weights2(N, IFRAC, k, Aee, Aei, Aie, Aie_NL, Aii):
    N_local = N/2
    Ni_local = N_local/IFRAC
    Ne_local = N_local - Ni_local
    Ne2 = Ne_local*2
    ks = np.sqrt(k)

    Jee = Aee/ks
    Jei = Aei/ks
    Jie = Aie/ks
    Jie_NL = Aie_NL/ks
    Jii = Aii/ks

    W = np.zeros((N,N))

    for i in range(Ne_local):
        ee1_inds = np.random.randint(0, Ne_local, k)
        ei1_inds = np.random.randint(Ne2, Ne2 + Ni_local, k)
        for j in range(k):
            W[i, ee1_inds[j]] += Jee
            W[i + Ne_local, ee1_inds[j] + Ne_local] += Jee
            W[i, ei1_inds[j]] += Jei
            W[i+Ne_local, ei1_inds[j] + Ni_local] += Jei


    for i in range(Ne2, Ne2 + Ni_local):
        ie1_inds = np.random.randint(0, Ne_local, k)
        ii1_inds = np.random.randint(Ne2, Ne2+Ni_local, k)
        ieNL1_inds = np.random.randint(Ne_local, Ne2, k)
        for j in range(k):
            W[i, ie1_inds[j]] += Jie
            W[i, ii1_inds[j]] += Jii
            W[i+Ni_local, ie1_inds[j] + Ne_local] += Jie
            W[i+Ni_local, ii1_inds[j] + Ni_local] += Jii
            W[i, ieNL1_inds[j]] += Jie_NL
            W[i+Ni_local, ieNL1_inds[j] - Ne_local] += Jie_NL

    return W

# W = molda_weights(N, IFRAC, k, Aee, Aei, Aie, Aie_NL, Aii)
W = molda_weights2(N, IFRAC, k, Aee, Aei, Aie, Aie_NL, Aii)


N_local = N/2
Ni_local = N_local/IFRAC
Ne_local = N_local - Ni_local
Ne2 = Ne_local*2
ks = np.sqrt(k)

Aeex = Aee * ks
Aeix = Aei * ks
Aiex = Aie * ks
AieNLx = Aie_NL * ks
Aiix = Aii * ks


c = 0

print(sum(W[0, 0:Ne_local]) - Aeex)
print(sum(W[0, Ne_local:Ne2]))
print(sum(W[0, Ne2:Ne2 + Ni_local]) - Aeix)
print(sum(W[0, Ne2+Ni_local:N]))

print(sum(W[Ne_local, 0:Ne_local]))
print(sum(W[Ne_local, Ne_local:Ne2]) - Aeex)
print(sum(W[Ne_local, Ne2:Ne2 + Ni_local]))
print(sum(W[Ne_local, Ne2+Ni_local:N]) - Aeix)

print(sum(W[Ne2, 0:Ne_local]) - Aiex)
print(sum(W[Ne2, Ne_local:Ne2]) - AieNLx)
print(sum(W[Ne2, Ne2:Ne2 + Ni_local]) - Aiix)
print(sum(W[Ne2, Ne2+Ni_local:N]))

print(sum(W[Ne2+Ni_local, 0:Ne_local]) - AieNLx)
print(sum(W[Ne2+Ni_local, Ne_local:Ne2]) - Aiex)
print(sum(W[Ne2+Ni_local, Ne2:Ne2 + Ni_local]))
print(sum(W[Ne2+Ni_local, Ne2+Ni_local:N]) - Aiix)

#weights worked to machine precision
