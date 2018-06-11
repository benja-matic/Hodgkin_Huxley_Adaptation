import numpy as np
import scipy as sp
import scipy.stats as ss
from scipy.integrate import ode
import time as TIME
import math
import time as TIME
import sys
'''
Simulate network of Buzsaki-Wang neurons using scipy implementation of VODE

WEIGHTS MATRICES NOW SET TO BE POSITIVE
SIGN OF SYNAPSE IS DECIDED BY REVERSAL POTENTIAL E_syn
CdV/dt = I_app - (sum of currents)

Neurons 0:Ne are excitatory, Ne:N are inhibitory
VODE requires a flat array, so initial values are 4*N long
RHS reshapes this vector, computes derivatives, reshapes it again, and returns it to VODE
'''

def basic_spike_detector(V):
    spikes = []
    for i in range(len(V)):
        if V[i] > -5:
            if V[i-1] <= -5:
                spikes.append(i)
    return np.array(spikes)

def schmitt_trigger_2T_spikes(V, t1, t2):
    Flag = False
    Flag_Storage = []
    Time_Storage = []
    Thresholds = [t1, t2]
    Threshold = Thresholds[0]
    for i in range(len(V)):
        Test = (V[i] > Threshold)
        if Test != Flag:
            Flag_Storage.append(Flag)
            Time_Storage.append(i)
            Flag = not Flag
            Thresholds = np.roll(Thresholds, 1)
            Threshold = Thresholds[0]
    if len(Time_Storage) %2 != 0:
        Flag_Storage.pop()
        Time_Storage.pop()
    spikes = np.zeros(len(Time_Storage)/2)
    for i in range(0, len(Time_Storage), 2):
        m = np.mean(Time_Storage[i:i+2])
        spikes[i/2] = m
    return spikes

#slice up stvars history so it only includes voltage in advance
def build_raster(V):
    N = np.shape(V)[0]
    te = []
    re = []
    for i in range(N):
        x = basic_spike_detector(v[i,:])
        for j in range(len(x)):
            te.append(x[j])
            re.append(i)
    m = np.argsort(te)
    ten, ren = np.array(te), np.array(re)
    return ten[m], ren[m]

def CV(d):
    m = np.mean(d)
    s = np.std(d)
    return s/m

def FANO(A):
    return np.var(A)/np.mean(A)

def kill_transient(t, r, trans):
    m = np.where(t > trans)[0]
    return t[m], r[m]

def neuron_finder(r, min_spikes, min_neurons):
    Neurons, Counts = np.unique(r, return_counts = True)
    N = []
    for i in range(len(Neurons)):
        if Counts[i] > min_spikes:
            N.append(Neurons[i])
    if len(N) > min_neurons:
        return N
    else:
        return -5.

def nt_diff(t, r, ntotal, half, netd_binsize):
    netd_bins = np.arange(0, ntotal, netd_binsize)
    ntd = np.zeros(len(netd_bins)-1)
    nts = np.empty_like(ntd)
    for j in range(len(ntd)):
        m = np.where((netd_bins[j] < t) & (t <= netd_bins[j+1]))[0]
        T = sum(r[m] > half)
        ntd[j] = T
        nts[j] = len(m)
    return ntd/nts

def comp(x):
    if x > 0.7:
        return "win"
    elif 0.3 <= x <= 0.7:
        return "draw"
    elif x < 0.3:
        return "lose"
    else:
        return "weird"

def WLD(s):
    if np.max(s) <= 0.3:
        return ["lose", "end"], [0, len(s)]
    elif np.min(s) >= 0.7:
        return ["win", "end"], [0, len(s)]
    elif (0.3 <= np.max(s) <= 0.7) & (0.3 <= np.min(s) <= 0.7):
        return ["draw", "end"], [0, len(s)]
    else:
        times = []
        flags = []
        flag = comp(s[0])
        times.append(0)
        flags.append(flag)
        s2 = s[1:]
        for i in range(len(s2)):
            f1 = comp(s2[i])
            if f1 != flag:
                flag = f1
                times.append(i+1)
                flags.append(flag)
        times.append(len(s))
        flags.append("end")
        return flags, times

def splice_reversions(flags, times):
    w0 = flags.index("win")
    l0 = flags.index("lose")
    m = [w0, l0]
    empezar = min(m)
    nf = [flags[empezar]]
    a = ["win", "lose"]
    t = [empezar]
    if empezar == w0:
        f = 1
    else:
        f = 0
    flag = a[f]
    for i in range(empezar, len(flags)):
        if flags[i] == a[f]:
            t.append(times[i])
            nf.append(flags[i])
            a = np.roll(a, 1)
    t.append(times[-1])
    return np.array(t), np.array(nf)

def ligase(exons, t, r, Neurons):
    n1 = min(Neurons)
    n2 = max(Neurons)
    c = 0
    tf, rf = np.zeros(1), np.zeros(1)
    for i in range(len(exons)):
        if i > 0:
            space = (exons[i][0] - exons[i-1][1]) - 1 #why -1?
            c+=space
        m = np.where((exons[i][0] <= t) & (t <= exons[i][1]))[0]
        tm = t[m] - c
        rm = r[m]
        m2 = np.where((n1 <= rm) & (rm <= n2))[0]
        rmm = rm[m2]
        tmm = tm[m2]
        tf = np.concatenate([tf, tmm])
        rf = np.concatenate([rf, rmm])
    return tf[1:], rf[1:]

def emptiness(x, funk, error_code):
    if len(x) > 0:
        return funk(x)
    else:
        return error_code

def splice_flags(flags, times, netd_binsize):
    l = [i for i in range(len(flags)) if flags[i] == "lose"]
    w = [i for i in range(len(flags)) if flags[i] == "win"]
    d = [i for i in range(len(flags)) if flags[i] == "draw"]
    bot = [[times[i]*netd_binsize, times[i+1]*netd_binsize] for i in l]
    top = [[times[i]*netd_binsize, times[i+1]*netd_binsize] for i in w]
    nmz = [[times[i]*netd_binsize, times[i+1]*netd_binsize] for i in d]
    bdom = emptiness([(i[1] - i[0]) for i in bot], sum, 0)
    tdom = emptiness([(i[1] - i[0]) for i in top], sum, 0)
    tnmz = emptiness([(i[1] - i[0]) for i in nmz], sum, 0)
    return top, tdom, bot, bdom, nmz, tnmz

def gather_CV(t, r, Neurons):
    CVS = np.zeros(len(Neurons))
    for i in range(len(Neurons)):
        m = np.where(r == Neurons[i])[0]
        T = t[m]
        D = np.diff(T)
        cv = CV(D)
        CVS[i] = cv
    CVS = [i for i in CVS if math.isnan(i) == False]
    return CVS

def build_CV(t, r, Neurons, Exons):
    CVS = []
    for n in range(len(Neurons)):
        sm = np.where(r == Neurons[n])[0]
        spikes = t[sm]
        ISI = []
        for i in range(len(Exons)):
            m = np.where((Exons[i][0] <= spikes) & (spikes <= Exons[i][1]))
            s = spikes[m]
            d = np.diff(s)
            for j in range(len(d)):
                ISI.append(d[j])
        cv = CV(ISI)
        if math.isnan(cv) == False:
            CVS.append(cv)
    return CVS

def grid_FF(spikes, time):
    counts = np.zeros(len(time)-1)
    for i in range(len(counts)):
        x = len(np.where((spikes > time[i]) & (spikes < time[i+1]))[0])
        counts[i] = x
    return np.var(counts)/np.mean(counts)

def pool_FF(Neurons, t, r, fbinsize):
    FFS = []
    for i in range(len(Neurons)):
        time = np.arange(0, t[-1], fbinsize) #chunk time
        bins = np.zeros(len(time)-1) #count bins
        m = np.where(r == Neurons[i]) #where this neuron fired
        spikes = t[m] #spike times of neuron i
        for j in range(len(bins)):
            x = len(np.where((spikes > time[j]) & (spikes < time[j+1]))[0]) #spike counts of neuron i in bin j
            bins[j] = x #update bins
        f = FANO(bins)
        if math.isnan(f) == False:
            FFS.append(f)
    return FFS

def rand_pair_cor(bin_size, t, r, Neurons, n):
    t0 = np.min(t)
    t1 = np.max(t)
    bins = np.arange(t0, t1, bin_size)
    count1 = np.zeros((n, len(bins)-1))
    ya = []
    lank = len(Neurons)
    shank = np.zeros((lank, len(bins) - 1))
    cor_store = []
    pairs = []
    for i in range(n):
        #draw two random neurons
        x1 = np.random.randint(0, lank)
        x2 = np.random.randint(0, lank)
        #if both have been checked already
        if ((x1 in ya) & (x2 in ya)):
            #if you've compared them to eachother, loop again
            if [x1, x2] in pairs:
                pass
            #if you already have data for both, but haven't compared them to eachother, correlate and update
            else:
                c = ss.pearsonr(shank[x1,:], shank[x2,:])[0]
                pairs.append([x1, x2])
                if math.isnan(c) == False:
                    cor_store.append(c)
        #if you have data for 1 but not the other, get counts, correlate and update
        elif ((x1 in ya) & ((x2 in ya) == False)):
            INT2 = t[np.where(r == Neurons[x2])[0]]
            for j in range(len(bins)-1):
                shank[x2, j] = len(np.where((bins[j] <= INT2) & (INT2 < bins[j+1]))[0])
            c = ss.pearsonr(shank[x1,:], shank[x2,:])[0]
            ya.append(x2)
            pairs.append([x1, x2])
            if math.isnan(c) == False:
                cor_store.append(c)
        #if you have data for the other but not the 1, get counts, correlate and update
        elif ((x2 in ya) & ((x1 in ya) == False)):
            INT2 = t[np.where(r == Neurons[x1])[0]]
            for j in range(len(bins)-1):
                shank[x1, j] = len(np.where((bins[j] <= INT2) & (INT2 < bins[j+1]))[0])
            c = ss.pearsonr(shank[x1,:], shank[x2,:])[0]
            ya.append(x1)
            pairs.append([x1, x2])
            if math.isnan(c) == False:
                cor_store.append(c)
        #if neither is in the list, get counts, correlate, and update
        else:
            INT1 = t[np.where(r == Neurons[x1])[0]]
            INT2 = t[np.where(r == Neurons[x2])[0]]
            for j in range(len(bins)-1):
                x1x = np.where((bins[j] <= INT1) & (INT1 < bins[j+1]))[0]
                x2x = np.where((bins[j] <= INT2) & (INT2 < bins[j+1]))[0]
                shank[x1, j] = len(x1x)
                shank[x2, j] = len(x2x)
            ya.append(x1)
            ya.append(x2)
            pairs.append([x1, x2])
            c = ss.pearsonr(shank[x1,:], shank[x2,:])[0]
            if math.isnan(c) == False:
                cor_store.append(c)
    return np.mean(cor_store)

#Right hand side function that plays nice with odeint
def BW_RHS_ode_V(t, stvars, args):
    stvars = stvars.reshape(5, args[2])
    #Sodium
    bam = -0.1*(stvars[0] + 35.)
    a_m = bam/(np.exp(bam)-1.)
    b_m = 4.*np.exp(-(stvars[0] + 60.)/18.)
    m_inf = a_m/(a_m + b_m)
    a_h = 0.07*np.exp(-(stvars[0] + 58.)/20.)
    b_h = 1./(1. + np.exp(-0.1*(stvars[0] + 28.)))
    #Potassium
    ban = stvars[0] + 34.
    a_n = -0.01*ban/(np.exp(-0.1*ban) - 1.)
    b_n = 0.125*np.exp(-(stvars[0] + 44.)/80.)
    #M-current
    M_inf = 1./(1 + np.exp(-(stvars[0] - (-10))/10.)) #v_half set to -10; Slope set to 10.; may need adjustment
    #Synapse
    fvp = 1./(1. + sp.exp(-stvars[0,:]/2.))
    I_syn = np.einsum('ij,ij->i', W, (stvars[3,:] * (stvars[0,:,np.newaxis] - E_syn)))
    #Currents
    I_Na = 35. * (m_inf**3.) * stvars[1] * (stvars[0] - 55.)
    I_K = 9. * (stvars[2]**4.) * (stvars[0] + 90.)
    I_L = 0.1*(stvars[0] + 65.)
    I_M = args[3] * stvars[4] * (stvars[0] + 90.)
    #derivatives; all currents negative by convention
    dVdt = args[0] - I_M - I_syn - I_Na - I_K - I_L
    dhdt = 5. * (a_h*(1. - stvars[1]) - (b_h*stvars[1]))
    dndt = 5. * (a_n*(1. - stvars[2]) - (b_n*stvars[2]))
    dsdt = 12.*fvp*(1.-stvars[3,:]) - (0.1*stvars[3,:]) #outgoing synapse
    dmdt = (M_inf - stvars[4])/1500. #tau_a~O(1s)

    return np.concatenate([dVdt, dhdt, dndt, dsdt, dmdt])

def von_mises_dist(x, k, mu, N):
    a = sp.exp(k*np.cos(x-mu))/(N*sp.i0(k))
    return a

def weights(Ne,Ni,kee,kei,kie_L,kii,Aee,Aei,Aie_L,Aii,pee,pei,pie,pii):
    wee = np.zeros((Ne,Ne))
    wei = np.zeros((Ne,Ni))
    wie = np.zeros((Ni,Ne))
    wii = np.zeros((Ni,Ni))
    we = 2*np.arange(Ne)*np.pi/Ne
    wi = 2*np.arange(Ni)*np.pi/Ni
    for i in range(Ne):
        wee[i,:]=Aee*np.roll(von_mises_dist(we, kee, 0, Ne), i)
        wei[i,:]=Aei*np.roll(von_mises_dist(wi, kei, 0, Ni),int(round((Ni*i)/Ne))) #\
    for i in range(Ni):
        wie[i,:]=Aie_L*np.roll(von_mises_dist(we, kie_L, 0, Ne),int(round((Ne*i)/Ni)))# + Aie_D*circshift(von_mises_dist(we, kie_D, pi, Ne),div(Ne*i,Ni)-1)
        wii[i,:]=Aii*np.roll(von_mises_dist(wi, kii, 0, Ni),i)
    wee = wee*(np.random.uniform(0, 1, (Ne,Ne))<pee)
    wei = wei*(np.random.uniform(0, 1, (Ne,Ni))<pei)
    wie = wie*(np.random.uniform(0, 1, (Ni,Ne))<pie)
    wii = wii*(np.random.uniform(0, 1, (Ni,Ni))<pii)
    ###note no sign introduced yet in this version
    return wee, wei, wie, wii


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
        ieNL1_inds = rand(Ne_local, Ne2, k)
        ieNL2_inds = rand(0, Ne_local, k)
        for j in range(k):
            W[i, Ie1_inds[j]] += Jie
            W[i, ii1_inds[j]] += Jii
            W[i+Ni_local, ie2_inds[j]] += Jie
            W[i+Ni_local, ii2_inds[j]] += Jii
            W[i, ieNL1_inds[j]] += Jie_NL
            W[i+Ni_local, IeNL2_inds[j]] += Jie_NL

    return W






#################### PARAMETERS ####################

Ne = 400
Ni = 100
N = Ne+Ni

pee = .2#float(sys.argv[1])
pei = .2#float(sys.argv[2])
pie = .2#float(sys.argv[3])
pii = .2#float(sys.argv[4])

Aee = .7#float(sys.argv[5])
Aei = 1.5#float(sys.argv[6])
Aie = 2.0#float(sys.argv[7])
Aii = 2.5#float(sys.argv[8])

kee = 2.#float(sys.argv[9])
kei = 1.25#1.5#float(sys.argv[10])
kie = .5#float(sys.argv[11])
kii = .5#float(sys.argv[12])

stim = .25#float(sys.argv[13])
g_adaptation = .5#float(sys.argv[14])

#first sim these neurons were ~asynchronous, regular, but with waves which boosted irregularity
#next I tried making kee more broad to see if it synchronized, or if it left us in AR

# N=500
# Ne=400
# Ni=100

#################### MATRIX, I_APP, IV, TIME ####################

wee, wei, wie, wii = weights(Ne, Ni, kee, kei, kie, kii, Aee, Aei, Aie, Aii, pee, pei, pie, pii)
W = np.zeros((N,N))
W[:Ne, :Ne] = wee
W[:Ne, Ne:] = wei
W[Ne:, :Ne] = wie
W[Ne:, Ne:] = wii

W_sparse = sp.sparse.csc_matrix(W)

InitialValues = np.zeros(5*N) #4 state variables include voltage, sodium, potassium, and synapse
InitialValues[:N] = np.random.uniform(-70, -50, N) #random initial conditions for voltage
InitialValues[N:3*N] = 1. #sodium and potassium start at this value
InitialValues[4*N:] = np.random.uniform(0, .1, N)

E_syn = np.zeros(N)
E_syn[Ne:] = -75. #inhibitory synapses have a reversal in the driving force

#Drive select neurons
I_app = np.zeros(Ne)
FPe = round(Ne/5.)
P1s = round((Ne/4.)-FPe)
P1e = round((Ne/4.)+FPe)
P2s = round((3.*Ne/4.)-FPe)
P2e = round((3.*Ne/4.)+FPe)

I_app[P1s:P1e] = stim
I_app[P2s:P2e] = stim
I_app = np.concatenate([I_app, np.zeros(Ni)])

g_adapt = np.zeros(N)
g_adapt[:Ne] = g_adaptation

IV = InitialValues
runtime = 10000. #ms
transient = 100. #ms
stime = (runtime-transient)/1000.
h = 0.01
t = np.linspace(0, runtime, runtime/h)

min_spikes = stime
min_neurons = Ne/10.
cbinsize = 1000
fbinsize = 50/h
netd_binsize = 250/h
half = int(round(Ne/2))
#################### SIMULATE ####################
args = [I_app, W_sparse, N, g_adapt]
v = np.zeros((N*5, len(t)+1))
r = ode(BW_RHS_ode_V).set_integrator('vode', method='bdf', rtol=1.49e-8)
r.set_initial_value(IV).set_f_params(args)
c=0
start_time = TIME.time()
while r.successful() and r.t < runtime:
    #print r.integrate(r.t + h)
    v[:,c] = r.integrate(r.t + h)
    c+=1
stop_time = TIME.time()
ntotal = runtime/h
te, re = build_raster(v[:Ne, :])
ti, ri = build_raster(v[Ne:,:])
sig = nt_diff(te, re, ntotal, half, netd_binsize)
flags, times = WLD(sig)
top, tdom, bot, bdom, nmz, tnmz = splice_flags(flags, times, netd_binsize)
Neurons = neuron_finder(re, 10, 100)
TN = [i for i in Neurons if i >= half]
BN = [i for i in Neurons if i < half]
t2, f2 = splice_reversions(flags, times)
d = np.diff(np.array(netd_binsize)*t2)
fw = np.array([i for i in range(len(f2)) if f2[i] == 'win'])
fl = np.array([i for i in range(len(f2)) if f2[i] == 'lose'])
MDT = np.mean(d[fw])
MDB = np.mean(d[fl])
CVD = CV(d)
tbf, rbf = ligase(bot, te, re, BN)
ttf, rtf = ligase(top, te, re, TN)
CVSTW = build_CV(te, re, TN, top)
CVSBW = build_CV(te, re, BN, bot)
FFST = pool_FF(TN, ttf, rtf, fbinsize)
FFSB = pool_FF(BN, tbf, rbf, fbinsize)
cwT = rand_pair_cor(cbinsize, ttf, rtf, TN, 1000)
cwB = rand_pair_cor(cbinsize, tbf, rbf, BN, 1000)

text =  "##RESULT {0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}, {11}, {12}, {13}, {14}, {15}, {16}, {17}, {18}, {19}, {20}, {21}, {22}\n".format(MDT, MDB, CVD, cwT, cwB, np.mean(CVSTW), np.mean(CVSBW), np.mean(FFST), np.mean(FFSB), pee, pei, pie, pii, Aee, Aei, Aie, Aii, kee, kei, kie, kii, stim, g_adaptation)

newfile.write(text)
print(text)











#
