#BW2.py
import numpy as np
import scipy as sp
import scipy.stats as ss
import pylab as plt
from scipy.integrate import odeint
import time as TIME
import math

'''
Simualte Network of Buzsaki-Wang neurons using manually implemented solvers
'''

#for now just detect zero crossings; implement shmitt trigger later
def basic_spike_detector(V):
    spikes = []
    for i in range(len(V)):
        if V[i] >= 0:
            if V[i-1] < 0:
                spikes.append(i)
    return np.array(spikes)

def CV(d):
    m = np.mean(d)
    s = np.std(d)
    return s/m

def FANO(A):
    return np.var(A)/np.mean(A)

#keep only spikes after transient; assumes raster stored conventionally
def kill_transient(t, r, trans):
    m = np.where(t > trans)[0]
    return t[m], r[m]

#get the set of neurons satisfying criteria so you don't have to search again
def neuron_finder(r, min_spikes, min_neurons):
    Neurons, Counts = np.unique(r, return_counts = True)
    N = []
    for i in range(len(Neurons)):
        if Counts[i] > min_spikes:
            N.append(i)
    if len(N) > min_neurons:
        return N
    else:
        return -5.
#assumes conventional raster format, computes CV_isi
def gather_CV(t, r, Neurons):
    CVS = np.zeros(len(Neurons))
    for i in range(len(Neurons)):
        m = np.where(r == Neurons[i])[0]
        T = t[m]
        D = np.diff(T)
        cv = CV(D)
        CVS[i] = cv
    return CVS

#time is fbinsize-d chunks from some start (after transient) to stop (end of sim, or last spike)
#spikes is spike times for this neuron
def grid_FF(spikes, time):
    counts = np.zeros(len(time)-1)
    for i in range(len(counts)):
        x = len(np.where((spikes > time[i]) & (spikes < time[i+1]))[0])
        counts[i] = x
    return np.var(counts)/np.mean(counts)

#build raster and do statistics off the cuff
#stores raster in 'special raster' (SR) format:
#   index is neuron number
#   te[i] is spike times of neuron i
#   re[i] is an array of i's for plotting convenience
def build_raster_plus(V, Ne, Ni, transient, runtime, mtime, min_spikes, min_neurons, fbinsize):
    te = []
    re = []
    ti = []
    ri = []
    half = round(Ne/2.)
    top_neurons = []
    bot_neurons = []
    Rates = []
    FFS = []
    CVS = []
    FF_time = np.arange(transient, runtime, fbinsize)
    for i in range(Ne):
        nes = basic_spike_detector(V[i,:]) #spikes
        nest = [k for k in nes if k > transient]
        te.append(np.array(nest))
        re.append(np.zeros(len(nest)) + i)
        Rates.append(len(nest)/mtime) #rate
        if len(nest) > min_spikes:
            fano = grid_FF(nest, FF_time) #fano
            FFS.append(fano)
            d = np.diff(nest)
            cv = CV(d) #cv
            CVS.append(cv)
            if i >= half:
                top_neurons.append(i)
            else:
                bot_neurons.append(i)
    #     for j in range(len(nest)):
    #         te.append(nest[j])
    #         re.append(i)
    # for i in range(Ne, Ne+Ni):
    #     nis = basic_spike_detector(V[i,:])
    #     nist = [l for l in nis if l > transient]
    #     for j in range(len(nist)):
    #         ti.append(nist[j])
    #         ri.append(i)
    if len(bot_neurons) + len(top_neurons) < min_neurons:
        return np.zeros(9) -5.
    else:
        return te, re, top_neurons, bot_neurons, CVS, FFS, Rates

#randomly sample neurons, calculate count trains, cross correlate them
#stores count-trains for neurons already computed, skips pairs already done together
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

#As above, but takes the 'special raster' format returned by build_raster_plus
def rand_pair_cor_SR(start, stop, bin_size, t, r, Neurons, n):
    bins = np.arange(start, stop +1., bin_size)
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
                #print "compared these two together already"
                pass
            #if you already have data for both, but haven't compared them to eachother, correlate and update
            else:
                c = ss.pearsonr(shank[x1,:], shank[x2,:])[0]
                pairs.append([x1, x2])
                if math.isnan(c) == False:
                    cor_store.append(c)
                    #print c
        #if you have data for 1 but not the other, get counts, correlate and update
        elif ((x1 in ya) & ((x2 in ya) == False)):
            INT2 = t[Neurons[x2]]
            for j in range(len(bins)-1):
                shank[x2, j] = len(np.where((bins[j] <= INT2) & (INT2 < bins[j+1]))[0])
            c = ss.pearsonr(shank[x1,:], shank[x2,:])[0]
            ya.append(x2)
            pairs.append([x1, x2])
            if math.isnan(c) == False:
                cor_store.append(c)
                #print c
        #if you have data for the other but not the 1, get counts, correlate and update
        elif ((x2 in ya) & ((x1 in ya) == False)):
            INT2 = t[Neurons[x1]]
            for j in range(len(bins)-1):
                shank[x1, j] = len(np.where((bins[j] <= INT2) & (INT2 < bins[j+1]))[0])
            c = ss.pearsonr(shank[x1,:], shank[x2,:])[0]
            ya.append(x1)
            pairs.append([x1, x2])
            if math.isnan(c) == False:
                cor_store.append(c)
                #print c
        #if neither is in the list, get counts, correlate, and update
        else:
            INT1 = t[Neurons[x1]]
            INT2 = t[Neurons[x2]]
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
                #print c
    #print cor_store
    #print shank
    return np.mean(cor_store)

#WTA_ness and which pool was winning
def bias(re, Ne):
    half = round(Ne/2.)
    tot = float(len(re))
    top = len(np.where((re > round(Ne/2.)) & (re < Ne))[0])
    bot = len(np.where(re < half)[0])
    if top > bot:
        bias = 1
        win = top
    else:
        bias = 2
        win = bot
    return bias, win/tot

#same, but using SR format
def bias_SR(te):
    half = int(round((len(te)/2.)))
    bot = float(sum([len(te[i]) for i in range(half)]))
    top = float(sum([len(te[i]) for i in range(half, len(te))]))
    tot = top + bot
    if top > bot:
        bias = 1
        win = top
    else:
        bias = 2
        win = bot
    return bias, win/tot

### Numerical Methods ###

def euler(RHS, stvars, h, args):
    stdevs = RHS(stvars, args)
    return stvars + h*stdevs

def RK2(RHS, stvars, h, args):
    k1 = RHS(stvars, args)
    h2 = h/2.
    k2 = RHS(stvars + h2*k1, args)
    return stvars + h*k2

def RK4(RHS, stvars, h, args):
    h2 = h/2.
    h6 = h/6.
    k1 = RHS(stvars, args)
    k2 = RHS(stvars + h2*k1, args)
    k3 = RHS(stvars + h2*k2, args)
    k4 = RHS(stvars + h*k3, args)
    return stvars + h6*(k1 + k2 + k3 +k4)

### Right hand side function
def BW_RHS(stvars, args):
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
    #synapse
    fvp = 1./(1. + sp.exp(-stvars[0,:]/2.))
    I_syn = np.inner(W, stvars[3,:]) * (stvars[0,:] - E_syn)
    #Currents
    I_Na = 35. * (m_inf**3.) * stvars[1] * (stvars[0] - 55.)
    I_K = 9. * (stvars[2]**4.) * (stvars[0] + 90.)
    I_L = 0.1*(stvars[0] + 65.)
    #Derivatives
    dVdt = I_app + I_syn - I_Na - I_K - I_L
    dhdt = 5. * (a_h*(1. - stvars[1]) - (b_h*stvars[1]))
    dndt = 5. * (a_n*(1. - stvars[2]) - (b_n*stvars[2]))
    dsdt = 12.*fvp*(1.-stvars[3,:]) - (0.1*stvars[3,:]) #outgoing synapse
    return np.array([dVdt, dhdt, dndt, dsdt])

#looper that stores state variables (ugly for now)
def driver_8(RHS, IV, method, time, h, args):
    stvars = IV
    nv = len(IV)
    nn = len(IV[0,:])
    vhist = np.zeros((nn*4, len(time)))
    for i in range(len(time)):
        vhist[:nn,i] = stvars[0,:]
        vhist[nn:nn*2,i] = stvars[1,:]
        vhist[nn*2:nn*3,i] = stvars[2,:]
        vhist[nn*3:nn*4,i] = stvars[3,:]
        stvars = method(RHS, stvars, h, args)
    return vhist

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
    return wee, -wei, wie, -wii

#################### PARAMETERS ####################

Ne = 400 #800
Ni = 100 #200
N = Ne+Ni

Aee = 3.
Aei = 10. #150
Aie = 20. #500
Aii = 5. #400

kee = 2.
kei = .3
kie = .3
kii = .3

stim = 1.
p = .2

#################### MATRIX, I_APP, IV, TIME ####################

wee, wei, wie, wii = weights(Ne, Ni, kee, kei, kie, kii, Aee, Aei, Aie, Aii, p, p, p, p)
InitialValues = np.zeros((4,Ne+Ni))
InitialValues[0,:] = np.random.uniform(-70, -50, Ne+Ni)
InitialValues[1:2,:] = 1.

RightHandSide = BW_RHS
method = euler
transient = 1000. #ms
runtime = 100. #ms
stime = (runtime-transient)/1000.
h = 0.001 #ms
time = np.arange(0.0, runtime, h)

#set up drive to specific neurons, pass in under args
I_app = np.zeros(Ne)
FPe = round(Ne/5.)
P1s = round((Ne/4.)-FPe)
P1e = round((Ne/4.)+FPe)
P2s = round((3.*Ne/4.)-FPe)
P2e = round((3.*Ne/4.)+FPe)

W = np.zeros((N,N))
W[:Ne, :Ne] = wee
W[:Ne, Ne:] = wei
W[Ne:, :Ne] = wie
W[Ne:, Ne:] = wii

I_app[P1s:P1e] = stim
I_app[P2s:P2e] = stim
I_app = np.concatenate([I_app, np.zeros(Ni)])

#################### SIMULATE ####################

args = [I_app, Ne, Ni, wee, wei, wie, wii]
start_time = TIME.time()
vhist = driver_8(RightHandSide, InitialValues, method, time, h, args)
stop_time = TIME.time()
print "simulation time: ", stop_time - start_time

fig, (a0, a1) = plt.subplots(2,1, figsize = (24,18))
for i in range(N):
    spikes = basic_spike_detector(vhist[:,i])
    if i <= Ne:
        a0.plot(spikes, np.zeros(len(spikes)) + i, "g.")
    else:
        a1.plot(spikes, np.zeros(len(spikes)) + i - Ne, "b.")


n0 = N/4

a0.set_title("Raster Plot of Network", fontsize = 54)
a0.tick_params(labelsize = 24)
a0.set_ylabel("Excitatory Neuron #", fontsize = 36)
a0.set_yticks([Ne/4., Ne/2., Ne*.75, Ne])
a1.tick_params(labelsize = 24)
a1.set_ylabel("Inhibitory Neuron #", fontsize = 36)
a1.set_yticks([Ni/4., Ni/2., Ni*.75, Ni])
a1.set_xlabel("Time (" + str(h) + " ms)", fontsize = 48)

fig2, ax = plt.subplots(4,1)
ax[0].set_title("State Variables for Single Neuron", fontsize = 54)
ax[0].plot(v[n0,:])
ax[0].set_ylabel("Voltage (mV)", fontsize = 24)
ax[1].plot(v[N+n0,:])
ax[1].set_ylabel("Sodium Activation", fontsize = 24)
ax[2].plot(v[n0 + (N*2),:])
ax[2].set_ylabel("Potassium Activation", fontsize = 24)
ax[3].plot(v[n0 + (N*3),:])
ax[3].set_ylabel("Synapse Activation", fontsize = 24)
ax[3].set_xlabel("Time (" + str(h) + " ms)", fontsize = 48)

plt.show()
# a0.set_title("Raster Plot of Network", fontsize = 54)
# a0.tick_params(labelsize = 24)
# a0.set_ylabel("Excitatory Neuron #", fontsize = 36)
# a0.set_yticks([Ne/4., Ne/2., Ne*.75, Ne])
# a1.tick_params(labelsize = 24)
# a1.set_ylabel("Inhibitory Neuron #", fontsize = 36)
# a1.set_yticks([Ni/4., Ni/2., Ni*.75, Ni])
# a1.set_xlabel("Time (0.02 ms)", fontsize = 48)
#
# args = [I_app, Ne, Ni, wee, wei, wie, wii]
# start_time = TIME.time()
# vhist = driver_8(RightHandSide, InitialValues, method, time, h, args)
# stop_time = TIME.time()
# print "standard time: ", stop_time - start_time
#
# fig, (a4, a5) = plt.subplots(2,1, figsize = (24,18))
# for i in range(Ne+Ni):
#     spikes = basic_spike_detector(vhist[i,:])
#     if i <= Ne:
#         a4.plot(spikes, np.zeros(len(spikes)) + i, "g.")
#     else:
#         a5.plot(spikes, np.zeros(len(spikes)) + i-Ne, "b.")
#
#
# a4.set_title("Raster Plot of Network", fontsize = 54)
# a4.tick_params(labelsize = 24)
# a4.set_ylabel("Excitatory Neuron #", fontsize = 36)
# a4.set_yticks([Ne/4., Ne/2., Ne*.75, Ne])
# a5.tick_params(labelsize = 24)
# a5.set_ylabel("Inhibitory Neuron #", fontsize = 36)
# a5.set_yticks([Ni/4., Ni/2., Ni*.75, Ni])
# a5.set_xlabel("Time (0.02 ms)", fontsize = 48)
#
# RightHandSide = BW_RHS_V_SIO3
# method = euler
# transient = 1000. #ms
# runtime = 2000. #ms
# stime = (runtime-transient)/1000.
# h = 0.01 #ms
# time = np.arange(0.0, runtime, h)
# args = [I_app, W]
# start_time = TIME.time()
# vhist = driver_8(RightHandSide, InitialValues, method, time, h, args)
# stop_time = TIME.time()
# print "matrix time: ", stop_time - start_time
#
#
# fig2, (a2, a3) = plt.subplots(2,1, figsize = (24,18))
# for i in range(Ne+Ni):
#     spikes = basic_spike_detector(vhist[i,:])
#     if i <= Ne:
#         a2.plot(spikes, np.zeros(len(spikes)) + i, "g.")
#     else:
#         a3.plot(spikes, np.zeros(len(spikes)) + i-Ne, "b.")
#
#
# a2.set_title("Raster Plot of Network", fontsize = 54)
# a2.tick_params(labelsize = 24)
# a2.set_ylabel("Excitatory Neuron #", fontsize = 36)
# a2.set_yticks([Ne/4., Ne/2., Ne*.75, Ne])
# a3.tick_params(labelsize = 24)
# a3.set_ylabel("Inhibitory Neuron #", fontsize = 36)
# a3.set_yticks([Ni/4., Ni/2., Ni*.75, Ni])
# a3.set_xlabel("Time (0.02 ms)", fontsize = 48)
#
# plt.show()
# for i in range(Ne):
#     spikes = basic_spike_detector(vhist[i,:])
#     # d = np.diff(spikes)
#     # cv_isi = CV(d)
#     # cv.append(cv_isi)
#     plt.plot(spikes, np.zeros(len(spikes)) + i, "g.")
#
# for i in range(Ni):
#     spikes = basic_spike_detector(vhist[i,:])
#     plt.plot(spikes, np.zeros(len(spikes)) + i, "g.")

# plt.title("WTA with BW Neurons", fontsize = 64)
# plt.xlabel("Time (0.01ms; total = 10s)", fontsize = 36)
# plt.ylabel("Neuron # (500 neuron network)", fontsize = 36)
# plt.xticks(np.arange(0, 1000000, 100000), fontsize = 24)
# plt.yticks(np.arange(0, 400, 50), fontsize = 24)
# stop_time = TIME.time()


# print "simulation time: ", stop_time - start_time
#
# stats_time = TIME.time()
# min_spikes = stime
# min_neurons = Ne/10.
# fbinsize = 150/h
# cbinsize = 10/h
#
# fig, (a0, a1) = plt.subplots(2,1, figsize = (24,18))
# for i in range(Ne+Ni):
#     spikes = basic_spike_detector(vhist[i,:])
#     if i <= Ne:
#         a0.plot(spikes, np.zeros(len(spikes)) + i, "g.")
#     else:
#         a1.plot(spikes, np.zeros(len(spikes)) + i-Ne, "b.")
#
#
# a0.set_title("Raster Plot of Network", fontsize = 54)
# a0.tick_params(labelsize = 24)
# a0.set_ylabel("Excitatory Neuron #", fontsize = 36)
# a0.set_yticks([Ne/4., Ne/2., Ne*.75, Ne])
# a1.tick_params(labelsize = 24)
# a1.set_ylabel("Inhibitory Neuron #", fontsize = 36)
# a1.set_yticks([Ni/4., Ni/2., Ni*.75, Ni])
# a1.set_xlabel("Time (0.02 ms)", fontsize = 48)
#
# fig3, ax3 = plt.sublots()
# N = Ne+Ni
# W = np.zeros((N,N))
# W[:Ne, :Ne] = wee
# W[:Ne, Ne:] = wei
# W[Ne:, :Ne] = wie
# W[Ne:, Ne:] = wii
# cax = ax3.imshow(W)
# ax.set_title("Weights Matrix for Neural Network", fontsize = 48)
# cbar = fig.colorbar(cax)
# cbar.ax.tick_params(labelsize = 24)
# cbar.set_label("Peak PSP (mV)", fontsize = 36)
# ax3.tick_params(labelsize = 24)
# ax3.set_ylabel("Presynaptic Neuron #", fontsize = 36)
# ax3.set_yticks([N/4., N/2., .75*N, N])
# ax3.set_xlabel("Postsynaptic Neuron #", fontsize = 36)
# ax3.set_xticks([N/4., N/2., .75*N, N])
#
#
# te, re, top_neurons, bot_neurons, CVS, FFS, Rates = build_raster_plus(vhist, Ne, Ni, transient/h, runtime/h, stime, min_spikes, min_neurons, fbinsize)
#
# if isinstance(te, list) == False:
#     print "##RESULT {0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}".format(-5, -5, -5, -5, -5, -5, -5, -5)
# else:
#     if (len(bot_neurons) < 20) & (len(top_neurons) < 20):
#         print "##RESULT {0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}".format(-5, -5, -5, -5, -5, -5, -5, -5)
#     elif (len(bot_neurons) < 20):
#         top_cor = rand_pair_cor_SR(transient/h, runtime/h, cbinsize, te, re, top_neurons, 1000)
#         bot_cor = -5.
#         mean_rate = np.mean(Rates)
#         max_rate = np.max(Rates)
#         MFF = np.mean([i for i in FFS if math.isnan(i) == False])
#         MCV = np.mean([i for i in CVS if math.isnan(i) == False])
#         winner, wta_ness = bias_SR(te)
#     elif (len(top_neurons) < 20):
#         bot_cor = rand_pair_cor_SR(transient/h, runtime/h, cbinsize, te, re, bot_neurons, 1000)
#         top_cor = -5.
#         mean_rate = np.mean(Rates)
#         max_rate = np.max(Rates)
#         MFF = np.mean([i for i in FFS if math.isnan(i) == False])
#         MCV = np.mean([i for i in CVS if math.isnan(i) == False])
#         winner, wta_ness = bias_SR(te)
#     else:
#         top_cor = rand_pair_cor_SR(transient/h, runtime/h, cbinsize, te, re, top_neurons, 1000)
#         bot_cor = rand_pair_cor_SR(transient/h, runtime/h, cbinsize, te, re, bot_neurons, 1000)
#         mean_rate = np.mean(Rates)
#         max_rate = np.max(Rates)
#         MFF = np.mean([i for i in FFS if math.isnan(i) == False])
#         MCV = np.mean([i for i in CVS if math.isnan(i) == False])
#         winner, wta_ness = bias_SR(te)
#     print "##RESULT {0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}".format(wta_ness, winner, mean_rate, max_rate, MFF, MCV, top_cor, bot_cor)
#
# done = TIME.time()
# print "\n", done - stats_time, "\n"
# print done - start_time
# plt.show()
