import numpy as np
import scipy as sp
import scipy.stats as ss
import pylab as plt
from scipy.integrate import odeint
import time as TIME
import math
import time as TIME

'''
Simulate network of Buzsaki-Wang neurons using scipy odeint, which calls LSODA, which alternates between Adams and Gear's methods
Neurons 0:Ne are excitatory, Ne:N are inhibitory
opeint requires a flat array, so initial values are 4*N long
RHS reshapes this vector, computes derivatives, reshapes it again, and returns it to VODE
'''

#for now just detect zero crossings; implement shmitt trigger later
def basic_spike_detector(V):
    spikes = []
    for i in range(len(V)):
        if V[i] >= 0:
            if V[i-1] < 0:
                spikes.append(i)
    return np.array(spikes)

#Right hand side function that plays nice with odeint
def BW_RHS_odeint(stvars, t, I_app, W, N, E_syn):
    #odeint requires 1d arrays, state variables unpacked %N
    #reshaping and defining had marginally faster performance than indexing %N in each array
    stvars = stvars.reshape(4, N) #V, h, n, s
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
    x = np.concatenate([dVdt, dhdt, dndt, dsdt])
    #print "new shape of dvs: ", np.shape(x)
    return x

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

Ne = 400
Ni = 100
N = Ne+Ni

Aee = 3.
Aei = 10.
Aie = 20.
Aii = 5.

kee = 2.
kei = .3
kie = .3
kii = .3

stim = 1.
p = .2
h = 0.01

#################### MATRIX, I_APP, IV, TIME ####################

wee, wei, wie, wii = weights(Ne, Ni, kee, kei, kie, kii, Aee, Aei, Aie, Aii, p, p, p, p)
W = np.zeros((N,N))
W[:Ne, :Ne] = wee
W[:Ne, Ne:] = wei
W[Ne:, :Ne] = wie
W[Ne:, Ne:] = wii

InitialValues = np.zeros(4*N) #4 state variables include voltage, sodium, potassium, and synapse
InitialValues[:N] = np.random.uniform(-70, -50, N) #random initial conditions for voltage
InitialValues[N:3*N] = 1.

E_syn = np.zeros(N)
E_syn[Ne:] = -75.

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

RightHandSide = BW_RHS_odeint
IV = InitialValues
runtime = 100. #ms
time = np.arange(0.0, runtime, h) #time domain

#################### SIMULATE ####################
start_time = TIME.time()
vhist = odeint(BW_RHS_odeint, IV, time, args = (I_app, W, N, E_syn,))
stop_time = TIME.time()
print "network simulation time: ", stop_time - start_time

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













#
