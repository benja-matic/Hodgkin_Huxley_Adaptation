#demonstrate
import numpy as np
import scipy as sp
import scipy.stats as ss
import pylab as plt
from scipy.integrate import ode
from scipy.integrate import odeint
import time as TIME
import math
import time as TIME

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

def BW_RHS(stvars, args):
    #I_app = 0. + np.random.normal()*2.5
    bam = -0.1*(stvars[0] + 35.)
    a_m = bam/(np.exp(bam)-1.)
    b_m = 4.*np.exp(-(stvars[0] + 60.)/18.)
    m_inf = a_m/(a_m + b_m)
    a_h = 0.07*np.exp(-(stvars[0] + 58.)/20.)
    b_h = 1./(1. + np.exp(-0.1*(stvars[0] + 28.)))

    ban = stvars[0] + 34.
    a_n = -0.01*ban/(np.exp(-0.1*ban) - 1.)
    b_n = 0.125*np.exp(-(stvars[0] + 44.)/80.)

    I_Na = 35. * (m_inf**3.) * stvars[1] * (stvars[0] - 55.)
    I_K = 9. * (stvars[2]**4.) * (stvars[0] + 90.)
    I_L = 0.1*(stvars[0] + 65.)

    dVdt = args - I_Na - I_K - I_L
    dhdt = 5. * (a_h*(1. - stvars[1]) - (b_h*stvars[1]))
    dndt = 5. * (a_n*(1. - stvars[2]) - (b_n*stvars[2]))

    return np.array([dVdt, dhdt, dndt])

def BW_RHS_odeint(stvars, t, args):
    #I_app = 0. + np.random.normal()*2.5
    bam = -0.1*(stvars[0] + 35.)
    a_m = bam/(np.exp(bam)-1.)
    b_m = 4.*np.exp(-(stvars[0] + 60.)/18.)
    m_inf = a_m/(a_m + b_m)
    a_h = 0.07*np.exp(-(stvars[0] + 58.)/20.)
    b_h = 1./(1. + np.exp(-0.1*(stvars[0] + 28.)))

    ban = stvars[0] + 34.
    a_n = -0.01*ban/(np.exp(-0.1*ban) - 1.)
    b_n = 0.125*np.exp(-(stvars[0] + 44.)/80.)

    I_Na = 35. * (m_inf**3.) * stvars[1] * (stvars[0] - 55.)
    I_K = 9. * (stvars[2]**4.) * (stvars[0] + 90.)
    I_L = 0.1*(stvars[0] + 65.)

    dVdt = args - I_Na - I_K - I_L
    dhdt = 5. * (a_h*(1. - stvars[1]) - (b_h*stvars[1]))
    dndt = 5. * (a_n*(1. - stvars[2]) - (b_n*stvars[2]))

    return np.array([dVdt, dhdt, dndt])

def BW_RHS_odeint_V(stvars, t, args):
    stvars = stvars.reshape(3, 10)
    #I_app = 0. + np.random.normal()*2.5
    bam = -0.1*(stvars[0] + 35.)
    a_m = bam/(np.exp(bam)-1.)
    b_m = 4.*np.exp(-(stvars[0] + 60.)/18.)
    m_inf = a_m/(a_m + b_m)
    a_h = 0.07*np.exp(-(stvars[0] + 58.)/20.)
    b_h = 1./(1. + np.exp(-0.1*(stvars[0] + 28.)))

    ban = stvars[0] + 34.
    a_n = -0.01*ban/(np.exp(-0.1*ban) - 1.)
    b_n = 0.125*np.exp(-(stvars[0] + 44.)/80.)

    I_Na = 35. * (m_inf**3.) * stvars[1] * (stvars[0] - 55.)
    I_K = 9. * (stvars[2]**4.) * (stvars[0] + 90.)
    I_L = 0.1*(stvars[0] + 65.)

    dVdt = args - I_Na - I_K - I_L
    dhdt = 5. * (a_h*(1. - stvars[1]) - (b_h*stvars[1]))
    dndt = 5. * (a_n*(1. - stvars[2]) - (b_n*stvars[2]))


    return np.concatenate([dVdt, dhdt, dndt])

def BW_RHS_ode(t, stvars, args):
    #I_app = 0. + np.random.normal()*2.5
    bam = -0.1*(stvars[0] + 35.)
    a_m = bam/(np.exp(bam)-1.)
    b_m = 4.*np.exp(-(stvars[0] + 60.)/18.)
    m_inf = a_m/(a_m + b_m)
    a_h = 0.07*np.exp(-(stvars[0] + 58.)/20.)
    b_h = 1./(1. + np.exp(-0.1*(stvars[0] + 28.)))

    ban = stvars[0] + 34.
    a_n = -0.01*ban/(np.exp(-0.1*ban) - 1.)
    b_n = 0.125*np.exp(-(stvars[0] + 44.)/80.)

    I_Na = 35. * (m_inf**3.) * stvars[1] * (stvars[0] - 55.)
    I_K = 9. * (stvars[2]**4.) * (stvars[0] + 90.)
    I_L = 0.1*(stvars[0] + 65.)

    dVdt = args - I_Na - I_K - I_L
    dhdt = 5. * (a_h*(1. - stvars[1]) - (b_h*stvars[1]))
    dndt = 5. * (a_n*(1. - stvars[2]) - (b_n*stvars[2]))

    return np.array([dVdt, dhdt, dndt])

def BW_RHS_ode_V(t, stvars, args):
    stvars = stvars.reshape(3, 10)
    #I_app = 0. + np.random.normal()*2.5
    bam = -0.1*(stvars[0] + 35.)
    a_m = bam/(np.exp(bam)-1.)
    b_m = 4.*np.exp(-(stvars[0] + 60.)/18.)
    m_inf = a_m/(a_m + b_m)
    a_h = 0.07*np.exp(-(stvars[0] + 58.)/20.)
    b_h = 1./(1. + np.exp(-0.1*(stvars[0] + 28.)))

    ban = stvars[0] + 34.
    a_n = -0.01*ban/(np.exp(-0.1*ban) - 1.)
    b_n = 0.125*np.exp(-(stvars[0] + 44.)/80.)

    I_Na = 35. * (m_inf**3.) * stvars[1] * (stvars[0] - 55.)
    I_K = 9. * (stvars[2]**4.) * (stvars[0] + 90.)
    I_L = 0.1*(stvars[0] + 65.)

    dVdt = args[0] - I_Na - I_K - I_L
    dhdt = 5. * (a_h*(1. - stvars[1]) - (b_h*stvars[1]))
    dndt = 5. * (a_n*(1. - stvars[2]) - (b_n*stvars[2]))


    return np.concatenate([dVdt, dhdt, dndt])


def driver(RHS, IV, method, time, h, args):
    stvars = IV
    vhist = np.zeros(len(time))
    for i in range(len(time)):
        vhist[i] = stvars[0]
        stvars = method(RHS, stvars, h, args)

    return vhist

#RHS = BW_RHS #equations
# stepper = RK4 #solver
# stepper = euler
stepper = euler
N = 10
IV = np.array([-65., 0., 0.]) #initial values
IVV = np.zeros(N*3)
IVV[:N] = np.random.uniform(-50, -70, N)
h = 0.01 #ms
runtime = 100 #ms
time = np.arange(0.0, runtime, h) #time domain
drive = 1. #applied current
args = (drive, 3.) #model parameters
# #vhist = driver(RHS, IV, stepper, time, h, args) #GO!
#
# start_time = TIME.time()
# vhist = odeint(BW_RHS_odeint_V, IVV, time, args = (1.,))
# stop_time = TIME.time()
# print "odeint for 10 neurons: ", stop_time - start_time
# plt.plot(vhist[:, 0])
# plt.plot(vhist[:, 1])

t = np.linspace(0, 1000, 10)
start_time = TIME.time()
r = ode(BW_RHS_ode_V).set_integrator('vode', method='bdf')
r.set_initial_value(IVV).set_f_params(args)
v = np.zeros((N*3, len(time)+1))
c=0
while r.successful() and r.t < runtime:
    #print "r[10]: ", r.integrate(r.t + h)[10]
    v[:,c] = r.integrate(r.t + h)
    c+=1

stop_time = TIME.time()
print "vode bdf for 10 neurons: ", stop_time - start_time

plt.plot(v[0,:])
plt.title("Single BW Neuron With Constant Input", fontsize = 48)
plt.xlabel("Time (" + str(h) + " ms)", fontsize = 36)
plt.ylabel("Voltage (mV)", fontsize = 36)
plt.show()


'''
t = np.linspace(0, 1000, runtime/h)
start_time = TIME.time()
r = ode(BW_RHS_ode).set_integrator('zvode', method='bdf')
r.set_initial_value(IV).set_f_params(1.)
v = np.zeros(len(time)+1)
c=0
while r.successful() and r.t < runtime:
    v[c] = r.integrate(r.t + h)[0]
    c+=1

stop_time = TIME.time()
print "zvode bdf time: ", stop_time - start_time
plt.plot(v)

print "next...\n"

start_time = TIME.time()
r = ode(BW_RHS_ode).set_integrator('vode', method='bdf')
r.set_initial_value(IV).set_f_params(1.)
v = np.zeros(len(time)+1)
c=0
while r.successful() and r.t < runtime:
    v[c] = r.integrate(r.t + h)[0]
    c+=1
stop_time = TIME.time()
print "vode bdf time: ", stop_time - start_time
plt.plot(v)

print "next...\n"

start_time = TIME.time()
r = ode(BW_RHS_ode).set_integrator('lsoda', method='bdf')
r.set_initial_value(IV).set_f_params(1.)
v = np.zeros(len(time)+1)
c=0
while r.successful() and r.t < runtime:
    v[c] = r.integrate(r.t + h)[0]
    c+=1
stop_time = TIME.time()
print "lsoda bdf time: ", stop_time - start_time
plt.plot(v)

print "next...\n"

start_time = TIME.time()
r = ode(BW_RHS_ode).set_integrator('dopri5')
r.set_initial_value(IV).set_f_params(1.)
v = np.zeros(len(time)+1)
c=0
while r.successful() and r.t < runtime:
    v[c] = r.integrate(r.t + h)[0]
    c+=1
stop_time = TIME.time()
print "dopri5 time: ", stop_time - start_time
plt.plot(v)

print "next...\n"

start_time = TIME.time()
vhist = odeint(BW_RHS_odeint, IV, t, args = (1.,))
stop_time = TIME.time()
print "odeint time: ", stop_time - start_time
plt.plot(v)
'''
###NOW SEE IF WE CAN VECTORIZE vodeBDF



# start_time = TIME.time()
# vhist = driver(BW_RHS, IV, euler, time, h, args) #GO!
# stop_time = TIME.time()
# print "Euler time: ", stop_time - start_time
#
#
# start_time = TIME.time()
# vhist = driver(BW_RHS, IV, RK2, time, h, args) #GO!
# stop_time = TIME.time()
# print "RK2 time: ", stop_time - start_time
# # def fun(y, t, a):
#     f = a*y
#     return f
#
# y0 = 100.
# t = np.linspace(0, 1, 51)
# a = -2.5
# y = odeint(fun, y0, t, args=(a,))


































# plt.plot(vhist)
# plt.title("Voltage Trace of a Buzsaki-Wang Neuron", fontsize = 64)
# plt.xlabel("Time (0.01ms; total = 1s)", fontsize = 36)
# plt.ylabel("Voltage (mV)", fontsize = 36)
# plt.xticks(np.arange(0, 100000, 10000), fontsize = 24)
# plt.yticks(fontsize = 24)
# plt.show()
