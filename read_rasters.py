import numpy as np
import matplotlib.pyplot as plt
import os
import math
import scipy.stats as ss


def read_raster(x):
    with open(x, 'r') as f:
        a = f.read()
    b = a.split('\n')
    b.pop()
    c = [i.split(',') for i in b]
    te = np.array([float(i[0]) for i in c])
    re = np.array([float(i[1]) for i in c])
    return te, re


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

def nt_diff_H(t, r, ntotal, half, netd_binsize):
    netd_bins = np.arange(0, ntotal, netd_binsize)
    ntd = np.zeros(len(netd_bins)-1)
    nts = np.empty_like(ntd)
    for j in range(len(ntd)):
        m = np.where((netd_bins[j] < t) & (t <= netd_bins[j+1]))[0]
        T = sum(r[m] > half)
        c = len(m) - T
        ntd[j] = T-c
        nts[j] = len(m)
    return ntd, nts

def WLD_01(s, tl, th):
    if np.max(s) < tl:
        return ["lose", -1], [0, len(s)]
    elif np.min(s) >= th:
        return ["win", -1], [0, len(s)]
    elif (tl <= np.max(s)) & (th >= np.max(s)) & (tl <= np.min(s)) & (th >= np.min(s)):
        return ["draw", -1], [0, len(s)]
    else:
        times = []
        flags = []
        if s[0] > th:
            flag = "win"
        elif (tl <= s[0]) & (s[0] <= th):
            flag = "draw"
        elif s[0] < tl:
            flag = "lose"
        times.append(0)
        flags.append(flag)
        s2 = s[1:]
        for i in range(len(s2)):
            f1 = comp_01(s2[i], tl, th)
            if f1 != flag:
                flag = f1
                times.append(i+1)
                flags.append(flag)
        times.append(len(s))
        flags.append("end")
        return flags, np.array(times)

def comp_01(x, tl, th):
    if x > th:
        return "win"
    elif tl <= x <= th:
        return "draw"
    elif x < tl:
        return "lose"
    else:
        return "weird"

function WLD_01(s, tl, th)
  if maximum(s) < tl
    return ["lose", "end"], [1, length(s)]
  elseif minimum(s) >= th
    return ["win", "end"], [1, length(s)]
  elseif (tl <= maximum(s) <= th) & (tl <= minimum(s) <= th)
    return ["draw", "end"], [1, length(s)]
  end
  times = []
  flags = []
  if s[1] >= th
    flag = "win"
  elseif tl <= s[1] <= th
    flag = "draw"
  elseif s[1] < tl
    flag = "lose"
  end

  push!(times, 1)
  push!(flags, flag)

  s2 = s[2:end]
  for i in eachindex(s2)
    f1 = comp_01(s2[i], tl, th)
    if f1 != flag
      flag = f1
      push!(times, i+1)
      push!(flags, flag)
    end
  end
  push!(times, length(s))
  push!(flags, "end")
  return flags, times
end

function comp_01(x, tl, th)
  if x > th
    return "win"
  elseif tl <= x <= th
    return "draw"
  elseif x < tl
    return "lose"
  else
    return "weird"
  end
end

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


ntotal = 10000000.
# os.chdir('c://Users/cohenbp/Documents/Neuroscience/2018/BW_Sims')
contents = os.listdir(os.getcwd())
files = []
for i in contents:
    if i[-3:] == 'txt':
        files.append(i)

# sim2 = [i for i in files if i[3] == '2']
sim3 = [i for i in files if i[3] == '3']
sim5 = [i for i in files if i[3] == '5']
Ne = 400
cbinsize = 1000
fbinsize = 5000
netd_binsize = 25000
half = int(round(Ne/2))

def get_stats(te, re, ntotal, half, netd_binsize, fbinsize, cbinsize):
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
    alts = len(t2)
    MDT = np.mean(d[fw])
    MDB = np.mean(d[fl])
    MD = np.mean(d)
    STD = np.std(d)
    CVD = CV(d)
    tbf, rbf = ligase(bot, te, re, BN)
    ttf, rtf = ligase(top, te, re, TN)
    CVSTW = build_CV(te, re, TN, top)
    CVSBW = build_CV(te, re, BN, bot)
    FFST = pool_FF(TN, ttf, rtf, fbinsize)
    FFSB = pool_FF(BN, tbf, rbf, fbinsize)
    cwT = rand_pair_cor(cbinsize, ttf, rtf, TN, 1000)
    cwB = rand_pair_cor(cbinsize, tbf, rbf, BN, 1000)
    return [MD, STD, CVD, alts, cwT, cwB, np.mean(CVSTW), np.mean(CVSBW), np.mean(FFST), np.mean(FFSB)]


data3 = np.zeros((10, len(sim3)))
data = np.zeros((10, len(sim5)))
for i in range(len(sim5)):
    t, r = read_raster(sim5[i])
    data[:, i] = get_stats(t, r, ntotal, half, netd_binsize, fbinsize, cbinsize)

for i in range(len(sim3)):
    t, r = read_raster(sim3[i])
    data3[:, i] = get_stats(t, r, ntotal, half, netd_binsize, fbinsize, cbinsize)


s = np.array([float(i.split('_')[1]) for i in sim5])
s3 = np.array([float(i.split('_')[1]) for i in sim3])

fit1 = np.polyfit(s, data[0,:]/100000, deg = 1)
fit2 = np.polyfit(s, data[2,:], deg = 1)
fit3 = np.polyfit(s, data[1,:]/100000, deg = 1)
fit4 = np.polyfit(s, data[3,:], deg = 1)

fig, ax = plt.subplots(3)

ax[0].set_title("L4 Profile: Conductance Based Model")
ax[0].plot(s, fit1[0]*s + fit1[1], 'r')
ax[0].plot(s, data[0,:]/100000, "g.")
ax[0].set_xticks([])
ax[0].set_ylabel("Mean")
ax[1].plot(s, fit2[0]*s + fit2[1], 'r')
ax[1].plot(s, data[2,:], "g.")
ax[1].set_ylabel("CV")
ax[1].set_xticks([])
ax[2].plot(s, fit3[0]*s + fit3[1], 'r')
ax[2].plot(s, data[1,:]/100000, "g.")
ax[2].set_ylabel("STD")
ax[2].set_xlabel("Input")

fig, ax = plt.subplots(3)

ax[0].set_title("L4 Profile: Conductance Based Model")
ax[0].plot(s, fit4[0]*s + fit4[1], 'r')
ax[0].plot(s, data[3,:], "g.")
ax[0].set_xticks([])
ax[0].set_ylabel("Mean")
ax[1].plot(s, fit2[0]*s + fit2[1], 'r')
ax[1].plot(s, data[2,:], "g.")
ax[1].set_ylabel("CV")
ax[1].set_xticks([])
ax[2].plot(s, fit3[0]*s + fit3[1], 'r')
ax[2].plot(s, data[1,:]/100000, "g.")
ax[2].set_ylabel("STD")
ax[2].set_xlabel("Input")

# fit1 = np.polyfit(s, data[0,1:]/100000, deg = 1)
# fit2 = np.polyfit(s[1:], data[2,1:], deg = 1)
# fit3 = np.polyfit(s[1:], data[1,1:]/100000, deg = 1)
# fit4 = np.polyfit(s[1:], data[3,1:], deg = 1)
#
# fig, ax = plt.subplots(3)
#
# ax[0].set_title("L4 Profile: Conductance Based Model")
# ax[0].plot(s[1:], fit1[0]*s[1:] + fit1[1], 'r')
# ax[0].plot(s[1:], data[0, 1:]/100000, "g.")
# ax[0].set_xticks([])
# ax[0].set_ylabel("Mean")
# ax[1].plot(s[1:], fit2[0]*s[1:] + fit2[1], 'r')
# ax[1].plot(s[1:], data[2, 1:], "g.")
# ax[1].set_ylabel("CV")
# ax[1].set_xticks([])
# ax[2].plot(s[1:], fit3[0]*s[1:] + fit3[1], 'r')
# ax[2].plot(s[1:], data[1, 1:]/100000, "g.")
# ax[2].set_ylabel("STD")
# ax[2].set_xlabel("Input")
#
# fig, ax = plt.subplots(3)
#
# ax[0].set_title("L4 Profile: Conductance Based Model")
# ax[0].plot(s[1:], fit4[0]*s[1:] + fit4[1], 'r')
# ax[0].plot(s[1:], data[3, 1:], "g.")
# ax[0].set_xticks([])
# ax[0].set_ylabel("Mean")
# ax[1].plot(s[1:], fit2[0]*s[1:] + fit2[1], 'r')
# ax[1].plot(s[1:], data[2, 1:], "g.")
# ax[1].set_ylabel("CV")
# ax[1].set_xticks([])
# ax[2].plot(s[1:], fit3[0]*s[1:] + fit3[1], 'r')
# ax[2].plot(s[1:], data[1, 1:]/100000, "g.")
# ax[2].set_ylabel("STD")
# ax[2].set_xlabel("Input")


fig, ax = plt.subplots(2,2)
ax[0,0].hist(data[4])
ax[0,1].hist(data[5])
ax[1,0].hist(data[6])
ax[1,1].hist(data[7])

fig, ax = plt.subplots(2,2)
ax[0,0].hist(data3[4])
ax[0,1].hist(data3[5])
ax[1,0].hist(data3[6])
ax[1,1].hist(data3[7])
