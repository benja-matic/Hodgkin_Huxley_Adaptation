import numpy as np
import matplotlib.pyplot as plt
plt.ion()


def read_raster(x):
    with open(x, 'r') as f:
        a = f.read()
    b = a.spit('\n')
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
