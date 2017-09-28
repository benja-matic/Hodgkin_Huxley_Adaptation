###generate simulations
import numpy as np

def SD_2_K(x):
    return 1./(2.*np.pi*(x**2))

N = 10000
data = np.zeros((13, N))

data[0,:] = np.random.uniform(0, 1, N)
data[1,:] = np.random.uniform(0, 1, N)
data[2,:] = np.random.uniform(0, 1, N)
data[3,:] = np.random.uniform(0, 1, N)

data[4,:] = np.random.uniform(0, 2, N)
data[5,:] = np.random.uniform(0, 4, N)
data[6,:] = np.random.uniform(0, 4, N)
data[7,:] = np.random.uniform(0, 4, N)

data[8,:] = SD_2_K(np.random.uniform(0.1, 0.5, N))
data[9,:] = SD_2_K(np.random.uniform(0.1, 1.0, N))
data[10,:] = SD_2_K(np.random.uniform(0.1, 1.0, N))
data[11,:] = SD_2_K(np.random.uniform(0.1, 1.0, N))

data[12,:] = np.random.uniform(0, 2., N)


newfile = open("BW_scan_2.py")
for i in range(len(N)):
    text = "{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13}\n".format(data[0,i], data[1,i], data[2,i], data[3,i], data[4,i], data[5,i], data[6,i], data[7,i], data[8,i], data[9,i], data[10,i], data[11,i], data[12,i], i)
    newfile.write(text)


newfile.close()
