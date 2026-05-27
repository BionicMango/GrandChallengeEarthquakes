# Imports
import numpy as np # for extensive numerical calculations
import pandas as pd # for csv handling as DataFrames and Series
import matplotlib.pyplot as plt # for plotting
import h5py # for reading hdf5 files which are highly optimised data storages

# Store File Paths so that they can be altered easily if necessary
csvFile =  r'C:\Users\teert\Desktop\Grand Challenge\chunk2.csv'
hdf5File = r'C:\Users\teert\Desktop\Grand Challenge\chunk2.hdf5'

df = pd.read_csv(csvFile)
dft = h5py.File(hdf5File, 'r') # r = read only

print(df.head()) # print first five rows to ensure it has loaded properly

print(list(dft.keys())) # hdf5 files can group datasets together under different 'keys' (like dictionaries) - only one key means only one group 'data'
print(list(dft['data'].keys())[0:10]) # checking how many keys are in the group dft['data'] - 'Data' is only one group, with 200,000 other groups of data.

dset = dft['data']['109C.TA_20060723155859_EV'] # this is just the first DataSet in the Group dft['data']
print(dset.shape, dset.dtype) # to get an idea of what the DataSet looks like (dim: 6000 x 3, type: float64)

# Convert dset to numpy array & print first 10 rows
data = np.array(dset)
print(data[0:10, :])

# There are three columns; we isolate the first column, which represents the amplitude of these seismic waves over time
amplitude = data[:, 0]

plt.plot(amplitude)
plt.show()plt.show()

# Function: Find P and S waves by using thresholds
def aboveThreshold(data, thresh): # returns index of first time it exceeds threshold
    for i in data:
        if i > thresh:
            return np.where(data == i)
        else:
            return None

# Function: Windowing / Sliding Average function
def slidingAverage(time, data, windowSize):
    windowAves = []
    end = (-int(windowSize - 1)//2,int(windowSize - 1)//2) # endpoints for each average - half a window size around each point

    for i in range(data.shape): # depending on if the index is near the end or not.
        if i < windowSize/2:
            windowAves.append(np.mean(np.abs(data[:i+end[1]]))) # append the mean between 0 and a + 1/2 window size
        elif i > data.shape[0] - windowSize/2:
            windowAves.append(np.mean(np.abs(data[i+end[0]:]))) # append the mean between a - 1/2 window size and last entry
        else:
            windowAves.append(np.mean(np.abs(data[i+end[0]:i+end[1]]))) # append the mean between 1/2 window size either side
    return np.array(windowAves)
