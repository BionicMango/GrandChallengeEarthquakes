# Imports
import numpy as np # for extensive numerical calculations
import pandas as pd # for csv handling as DataFrames and Series
import matplotlib.pyplot as plt # for plotting
import h5py # for reading hdf5 files which are highly optimised data storages

# Store File Paths so that they can be altered easily if necessary
csvFile =  r'C:\Users\teert\Desktop\Grand Challenge\chunk2.csv'
hdf5File = r'C:\Users\teert\Desktop\Grand Challenge\chunk2.hdf5'

sampleF = 100

df = pd.read_csv(csvFile)
dft = h5py.File(hdf5File, 'r') # r = read only

print('First five rows of \n', df.head()) # print first five rows to ensure it has loaded properly

print(list(dft.keys())) # hdf5 files can group datasets together under different 'keys' (like dictionaries) - only one key means only one group 'data'
print(list(dft['data'].keys())[0:10]) # checking how many keys are in the group dft['data'] - 'Data' is only one group, with 200,000 other groups of data.

dset = dft['data']['109C.TA_20060723155859_EV'] # this is just the first DataSet in the Group dft['data']
print(dset.shape, dset.dtype) # to get an idea of what the DataSet looks like (dim: 6000 x 3, type: float32)
print(dset.attrs.keys()) # print keys for this dataset's metadat (has useful information e.g. p_status); similar to a dictionary but not quite

# Setting manual onset & coda times to those specified in the metadata (note: coda is written as 2d array with single entry)
onsetManual = (dset.attrs['p_arrival_sample']/100, dset.attrs['s_arrival_sample']/100, dset.attrs['coda_end_sample'][0][0]/100) # originally in hundredths of a second (e.g. 700.0 = 7.000 seconds)
print(onsetManual) # printing to verify we have the right data

# Convert dset to numpy array & print first 10 rows
data = np.array(dset)
print(data[0:10, :])

# There are three columns; we isolate the first column, which represents the amplitude of these seismic waves over time
amplitude = data[:, 0]

# Function: Find P and S waves by using thresholds
def aboveThreshold(data, thresh): # returns index of first time it exceeds threshold
    for i in data:
        if i > thresh:
            return np.where(data == i)
        else:
            return None

# Function: Windowing / Sliding Average function
def slidingAverage(data, windowSize):
    slidingAves = np.zeros(data.shape)
    end = (-int(windowSize - 1)//2,int(windowSize - 1)//2) # endpoints for each average - half a window size around each point

    for col in range(data.shape[1]):
        for row in range(data.shape[0]): # depending on if the index is near the end or not.
            if row < windowSize/2:
                slidingAves[row, col] = np.mean(np.abs(data[:row+end[1], col])) # mean of values between 0 and a + 1/2 window size
            elif row > data.shape[0] - windowSize/2:
                slidingAves[row, col] = np.mean(np.abs(data[row+end[0]:, col])) # mean of values between a - 1/2 window size and last entry
            else:
                slidingAves[row, col] = np.mean(np.abs(data[row+end[0]:row+end[1], col])) # mean of values between 1/2 window size either side

    return slidingAves

def plotData(time, data, onset: tuple = (None, None, None), margins=(0.5, 15000)): # onset = (p wave onset time, s wave onset time, coda (end)), 
    fig, ax = plt.subplots(data.shape[1], 1, figsize=(15, 15))
    i = 0 # looping variable for each axis
    for waveform in data.T:
        ax[i].plot(time, waveform, color='black', label='Seismic Amplitude')
        ax[i].grid(True)

        # Set x/y limits & labels
        ax[i].set_xlim(min(time) - margins[0], max(time) + margins[0])
        ax[i].set_xlabel('Time (s)')
        ax[i].set_ylim(min(waveform) - margins[1], max(waveform) + margins[1])
        ax[i].set_ylabel('Amplitude')

        # If onset times are specified
        if onset[0] != None:
            # P Wave Onset Time (Line)
            ax[i].vlines(onset[0], min(waveform), max(waveform), color='orange', ls='--')
            ax[i].scatter(onset[0]+0.07, min(waveform), color='orange', marker='^')

            # P Wave Onset Time (Label) to 1 d.p.
            ax[i].annotate(
                f'P Wave Onset at {onset[0]:.1f}',
                xy = (onset[0] + 0.5, min(waveform) - 2/3 * margins[1]),
                xytext = (onset[0]  + 0.5, min(waveform) - 2/3 * margins[1]),
                fontweight = 'bold',
                color = 'orange'
            )

        if onset[1] != None:
            # S Wave Onset Time (Line)
            ax[i].vlines(onset[1], min(waveform), max(waveform), color='red', ls='--')
            ax[i].scatter(onset[1]+0.07, min(waveform), color='red', marker='^')

            # S Wave Onset Time (Label) to 1 d.p.
            ax[i].annotate(
                f'S Wave Onset at {onset[1]:.1f}',
                xy = (onset[1] + 0.5, min(waveform) - 2/3 * margins[1]),
                xytext = (onset[1] + 0.5, min(waveform) - 2/3 * margins[1]),
                fontweight = 'bold',
                color = 'red'
            )

        if onset[2] != None:
            # Coda End Sample (Line)
            ax[i].vlines(onset[2], min(waveform), max(waveform), color='deepskyblue', ls='--')
            ax[i].scatter(onset[2]+0.05, min(waveform), color='deepskyblue', marker='^')

            # Coda End Sample
            ax[i].annotate(
                f'Coda at {onset[1]:.1f}',
                xy = (onset[2] + 0.5, min(waveform) - 2/3 * margins[1]),
                xytext = (onset[2] + 0.5, min(waveform) - 2/3* margins[1]),
                fontweight = 'bold',
                color = 'deepskyblue'
            )

            # Adding onset detection signs
            ax[i].scatter(onset, (min(waveform), min(waveform), min(waveform)), color=('orange', 'red', 'cyan'), marker='^')

        # Loop increment
        i += 1

    fig.legend()

    plt.show()

# Plotting the three waveforms associated with the chosen earthquake
time = np.arange(0, 60, 1/sampleF)
plotData(time, data, onset=onsetManual)

dataSmoothed = slidingAverage(data, 0.05*sampleF)
plotData(time, dataSmoothed)