# Imports
import numpy as np # for extensive numerical calculations
import pandas as pd # for csv handling as DataFrames and Series
import matplotlib.pyplot as plt # for plotting
import h5py # for reading hdf5 files which are highly optimised data storages

### CONSTANTS ###
sampleF = 100 # in Hz, i.e. 100 samples every second (every 0.01 time step)
time = np.arange(0, 60, 1/sampleF)
windowSize = 80
threshold = {
    'P': 200,
    'S': 4750,
    'C': 5500
}

# Function: Find P and S waves by using thresholds
def aboveThreshold(waveform, thresh): # returns index of first time it exceeds threshold
    aboveThresh = []

    i = 0 # looping variable (easier to use while loop than for loop in this case)
    while i < waveform.shape[0]:
        if waveform[i] > thresh:
            aboveThresh.append(i) # returns the index where this is true
        i += 1

    return int(aboveThresh[0]), int(aboveThresh[-1])

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

# Plotting Function (to plot the three waveforms + onset/coda nicely)
def plotData(time, data, onset: tuple = (None, None, None), margins=(0.5, 12000)): # onset = (p wave onset time, s wave onset time, coda (end)), 
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
            ax[i].vlines(onset[0], min(waveform) - margins[1], max(waveform) + margins[1], color='orange', ls='--')

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
            ax[i].vlines(onset[1], min(waveform) - margins[1], max(waveform) + margins[1], color='red', ls='--')

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
            ax[i].vlines(onset[2], min(waveform) - margins[1], max(waveform) + margins[1], color='deepskyblue', ls='--')

            # Coda End Sample
            ax[i].annotate(
                f'Coda at {onset[2]:.1f}',
                xy = (onset[2] + 0.5, max(waveform)),
                xytext = (onset[2] + 0.5, max(waveform)),
                fontweight = 'bold',
                color = 'deepskyblue'
            )

        # Loop increment
        i += 1

    plt.show()

# Combined earthquake detection function
def earthquakeDetection(dset, margins=(0.5, 12000)):
    data = np.array(dset)

    # Plot manual onset/offset times data
    onsetManual = ( # originally in hundredths of a second, e.g.n 700.0 = 7 seconds
        dset.attrs['p_arrival_sample']/100,
        dset.attrs['s_arrival_sample']/100,
        dset.attrs['coda_end_sample'][0][0]/100
    )
    plotData(time, data, onset=onsetManual, margins=margins)

    # Algorithm detection
    dataSmoothed = slidingAverage(data, windowSize)
    onsetAlg = (
        (1/sampleF) * min(aboveThreshold(dataSmoothed[:, 0], threshold['P'])[0], aboveThreshold(dataSmoothed[:, 1], threshold['P'])[0]),
        (1/sampleF) * min(aboveThreshold(dataSmoothed[:, 0], threshold['S'])[0], aboveThreshold(dataSmoothed[:, 1], threshold['S'])[0]), # takes the first onset time of any direction
        (1/sampleF) * max([aboveThreshold(dataSmoothed[:, 0], threshold['C'])[1], aboveThreshold(dataSmoothed[:, 1], threshold['C'])[1]]) # last coda time
    )
    plotData(time, data, onset=onsetAlg, margins=margins)
    plotData(time, dataSmoothed, onset=onsetAlg, margins=margins)

## TESTTING EARTHQUAKE DETECTION ALGORITHM
hdf5File = r'C:\Users\teert\Desktop\Grand Challenge\chunk2.hdf5'

dft = h5py.File(hdf5File, 'r') # r = read only
print(list(dft.keys())) # hdf5 files can group datasets together under different 'keys' (like dictionaries) - only one key means only one group 'data'
print(list(dft['data'].keys())[0:10]) # checking how many keys are in the group dft['data'] - 'Data' is only one group, with 200,000 other groups of data.

dset = dft['data']['109C.TA_20070109140205_EV'] # random waveform
print(dset.shape, dset.dtype) # to get an idea of what the DataSet looks like (dim: 6000 x 3, type: float32)
print(dset.attrs.keys()) # print keys for this dataset's metadat (has useful information e.g. p_status); similar to a dictionary but not quite

earthquakeDetection(dset)