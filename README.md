# Grand Challenge - Earthquakes
ENGG1810 (2026 Semester 1) Grand Challenge Project - Data &amp; Analysis

The data cleanup and analysis was conducted in the 'EartquakeAnalysis.py' python file. The process included:
- Extracting Waveform Data and Metadata (metadata provided information on the manual p wave onset, s wave onset, and coda times)
- Smoothing the desired waveforms using a sliding average (using a windowing function)
- Using thresholds (as a ratio of the maximum amplitude), finding the p wave onset, s wave onset, and coda times
- Comparing the manual and algorithm's times (i.e. finding their difference)

This algorithm was applied to 200 waveforms, and the appropriate thresholds chosen by trial and error. The differences between the algorithm's and manual times were outputted as a csv file, which can be accessed in 'output.csv'.

## Output from the first dataset
Listed below are visualisations of one dataset (with three waveforms corresponding to the East/West, North/South, and Vertical directions respectively). The original data is sourced from stations around the world.

Data for this project was sourced from STanford EArthquarke Dataset (STEAD):
Mousavi, S. M., Sheng, Y., Zhu, W., Beroza G.C., (2019).  STanford EArthquake Dataset (STEAD): A Global Data Set of Seismic Signals for AI,  IEEE Access, doi:10.1109/ACCESS.2019.2947848

(File: 'chunk2.hdf5', dataset name = '109C.TA_20060723155859_EV')
```windowSize = 80```
```threshRatio = {'P': 0.018, 'S': 0.15, 'C': 0.090}```

Plot of waveforms w/ manual onset/coda times:

<img width="1500" height="1052" alt="dset1 Manual Onset and coda data" src="https://github.com/user-attachments/assets/a6f9baca-45a7-4287-bd14-ca8b9666286e" />

Plot of smoothed waveforms w/ algorithm's onset/coda times:

<img width="1500" height="1052" alt="dset1 Algorithm Onset and coda SMOOTHED data" src="https://github.com/user-attachments/assets/8840a9a7-06b1-4180-8bae-f1e5695cc525" />

Plot of original waveforms w/ algorithm's onset/coda times:

<img width="1500" height="1052" alt="dset1 Algorithm Onset and coda data" src="https://github.com/user-attachments/assets/1c5cdc07-dabd-428b-bbde-965b4e43b769" />
