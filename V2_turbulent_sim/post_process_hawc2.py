# -*- coding: utf-8 -*-
"""Calculate statistics and short-term fatigue loads from HAWC2 output files.

Requirements to run file:
    - All HAWC2 outputs must be in the same folder.
    - All HAWC2 output files in the folder must have the same channels and channel order.
    - All HAWC2 output files must be ascii.
    - Python installation must have NumPy.
    - Python version is >=3.6.

How to run this file:
    1. Copy this script and _run_hawc2_utils to the same folder.
    1. Plase all HAWC2 ascii outputs to be processed in a single folder.
    2. Update the script inputs.
    3. Run this script: `python post_process_hawc2.py`.
    4. There should now be a collection of files in the results folder with different
       statistics.
"""
import os
import numpy as np
from _proc_hawc2_utils import initialize_stat, calculate_stat, update_stat


res_folder = './res_turb/'  # folder with the HAWC2 results to post-process !!! MUST END WITH SLASH !!!
sel_idxs = [4, 10, 11, 13, 15, 19, 20, 22, 23, 27, 28, 29, 63, 66, 72,
            102, 110]  # channel indices in .sel file that you want to process

# =======================================================================================
# you shouldn't need to change anything below this line :)

stats = ['mean', 'max', 'min', 'std', 'del4', 'del10', 'p99', 'p01']

# convert channel numbers in sel file to python indices
chan_idxs = [i-1 for i in sel_idxs]

# get the list of dat files
dat_files = [f for f in os.listdir(res_folder) if f.endswith('.dat')]

# initialize the statistics files
stat_paths = {}  # dictionary of paths to stat files
for stat in stats:
    stat_paths[stat] = res_folder + f'stats_{stat}.txt'
    initialize_stat(stat, stat_paths, sel_idxs, len(dat_files))

# loop through dat files
for i, dat_file in enumerate(dat_files):
    print(f'Processing file {i+1}/{len(dat_files)}...')

    # if the dat file is empty, make data a 2D array of nans
    if not os.stat(res_folder + dat_file).st_size:
        data = np.full((1, len(chan_idxs)), np.nan)

    # otherwise, load the dat file and isolate the channels we want
    else:
        data = np.loadtxt(res_folder + dat_file)[:, chan_idxs]

    # loop through the stats to calculate
    for stat in stats:

        # calculate the statistic
        val = calculate_stat(data, stat)

        # update the text file
        update_stat(dat_file, stat_paths[stat], val)

print(f'Statistics files saved to {res_folder}')