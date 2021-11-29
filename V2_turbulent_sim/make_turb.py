# -*- coding: utf-8 -*-
"""Create the htc files and a .bat script to run simulations.

Each htc file that is made is/has:
    - a different wind speed
    - initial rotor speed that is determined based on wind speed and operation.dat file
    - 700 seconds total simulated, but first 100 seconds discarded
    - mann turbulence for given wind class (A, B or C)
    - power-law shear per IEC 61400-1
    - with tower shadow

Master htc file:
    - turbulent wind
    - placed in folder "htc_master/"

HAWC folder structure:
    - control/
    - data/
    - htc_master/
        - <master file name>.htc
    - (script creates) htc_turb/
        - (htc files to run)
    - (script creates) run_turb.bat

STEPS IN RUNNING THIS SCRIPT:
    1. Place your master htc file in a folder called "htc_master/". This folder should
       be in your hawc model folder, same level as control/ and data/.
    2. Copy this .py file and _run_hawc2_utils.py into the hawc folder, at the same
       level of control/ and data/.
    3. Update the input parameters in the script.
    4. Run this script: `python make_turb.py`. You should now have the htc files you
       want to run in htc_turb/. The .bat file will be in the same location as this
       script.
    5. Open one of the created htc files and skim it to make sure everything looks ok.
    6. Open a command prompt in the hawc folder. Run the .bat file by entering its name
       into the Command Prompt, then hitting Enter.
    7. When the simulations are done, check a few log files to make sure things ran okay.
    8. If you ran on multiple machines, transfer the results files to a single folder on
       a single computer with NumPy.
    9. Post-process the statistics using the post-processing script.
"""
import random, os
from _run_hawc2_utils import clean_directory, get_rotation_speed


operation_dat = './data/V2_hs2.opt'  # path to operation.dat file
master_name = 'V2_turb.htc'  # name of file in the htc_master folder
wsps = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]  # wind speeds you want to simulate
nseeds = 6  # number of random simulations per wind speed
iref = 0.14  # reference turbulence intensity for turbulence class, given in IEC 61400-1 Table 1
hawc2_exe = 'C:/Users/rink/Desktop/hawc_simulations/HAWC2_12.8_1900/HAWC2MB.exe'  # path to HAWC2 executable

# =======================================================================================
# you shouldn't need to change anything below this line :)

seed = 1337  # Updating is optional. Set a random seed, which fixes the order of the random number generator.
clean_htc = False  # clean the htc directory? !!! WILL DELETE ALL FILES IN HTC DIR. IF TRUE !!!
htc_dir = './htc_turb/'  # name of output directory !!! END WITH SLASH !!!
master_dir = './htc_master/'  # name of path with htc master file
res_dir = './res_turb/'  # folder to save output !!! END WITH SLASH !!!
time_start = 100  # time to start recording output
time_stop = 700  # time to stop recording output
bat_name = './run_turb.bat'  # name of bat file
nx = 8192  # number of points in the longitudinal direction

htc_master = master_dir + master_name  # path to htc master file
master_noext = os.path.splitext(master_name)[0]  # name of master file w/o extension
random.seed(seed)  # initialize the random number generator

# clear or create the htc directory
clean_directory(htc_dir, clean_htc)

# load the master file's contents into memory
with open(htc_master, 'r') as htc:
    contents = htc.readlines()
    for line in contents:
        if line.lstrip().startswith('wind_ramp_abs'):
            print('WARNING!!! Your master htc has a wind_ramp_abs command. Are you '
                  + 'sure it is a turbulent simulation?')

htc_files = []
# loop over the wind speeds
for iw, wsp in enumerate(wsps):

    # define turbulence intensity per IEC 61400-1 Eq. 10
    tint = iref*(0.75*wsp + 5.6) / wsp

    # get the longitudinal box spacing
    dx = (time_stop - time_start) * float(wsp) / float(nx)

    # get initial rotation speed
    Omega0 = get_rotation_speed(wsp, operation_dat)

    # loop over the seed in that wind speed
    for js in range(nseeds):

        # randomly sample the seed for the turbulence generator
        seed = random.randrange(int(2**16))

        # define path names
        filename = ('%s_' % master_noext) + ('%.1f' % wsp).zfill(4) + ('_%i' % seed)  # name of htc file w/o extension
        htc_path = htc_dir + filename + '.htc'

        # loop over the contents and update them accordingly
        output = 0  # tracker to check if we've hit output block yet
        for i, line in enumerate(contents):

            # if we've passed the dll block, toggle tracker to prevent new_htc_structure
            #   and dll filenames from being overwritten.
            if line.lstrip().startswith('end dll'):
                output = 1

            # simulation time
            if line.lstrip().startswith('time_stop'):
                contents[i] = ('time_stop   %.1f ; \n' % time_stop)
            # log file
            elif line.lstrip().startswith('logfile'):
                contents[i] = ('logfile ./log/%s.log ; \n' % filename)
            # initial rotation speed
            elif line.lstrip().startswith('mbdy2_ini_rotvec_d1'):
                contents[i] = ('mbdy2_ini_rotvec_d1 0.0 0.0 -1.0 %.2f ; \n' % Omega0)
            # wind speed
            elif line.lstrip().startswith('wsp'):
                contents[i] = ('wsp                     %.1f   ; \n' % wsp)
            # turbulence intensity
            elif line.lstrip().startswith('tint'):
                contents[i] = ('tint                    %.3f   ; \n' % tint)
            # shear format
            elif line.lstrip().startswith('shear_format'):
                contents[i] = ('shear_format            3 0.2 ;  \n')
            # turbulence format
            elif line.lstrip().startswith('turb_format'):
                contents[i] = ('turb_format             1     ;\n')
            # tower shadow
            elif line.lstrip().startswith('tower_shadow_method'):
                contents[i] = ('tower_shadow_method     3     ;\n')
            # random seed in mann parameters
            elif line.lstrip().startswith('create_turb_parameters'):
                contents[i] = ('create_turb_parameters	29.4 1.0 3.9 %i 0; \n' % seed)
            # turb filename u
            elif line.lstrip().startswith('filename_u'):
                contents[i] = ('filename_u	./turb/%s_turb_u.bin; \n' % filename)
            # turb filename v
            elif line.lstrip().startswith('filename_v'):
                contents[i] = ('filename_v	./turb/%s_turb_v.bin; \n' % filename)
            # turb filename w
            elif line.lstrip().startswith('filename_w'):
                contents[i] = ('filename_w	./turb/%s_turb_w.bin; \n' % filename)
            # lateral box dimensions
            elif line.lstrip().startswith('box_dim_u'):
                contents[i] = ('box_dim_u	8192 %.6f; \n' % dx)
            # output filename
            elif line.lstrip().startswith('filename') and output:
                contents[i] = ('filename %s%s ; \n' % (res_dir, filename))
            # output time
            elif line.lstrip().startswith('time') and output:
                contents[i] = ('time %.1f %.1f ; \n' % (time_start, time_stop))

        # write the new htc file
        with open(htc_path, 'w') as htc:
            htc.writelines(contents)

        # append it's name to the list to be added to the bat file
        htc_files.append(htc_path)

# write the .bat file
with open(bat_name, 'w') as bat:
    for htc_path in htc_files:
        bat.write('"%s" "%s" \n' % (hawc2_exe, htc_path))

