# -*- coding: utf-8 -*-
"""Utilities for run_hawc2 scripts
"""
import os


pi = 3.141592653589793  # define this so we don't need numpy for htc generation


def clean_directory(htc_dir, clean_htc):
    """Clean or create a directory as requested"""
    # if the folder exists but we want a clean run
    if os.path.isdir(htc_dir) and clean_htc:
        os.rmdir(htc_dir)  # delete the folder
        os.mkdir(htc_dir)  # make an empty folder
    # if the folder doesn't exists
    elif not os.path.isdir(htc_dir):
        os.mkdir(htc_dir)  # make the folder
    return


def get_rotation_speed(wsp, operation_dat):
    """Determine the initial rotation speed from an operation.dat file"""
    with open(operation_dat, 'r') as oper:
        contents = oper.readlines()
    contents = contents[1:]  # ignore first line
    Omega0 = 9.6 * pi/30  # default rotation speed [rad/s]
    for line in contents:
        wsp_line = float(line.split()[0])
        if wsp_line > wsp:
            Omega0 = float(line.split()[2]) * pi/30
            break
    return Omega0
