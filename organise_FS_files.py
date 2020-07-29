#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 16:12:34 2020

@author: k1511004
"""

import glob as glob
import os
from os import system

# set directories
FS_source_data_dir = '/data/project/PSYSCAN/Sarah/data_prep/FEP/baseline/FS/FEPS_FS6_Feb20/FEPS_FS6_Feb20/'
annot_dir = '/home/k1511004/Data/PSYSCAN/WP5_data/Prelim_FEP_dataset/253_subjects_Apr_20/Data/sMRI/annot_forJonathan/'
FS_out_dir = '/home/k1511004/Data/PSYSCAN/WP5_data/Prelim_FEP_dataset/253_subjects_Apr_20/Data/sMRI/FS_data/'

# get list of subjects
subjects = os.listdir(FS_source_data_dir)


# loop through subject directories
for subject in subjects :
    
    print (subject)
    
    # create subject dir with /surf and /label directories
    subject_dir = FS_out_dir + subject + '/'
#    cmd = 'mkdir ' + subject_dir
#    system(cmd)
#    cmd = 'mkdir ' + subject_dir + 'surf/'
#    system(cmd)
#    cmd = 'mkdir ' + subject_dir + 'label/'
#    system(cmd)
    
    # move files into appropriate directories and rename them
    cmd = 'cp ' + annot_dir + '/' + subject + '_lh.500.aparc.annot ' + subject_dir + 'label/lh.500.aparc.annot'
    system(cmd)
    cmd = 'cp ' + annot_dir + '/' + subject + '_rh.500.aparc.annot ' + subject_dir + 'label/rh.500.aparc.annot'
    system(cmd)
#    cmd = 'cp ' + FS_source_data_dir + subject + '/surf/lh.thickness ' + subject_dir + 'surf/lh.thickness'
#    system(cmd)
#    cmd = 'cp ' + FS_source_data_dir + subject + '/surf/rh.thickness ' + subject_dir + 'surf/rh.thickness'
#    system(cmd)