#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 12:48:17 2020

@author: k1511004
"""

# import what we need
import sys
sys.path.append("/home/k1511004/Projects/neuroCombat/neuroCombat/") 
sys.path.append("/home/k1511004/Projects/anatomical_covariance/") 
sys.path.append("/home/k1511004/Projects/riem_mglm/")
sys.path.append("/home/k1511004/Projects/BrainNetCNN_keras/")

import tensorflow as tf
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
from brainNetCNN_keras import EdgeToEdge, EdgeToNode, NodeToGraph
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, Dropout, BatchNormalization
from tensorflow.keras.activations import relu, sigmoid
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, RMSprop, Adadelta, Adam

from scipy.linalg import logm
import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit, StratifiedKFold
from sklearn import svm
#from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, classification_report, r2_score, mean_squared_error, recall_score, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from neuroCombat import neuroCombat
from neuroCombatCV3 import fit_transform_neuroCombat, apply_neuroCombat_model, ShuffleSplitFixed
from sklearn.preprocessing import OneHotEncoder
from riem_mglm import mglm_spd, logmap_spd
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error, roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import LeaveOneOut
from scipy.stats import pearsonr
from datetime import datetime
from anatomical_covariance import cortical_regional_means, anatomical_covariance_matrix
from os import listdir
#import pyGPs_mod
#from pyGPs_mod import mean, cov, GPR

# function to calculate response to treatment at each timepoint
def response_to_treatment(x) :
    
    total_panss = x[0]
    baseline_panss = x[1]
    
    if np.isnan(baseline_panss) :
        
        return 'N/A'
    
    elif total_panss / baseline_panss < 0.8 :
        
        return 'response'
    
    else :
        
        return 'non-response'

# specialized CV giving the closest thing to LOOCV that is compatible with supervised comBat.
# split each group into test inds of size two
def min_two_test_CV(group_data) :
    
    # initialise list of test inds and train inds
    test_indices = []
    train_indices = []
    
    # list of all inds
    n_subjects = len(group_data)
    all_inds = np.arange(n_subjects)
    
    # find unique groups and loop though them
    unique_groups = list(set(group_data))
    for group in unique_groups :
        
        # get group indices
        group_inds = np.squeeze(np.where(group_data == group))
        n_subjects = len(group_data)
        
        
        # get group size and n_chunks
        # round down for n_chunks so no chunk is below size 2
        group_size = len(group_inds)
        n_chunks = np.floor_divide(group_size, 2)
        
        # divide into chunks
        group_test_inds = np.array_split(group_inds, n_chunks)
        
        # add to list of test indices
        test_indices = test_indices + group_test_inds
        
    
    # list of all inds
    n_subjects = len(group_data)
        
    # generate complementary group train indices
    train_indices = [np.setdiff1d(np.arange(n_subjects), test_index) for test_index in test_indices]
    
    return train_indices, test_indices

# lightly regularized
def brainnetCNN_model_1(input_shape, n_filters, use_bias):
    
    # define network architecture
    # 2 E2E
    # followed by 1 E2N
    # followed by 1 N2G
    # followed by dense
    # then sigmoid for classification
    # sprinkle with batch normalization and dropout
    X_input = Input(shape=input_shape, name='input_layer')
    E2E_1 = EdgeToEdge(n_filters[0], use_bias[0])(X_input)
    E2E_1 = Activation('relu')(E2E_1)
    E2E_2 = EdgeToEdge(n_filters[1], use_bias[1])(E2E_1)
    E2E_2 = Activation('relu')(E2E_2)
    E2N = EdgeToNode(n_filters[2], use_bias[2])(E2E_2)
    E2N = Activation('relu')(E2N)
    N2G = NodeToGraph(n_filters[3], use_bias[3])(E2N)
    N2G = Activation('relu')(N2G)
    N2G = Flatten()(N2G)
    dense_1 = Dense(n_filters[4], activation = 'relu', use_bias = use_bias[4])(N2G)
    dense_1 = BatchNormalization()(dense_1)
    dense_final = Dense(1, activation='sigmoid', use_bias = use_bias[5])(dense_1)
    model = Model(inputs = X_input, outputs= dense_final)
    return model

# heavily regularized
def brainnetCNN_model_2(input_shape, n_filters, use_bias):
    act = 'relu'
    
    # define network architecture
    # 2 E2E
    # followed by 1 E2N
    # followed by 1 N2G
    # followed by dense
    # then sigmoid for classification
    # sprinkle with batch normalization and dropout
    X_input = Input(shape=input_shape, name='input_layer')
    E2E_1 = EdgeToEdge(n_filters[0], use_bias[0])(X_input)
    E2E_1 = BatchNormalization()(E2E_1)
    E2E_1 = Activation(act)(E2E_1)
    E2E_2 = EdgeToEdge(n_filters[1], use_bias[1])(E2E_1)
    E2E_2 = BatchNormalization()(E2E_2)
    E2E_2 = Activation(act)(E2E_2)
    E2N = EdgeToNode(n_filters[2], use_bias[2])(E2E_2)
    E2N = Activation(act)(E2N)
    N2G = NodeToGraph(n_filters[3], use_bias[3])(E2N)
    N2G = Activation(act)(N2G)
    N2G = Flatten()(N2G)
    N2G = Dropout(0.2)(N2G)
    dense_1 = Dense(n_filters[4], activation = act, use_bias = use_bias[4])(N2G)
    dense_1 = Dropout(0.2)(dense_1)
    dense_final = Dense(1, activation='linear', use_bias = use_bias[5])(dense_1)
    model = Model(inputs = X_input, outputs= dense_final)
    return model

# test
def brainnetCNN_model_3(input_shape, n_filters, use_bias):
    
    # define network architecture
    # 2 E2E
    # followed by 1 E2N
    # followed by 1 N2G
    # followed by dense
    # then sigmoid for classification
    # sprinkle with batch normalization and dropout
    X_input = Input(shape=input_shape, name='input_layer')
    E2E_1 = EdgeToEdge(n_filters[0], use_bias[0])(X_input)
    E2E_1 = BatchNormalization()(E2E_1)
    E2E_1 = Activation('relu')(E2E_1)
    dense_final = Dense(1, activation='linear', use_bias = use_bias[5])(E2E_1)
    model = Model(inputs = X_input, outputs= dense_final)
    return model

# test
def brainnetCNN_model_4(input_shape, n_filters, use_bias):
    
    # define network architecture
    X_input = Input(shape=input_shape, name='input_layer')
    dense_1 = Dense(8, activation='linear', use_bias = use_bias[5])(X_input)
    dense_final = Dense(1, activation='linear', use_bias = use_bias[5])(dense_1)
    model = Model(inputs = X_input, outputs= dense_final)
    return model

# set model parameters
# biases - off for brainnet CNN layers, on for dense
use_bias = [False, False, False, False, True, True]

# set directories 
#sMRI_data_dir = '/home/k1511004/Data/PSYSCAN/WP5_data/Prelim_FEP_dataset/174_subjects_Oct_19/Data/sMRI/'
sMRI_data_dir = '/home/k1511004/Data/PSYSCAN/WP5_data/Prelim_FEP_dataset/253_subjects_Apr_20/Data/sMRI/FS_data/'
fMRI_data_dir = '/home/k1511004/Data/PSYSCAN/WP5_data/Prelim_FEP_dataset/253_subjects_Apr_20/Data/fMRI/'
FS_data_dir = sMRI_data_dir
metadata_dir = '/home/k1511004/Data/PSYSCAN/WP5_data/Prelim_FEP_dataset/253_subjects_Apr_20/metadata/'
outcomes_dir = '/home/k1511004/Data/PSYSCAN/WP5_data/FEP/metadata/'
results_dir = '/home/k1511004/Data/PSYSCAN/WP5_data/FEP/full_cohort_results/CV_results/'

# set options
min_site_size = 5
max_mean_SP = 7.5

# do we want to exclude subjects where there is a problem with the T1 image?
T1W_motion_exclude_flag = True

# do we want to correct for site?
site_correction = 'None'

# what timepoint do we want to predict?

timepoint = 'Month_06'

# set data type
data_type = 'fMRI'

# excldue site 16?
exclude_site_16 = False

# anatomical covariance matrix parameters: atlas choice and number of histogram bins
atlas = 'Cambridge'
n_bins = 2
hemispheres = 'both'

# set network size
network_size = 'medium'

# number of filters/units
# test network
#n_filters=[1, 1, 1, 1, 4]
if network_size == 'small' :
    
    n_filters=[4, 8, 16, 16, 32]
    
if network_size == 'medium' :    
    
    n_filters=[8, 16, 32, 32, 64]

if network_size == 'big' :
    
    n_filters=[16, 32, 64, 64, 128]

#get job id
job_id = sys.argv[1]
#job_id = '1'
job_id_num = int(job_id)

print ('I am running!')
print ('job id is ' + job_id)

if data_type == 'fMRI' or data_type == 'sMRI_fMRI_subjects' :

    # read in fMRI data
    data_file = 'covariance_data_scaled_GraphLasso.npy'
    data = np.load(fMRI_data_dir + data_file)
    #
    # read in data subjects + motion metadata
    data_subjects = pd.read_excel(metadata_dir + 'FEP_fMRI_253subjs.xlsx')
    
    data = pd.DataFrame(data)
    data_subjects = pd.concat((data_subjects, data), axis=1)
    
    # filter out all data rows with all-zeros
    to_keep = (~(data_subjects.iloc[:, 9:]==0).all(axis=1)).tolist()
    data_subjects = data_subjects.iloc[to_keep, :]
    
    # read in extra metadata for age and sex
    # 1) subject data for sex and DOB
    # add Site column
    # then join on subject
    subject_metadata = pd.read_csv(metadata_dir + 'PSYSCAN_demographics.csv', delimiter='|')
    subject_metadata['Site'] = subject_metadata['Site'] = subject_metadata['Subject ID'].apply(lambda x: int(x[4:6]))
    names = data_subjects.columns.values
    names[0] = 'Subject ID'
    subject_metadata_data = pd.merge(subject_metadata, data_subjects, on='Subject ID', how='inner')
    
    # remove data with excessive motion in the fMRI or bad registration
    to_keep = (subject_metadata_data[['Co-reg?', 'fMRI ok?']]=='ok').all(axis=1).tolist()
    subject_metadata_data = subject_metadata_data.iloc[to_keep, :]
    
    # optionally exclude subjects wtih excessive motion on T1W scan
    if T1W_motion_exclude_flag :
    
        to_keep = (subject_metadata_data['T1w QC'] == 'ok').to_list()
        subject_metadata_data = subject_metadata_data.iloc[to_keep, :]
        
    # read in outcomes
    outcomes = pd.read_excel(outcomes_dir + 'Month_02_Month_06_outcomes.xlsx')
    outcomes.rename({'subjectid':'Subject ID'}, axis=1, inplace=True)
    
    # filter out rows where panss fpr baseline or prediction timepoint is missing/nan
    timepoint_panss_cols = list(filter(lambda x: timepoint + '|panss' in x, outcomes.columns))
    baseline_panss_cols = list(filter(lambda x: 'Baseline|panss' in x, outcomes.columns))
    outcomes = outcomes[outcomes[timepoint_panss_cols].notna().any(axis=1)]
    outcomes = outcomes[outcomes[baseline_panss_cols].notna().any(axis=1)]
    
    # extract the columns we need
    outcomes = outcomes[['Subject ID', 'Baseline|assessment age', 'Baseline|total_panss', timepoint + '|total_panss', timepoint + '|response_to_treatment']]
    
    # join outcomes to metadata/data
    outcomes_metadata_data = pd.merge(outcomes, subject_metadata_data, on='Subject ID')
    
    # remove data from sites with < min_site_size subjects
    site_counts = outcomes_metadata_data['Site'].value_counts()
    to_keep_sites = pd.Series(site_counts[site_counts >= min_site_size].index)
    if exclude_site_16 :
        
        to_keep_sites = to_keep_sites[to_keep_sites.apply(lambda x: not x == 16)] 
    
    
    outcomes_metadata_data = outcomes_metadata_data[outcomes_metadata_data['Site'].isin(to_keep_sites)]
    
    site = outcomes_metadata_data['Site'].values
    train_inds, test_ind = min_two_test_CV(site)
    
    
    # get sMRI data for matching subjects if needed
    if data_type == 'sMRI_fMRI_subjects' :
        
        sMRI_subjects = outcomes_metadata_data['Subject ID'].to_list()        
        n_subjects = len(sMRI_subjects)
        anatomical_covariance_matrix_1, lh_ROIs_1, rh_ROIs_1 = anatomical_covariance_matrix(FS_data_dir + sMRI_subjects[0], atlas, n_bins, 'thickness', hemispheres)
        n_regions = np.shape(anatomical_covariance_matrix_1)[0]
        n_covs = n_regions * n_regions
        tril_inds = np.ravel_multi_index(np.tril_indices(n_regions, k=-1), (n_regions, n_regions))
        anatomical_covariance_data = np.zeros((n_subjects, n_covs))
        anatomical_covariance_data[0, :] = np.reshape(anatomical_covariance_matrix_1, (1, n_covs))
    
        # roll through remaining subjects
        # check ROIs match as we go
        # initialise empty lists of subject with non-matching ROIs
        ROI_mismatch_subjects = []
        ROI_mismatch = False
        for i in range(1, n_subjects) :
    
            print (i)
    
            subj_anatomical_covariance_matrix, lh_ROIs, rh_ROIs = anatomical_covariance_matrix(FS_data_dir + sMRI_subjects[i], atlas, n_bins, 'thickness', hemispheres)
            if (not np.array_equal(lh_ROIs, lh_ROIs_1)) or (not np.array_equal(rh_ROIs, rh_ROIs_1)) :
        
                print ('ROI mismatch!')
                ROI_mismatch = True
                ROI_mismatch_subjects.append(sMRI_subjects[i])
        
                #    try again, this this imposing the ROI labels
                subj_anatomical_covariance_matrix, lh_ROIs, rh_ROIs = anatomical_covariance_matrix(sMRI_data_dir + sMRI_subjects[i], atlas, n_bins, 'thickness', hemispheres, lh_ROI_labels = lh_ROIs_1, rh_ROI_labels = rh_ROIs_1)

            else :
        
                anatomical_covariance_data[i, :] = np.reshape(subj_anatomical_covariance_matrix, (1, n_covs))    
        
        # replace fMRI data in outcomes_metadata_data
        outcomes_metadata_data = outcomes_metadata_data.iloc[:, :10]
        outcomes_metadata_data = outcomes_metadata_data.reset_index(drop=True)
        outcomes_metadata_data = pd.concat([outcomes_metadata_data, pd.DataFrame(anatomical_covariance_data)], axis=1)
        foo =1
        
    # pull out data, metadata and outcomes again
    data = outcomes_metadata_data.iloc[:, 16:].as_matrix()
    metadata = outcomes_metadata_data.iloc[:, :16]
    site = metadata['Site'].values
        
elif data_type == 'sMRI' :
    
    # get the list of available subjects with sMRI
    available_subjects = listdir(FS_data_dir)
    
    if T1W_motion_exclude_flag :
    
        # exclude subjects with motion artifacts
        exclude = pd.read_excel(metadata_dir + 'exclusions_Jonathan.xlsx', header=None, sheet_name='T1w exclusions')[0].tolist()
        available_subjects = list(filter(lambda x: not x in exclude, available_subjects))
        available_subjects = pd.DataFrame({'Subject ID':available_subjects})
        
    # read in extra metadata for age and sex
    # 1) subject data for sex and DOB
    # then join on subject
    subject_metadata = pd.read_csv(metadata_dir + 'PSYSCAN_demographics.csv', delimiter='|')
    subject_metadata = pd.merge(available_subjects, subject_metadata, on='Subject ID', how='inner')
        
    # read in outcomes
    outcomes = pd.read_excel(outcomes_dir + 'Month_02_Month_06_outcomes.xlsx')
    outcomes.rename({'subjectid':'Subject ID'}, axis=1, inplace=True)
    
    # filter out rows where panss fpr baseline or prediction timepoint is missing/nan, or age is nan
    timepoint_panss_cols = list(filter(lambda x: timepoint + '|panss' in x, outcomes.columns))
    baseline_panss_cols = list(filter(lambda x: 'Baseline|panss' in x, outcomes.columns))
    outcomes = outcomes[outcomes[timepoint_panss_cols].notna().any(axis=1)]
    outcomes = outcomes[outcomes[baseline_panss_cols].notna().any(axis=1)]
    outcomes = outcomes[outcomes['Baseline|assessment age'].notna()]
    
    # extract the columns we need
    outcomes = outcomes[['Subject ID', 'Baseline|assessment age', 'Baseline|total_panss', timepoint + '|total_panss', timepoint + '|response_to_treatment']]
    
    # join outcomes to metadata/data
    outcomes_metadata = pd.merge(outcomes, subject_metadata, on='Subject ID')
    outcomes_metadata['Site'] = outcomes_metadata['Subject ID'].apply(lambda x: int(x[4:6]))
    
    # remove data from sites with < min_site_size subjects
    site_counts = outcomes_metadata['Site'].value_counts()
    to_keep_sites = pd.Series(site_counts[site_counts >= min_site_size].index)
    if exclude_site_16 :
        
        to_keep_sites = to_keep_sites[to_keep_sites.apply(lambda x: not x == 16)] 
    outcomes_metadata = outcomes_metadata[outcomes_metadata['Site'].isin(to_keep_sites)]
    
    # use list of avaailable subjects to generate sMRI data
    sMRI_subjects = outcomes_metadata['Subject ID'].to_list()        
    n_subjects = len(sMRI_subjects)
    anatomical_covariance_matrix_1, lh_ROIs_1, rh_ROIs_1 = anatomical_covariance_matrix(FS_data_dir + sMRI_subjects[0], atlas, n_bins, 'thickness', hemispheres)
    n_regions = np.shape(anatomical_covariance_matrix_1)[0]
    n_covs = n_regions * n_regions
    tril_inds = np.ravel_multi_index(np.tril_indices(n_regions, k=-1), (n_regions, n_regions))
    anatomical_covariance_data = np.zeros((n_subjects, n_covs))
    anatomical_covariance_data[0, :] = np.reshape(anatomical_covariance_matrix_1, (1, n_covs))
    
    # roll through remaining subjects
    # check ROIs match as we go
    # initialise empty lists of subject with non-matching ROIs
    ROI_mismatch_subjects = []
    ROI_mismatch = False
    for i in range(1, n_subjects) :

        print (i)

        subj_anatomical_covariance_matrix, lh_ROIs, rh_ROIs = anatomical_covariance_matrix(FS_data_dir + sMRI_subjects[i], atlas, n_bins, 'thickness', hemispheres)
        if (not np.array_equal(lh_ROIs, lh_ROIs_1)) or (not np.array_equal(rh_ROIs, rh_ROIs_1)) :
    
            print ('ROI mismatch!')
            ROI_mismatch = True
            ROI_mismatch_subjects.append(sMRI_subjects[i])
    
            #    try again, this this imposing the ROI labels
            subj_anatomical_covariance_matrix, lh_ROIs, rh_ROIs = anatomical_covariance_matrix(sMRI_data_dir + sMRI_subjects[i], atlas, n_bins, 'thickness', hemispheres, lh_ROI_labels = lh_ROIs_1, rh_ROI_labels = rh_ROIs_1)

        else :
    
            anatomical_covariance_data[i, :] = np.reshape(subj_anatomical_covariance_matrix, (1, n_covs))
        
        
    foo=1
    
    # rename for compatibility
    data = anatomical_covariance_data
    metadata = outcomes_metadata
    site = metadata['Site'].values

n_subjects, n_connections = np.shape(data)
n_regions = int(np.sqrt(n_connections))
logm_connectivity_data = np.zeros((n_subjects, n_connections))

# do site correction
if data_type == 'fMRI' and (site_correction == 'comBat_unsupervised' or site_correction == 'comBat_supervised' or site_correction == 'None') :

    # take matrix logs at start
    for i in range(n_subjects) :
    
        print(i)
    
        connectivity_vector = data[i, :]
        connectivity_matrix = np.reshape(connectivity_vector, (n_regions, n_regions))
        logm_connectivity_matrix = logm(connectivity_matrix)
        logm_connectivity_vector = np.reshape(logm_connectivity_matrix, (n_regions * n_regions, ))
        logm_connectivity_data[i, :] = logm_connectivity_vector
        
if data_type == 'sMRI' or data_type == 'sMRI_fMRI_subjects' :
    
    logm_connectivity_data = data

# do unsupervised comBat correction if that is selected
if site_correction == 'comBat_unsupervised' : 
       
    logm_connectivity_data = neuroCombat(logm_connectivity_data, metadata, batch_col = 'Site')
    
# do MGLM correction if that is selected
if site_correction == 'MGLM' : 
        
    # pull out site variable and convert to one-hot
    enc = OneHotEncoder(sparse=False)
    enc.fit(site.reshape(-1,1))
    site_one_hot = enc.transform(site.reshape(-1, 1))
    site_one_hot = np.transpose(site_one_hot)

    # transpose and reshape the connectivity data
    data_perm = np.reshape(data, (n_subjects, n_regions, n_regions))
    data_perm = np.transpose(data_perm, (1, 2, 0))
    
    # generate MGLM regressed data
    p, V, E, Y_hat, gnorm = mglm_spd(site_one_hot, data_perm, 100)
    
    # calculate matrix 'residuals'
    for i in range(n_subjects) :
        
        mglm_corrected_matrix = Y_hat[:, :, i]
        mglm_residuals_matrix = logmap_spd(mglm_corrected_matrix, data_perm[:, :, i])
        mglm_residuals_vector = np.reshape(mglm_residuals_matrix, (n_regions * n_regions, ))
        logm_connectivity_data[i, :] = mglm_residuals_vector

# set up MCCV
n_repeats = 20
test_fraction = 0.2
train_inds_all, test_inds_all, test_size = ShuffleSplitFixed(logm_connectivity_data, site, site, test_fraction, n_repeats)
train_size = n_subjects - test_size
#ss = StratifiedShuffleSplit(n_splits = n_repeats, test_size = test_size,    )

# data structures to hold results
predicted_followup_total_PANSS = np.zeros((n_repeats * test_size, 1))
test_baseline_total_PANSS = np.zeros_like(predicted_followup_total_PANSS)
test_followup_total_PANSS = np.zeros_like(predicted_followup_total_PANSS)

# pull out targets (followup total PANSS), baseline total PANSS and subject ids
baseline_total_PANSS = metadata['Baseline|total_panss'].values
followup_total_PANSS = metadata[timepoint + '|total_panss'].values
subjectids = metadata['Subject ID'].to_list()

# initialise list of test subjects
test_subjects = []


# initialise regressor
rgr = KernelRidge(kernel='precomputed')    
rgr_lin = KernelRidge(kernel='linear')
enc = OneHotEncoder(sparse=False)

# prepare other variables for stacking: age, sex, site
# encode sex numerically and site as one-hot
# stack them together
age = metadata['Baseline|assessment age'].values
sex = metadata['Gender'].to_list()
site = metadata['Site'].values
sex = np.array([0 if sex == 'M' else 1 for sex in sex])
baseline_total_PANSS = metadata['Baseline|total_panss'].values
enc.fit(site.reshape(-1,1))
site_one_hot = enc.transform(site.reshape(-1, 1))
#stacking_vars = np.hstack((age[:, np.newaxis], sex[:, np.newaxis], baseline_total_PANSS[:, np.newaxis], site_one_hot))
stacking_vars = np.hstack((age[:, np.newaxis], sex[:, np.newaxis], site_one_hot))

# set up MCCV
# set random number seed     
np.random.seed(job_id_num) 

# do MCCV
for i in range(n_repeats) :
    
    train_index = train_inds_all[i]
    test_index = test_inds_all[i]

    print (i)    
    
    # generate (outer) train and test data and metadata
    train_data = logm_connectivity_data[train_index, :]
    test_data = logm_connectivity_data[test_index, :]        
    train_metadata = metadata.iloc[train_index, :]
    
    # calculate output indices
    start_ind = i * test_size
    stop_ind = start_ind + test_size
    
    train_targets = followup_total_PANSS[train_index]
    test_targets = followup_total_PANSS[test_index]
    test_subjects = test_subjects + list(np.array(subjectids)[test_index])    
        
    # generate loo training prediction
    loo = LeaveOneOut()
    inner_predictions = np.zeros((train_size,))
    inner_train_indices, inner_test_indices = min_two_test_CV(site[train_index]) 
    #for j in range(len(inner_train_indices)) :
    #for inner_train_index, inner_test_index in loo.split(train_data) :
    skf = StratifiedKFold(n_splits=3)
    j = 0
    for inner_train_index, inner_test_index in skf.split(train_data, site[train_index]):
    
#            print ('i = ' +str(i))
#            print ('j = ' +str(j))
#            
#            inner_train_index = inner_train_indices[j]
#            inner_test_index = inner_test_indices[j]
#      print  
        
        print ('inner loop iteration ' + str(j+1))
        # re-split data
        inner_train_data = train_data[inner_train_index, :]
        inner_test_data = train_data[inner_test_index, :]
        inner_train_targets = train_targets[inner_train_index]
        inner_train_size = len(inner_train_index)
        inner_test_size = len(inner_test_index)
        
        # do supervised site correction?
        if site_correction == 'comBat_supervised' :
        
            # correct inner training data
            #inner_train_data, LS_dict, v_pool = fit_transform_neuroCombat(inner_train_data, train_metadata.iloc[inner_train_index, :], 'Site', continuous_cols=[timepoint + '|total_panss'])
            inner_train_data, model = fit_transform_neuroCombat(inner_train_data, train_metadata.iloc[inner_train_index, :], 'Site', continuous_cols=[timepoint + '|total_panss'])
            # correct testing data
            inner_test_data = apply_neuroCombat_model(inner_test_data,
                                                train_metadata.iloc[inner_test_index, :],
                                                model,
                                                'Site')
        
        # make inner predictions
        inner_train_matrices = np.reshape(inner_train_data, (inner_train_size, n_regions, n_regions))
        inner_test_matrices = np.reshape(inner_test_data, (inner_test_size, n_regions, n_regions))
        inner_train_matrices = inner_train_matrices[:, :, :, np.newaxis]
        inner_test_matrices = inner_test_matrices[:, :, :, np.newaxis]
        
        model = brainnetCNN_model_2((n_regions, n_regions, 1), n_filters, use_bias)
        #   my custom
        #   opt = SGD(lr=0.002, momentum=0.2, decay=0.01)
        # default
        opt = Adam()        
        model.compile(loss='mean_absolute_error', optimizer=opt, metrics=['mean_absolute_error'])
    
        # fit and predict
        model.fit(inner_train_matrices, inner_train_targets ,epochs=100, batch_size=20, verbose=1)
        preds = model.predict(inner_test_matrices)
        inner_predictions[inner_test_index] = np.squeeze(preds)         
        
    # split stacking vars and add inner predictions to the testing ones
    train_stacking_vars = stacking_vars[train_index, :]
    test_stacking_vars = stacking_vars[test_index, :]
    train_stacking_vars = np.hstack((train_stacking_vars, inner_predictions[:, np.newaxis]))
    
    # do supervised site correction?
    if site_correction == 'comBat_supervised' :
            
        # correct outer training data
        train_data, model = fit_transform_neuroCombat(train_data, metadata.iloc[train_index, :], 'Site', continuous_cols=[timepoint + '|total_panss'])
        
        # correct outer testing dara
        test_data = apply_neuroCombat_model(test_data,
                      metadata.iloc[test_index, :],
                      model,
                      'Site')
    
    # make base predictions and add them to test stacking vars
    train_matrices = np.reshape(train_data, (train_size, n_regions, n_regions))
    test_matrices = np.reshape(test_data, (test_size, n_regions, n_regions))
    train_matrices = train_matrices[:, :, :, np.newaxis]
    test_matrices = test_matrices[:, :, :, np.newaxis]
    model = brainnetCNN_model_2((n_regions, n_regions, 1), n_filters, use_bias)
    # my custom
    #opt = SGD(lr=0.002, momentum=0.2, decay=0.01)
    # default
    opt = Adam()
    model.compile(loss='mean_absolute_error', optimizer=opt, metrics=['mean_absolute_error'])
    
    # fit and predict
    model.fit(train_matrices, train_targets ,epochs=100, batch_size=20, verbose=1)
    base_preds = model.predict(test_matrices)
    test_stacking_vars = np.hstack((test_stacking_vars, base_preds))
    
    
    # scale the stacking vars based on train only
    train_stacking_min = np.min(train_stacking_vars, axis=0)
    train_stacking_vars = train_stacking_vars - train_stacking_min
    test_stacking_vars = test_stacking_vars - train_stacking_min
    train_stacking_max = np.max(train_stacking_vars, axis=0)
    train_stacking_vars = train_stacking_vars / train_stacking_max
    test_stacking_vars = test_stacking_vars / train_stacking_max
    
    # make kernels
    train_stacking_kernel = np.dot(train_stacking_vars, np.transpose(train_stacking_vars))
    cross_stacking_kernel = np.dot(test_stacking_vars, np.transpose(train_stacking_vars))
    
    # make base predictions and add them to test stacking vars
    rgr.fit(train_stacking_kernel, train_targets)
    final_preds = rgr.predict(cross_stacking_kernel) 
    
    # store results
    predicted_followup_total_PANSS[start_ind:stop_ind, 0] = final_preds
    test_baseline_total_PANSS[start_ind:stop_ind, 0] = baseline_total_PANSS[test_index]
    test_followup_total_PANSS[start_ind:stop_ind, 0] = followup_total_PANSS[test_index]

# put results in a DF with subjects
test_subjects = list(map(lambda x: str(x), test_subjects))
subjects_response_results = pd.DataFrame(test_subjects)
subjects_response_results['baseline total PANSS'] = pd.Series(np.squeeze(test_baseline_total_PANSS))
subjects_response_results['followup total PANSS'] = pd.Series(np.squeeze(test_followup_total_PANSS))
subjects_response_results['predicted followup total PANSS'] = pd.Series(np.squeeze(predicted_followup_total_PANSS))
subjects_response_results['response to treatment'] = subjects_response_results[['followup total PANSS', 'baseline total PANSS']].apply(response_to_treatment, axis=1)
subjects_response_results['predicted response to treatment'] = subjects_response_results[['predicted followup total PANSS', 'baseline total PANSS']].apply(response_to_treatment, axis=1)
subjects_response_results['total PANSS followup/baseline ratio'] = subjects_response_results['baseline total PANSS']/subjects_response_results['predicted followup total PANSS']
subjects_response_results['Site'] = subjects_response_results[0].apply(lambda x: int(x[4:6]))

# save results DF
subjects_response_results.to_csv(results_dir + 'predict_treatment_response_stacked_results_brainnet_CNN_MSE_' + data_type + '_' + network_size + '_' + site_correction + '_CV_fold_' + job_id + '.csv')