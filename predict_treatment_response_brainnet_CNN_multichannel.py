#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 09:53:20 2020

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

from scipy.linalg import logm
import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn import svm
#from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, classification_report, r2_score, mean_squared_error, recall_score, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from neuroCombat import neuroCombat
from neuroCombatCV3 import fit_transform_neuroCombat, apply_neuroCombat_model, ShuffleSplitFixed
from sklearn.preprocessing import OneHotEncoder
from riem_mglm import mglm_spd, logmap_spd
from sklearn.metrics import mean_absolute_error, roc_auc_score, balanced_accuracy_score
from scipy.stats import pearsonr
from datetime import datetime
from anatomical_covariance import cortical_regional_means, anatomical_covariance_matrix
from os import listdir
from brainNetCNN_keras import EdgeToEdge, EdgeToNode, NodeToGraph
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import LabelEncoder
#from remove_confounds_fast import remove_confounds_fast
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, Dropout, BatchNormalization
from tensorflow.keras.activations import relu, sigmoid
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, RMSprop, Adadelta, Adam

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


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

#from tensorflow.keras import backend as K
#print(K.tensorflow_backend._get_available_gpus())


# set directories 
#sMRI_data_dir = '/home/k1511004/Data/PSYSCAN/WP5_data/Prelim_FEP_dataset/174_subjects_Oct_19/Data/sMRI/'
sMRI_data_dir = '/home/k1511004/Data/PSYSCAN/WP5_data/Prelim_FEP_dataset/253_subjects_Apr_20/Data/sMRI/FS_data/'
fMRI_data_dir = '/home/k1511004/Data/PSYSCAN/WP5_data/Prelim_FEP_dataset/253_subjects_Apr_20/Data/fMRI/'
FS_data_dir = sMRI_data_dir
metadata_dir = '/home/k1511004/Data/PSYSCAN/WP5_data/Prelim_FEP_dataset/253_subjects_Apr_20/metadata/'
outcomes_dir = '/home/k1511004/Data/PSYSCAN/WP5_data/FEP/metadata/'
results_dir = '/home/k1511004/Data/PSYSCAN/WP5_data/FEP/full_cohort_results/'

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
# small network
#
#bigger network
#

# regions used in fMRI
fMRI_regions = np.genfromtxt(fMRI_data_dir + 'regions_FEP.dat')
fMRI_regions = (fMRI_regions - 1).astype(int)

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
    
# get sMRI data for matching subjects
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

        # try again, this this imposing the ROI labels
        subj_anatomical_covariance_matrix, lh_ROIs, rh_ROIs = anatomical_covariance_matrix(sMRI_data_dir + sMRI_subjects[i], atlas, n_bins, 'thickness', hemispheres, lh_ROI_labels = lh_ROIs_1, rh_ROI_labels = rh_ROIs_1)

    else :
        
        anatomical_covariance_data[i, :] = np.reshape(subj_anatomical_covariance_matrix, (1, n_covs))    
 
# pull out fMRI data
fMRI_data = outcomes_metadata_data.iloc[:, 16:].values
metadata = outcomes_metadata_data.iloc[:, :16]
       
n_subjects, n_fMRI_connections = np.shape(fMRI_data)
n_fMRI_regions = int(np.sqrt(n_fMRI_connections))

sMRI_data = anatomical_covariance_data
n_subjects, n_sMRI_connections = np.shape(sMRI_data)
n_sMRI_regions = int(np.sqrt(n_sMRI_connections))

# take fMRI matrix logs at start
for i in range(n_subjects) :

    print(i)

    fMRI_connectivity_vector = fMRI_data[i, :]
    fMRI_connectivity_matrix = np.reshape(fMRI_connectivity_vector, (n_fMRI_regions, n_fMRI_regions))
    logm_fMRI_connectivity_matrix = logm(fMRI_connectivity_matrix)
    logm_fMRI_connectivity_vector = np.reshape(logm_fMRI_connectivity_matrix, (n_fMRI_connections, ))
    fMRI_data[i, :] = logm_fMRI_connectivity_vector

# do unsupervised comBat correction if that is selected
if site_correction == 'comBat_unsupervised' : 
       
    fMRI_data = neuroCombat(fMRI_data, metadata, batch_col = 'Site')
    sMRI_data = neuroCombat(sMRI_data, metadata, batch_col = 'Site')

# take tril inds
#fMRI_tril_inds = np.ravel_multi_index(np.tril_indices(n_regions, k=-1), (n_fMRI_regions, n_fMRI_regions))    
#fMRI_data = fMRI_data[:, fMRI_tril_inds]
#sMRI_tril_inds = np.ravel_multi_index(np.tril_indices(n_regions, k=-1), (n_sMRI_regions, n_sMRI_regions))    
#sMRI_data = sMRI_data[:, sMRI_tril_inds]



# set up MCCV
site = metadata['Site'].values
n_repeats = 200
test_fraction = 0.2
train_inds_all, test_inds_all, test_size = ShuffleSplitFixed(fMRI_data, site, site, test_fraction, n_repeats)
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

# do MCCV
for i in range(n_repeats) :
    
    train_index = train_inds_all[i]
    test_index = test_inds_all[i]

    print (i)    
    
    # calculate output indices
    start_ind = i * test_size
    stop_ind = start_ind + test_size
    
    train_targets = followup_total_PANSS[train_index]
    test_targets = followup_total_PANSS[test_index]
    test_subjects = test_subjects + list(np.array(subjectids)[test_index])
    
    train_fMRI_data = fMRI_data[train_index, :]
    test_fMRI_data = fMRI_data[test_index, :]
    train_sMRI_data = sMRI_data[train_index, :]
    test_sMRI_data = sMRI_data[test_index, :]
    
    # do supervised site correction?
    if site_correction == 'comBat_supervised' :
        
    
        # correct training data
        train_fMRI_data, fMRI_model = fit_transform_neuroCombat(train_fMRI_data, metadata.iloc[train_index, :], 'Site', continuous_cols=[timepoint + '|total_panss'])
        train_sMRI_data, sMRI_model = fit_transform_neuroCombat(train_sMRI_data, metadata.iloc[train_index, :], 'Site', continuous_cols=[timepoint + '|total_panss'])
        
        # correct testing dara
        test_fMRI_data = apply_neuroCombat_model(test_fMRI_data,
                      metadata.iloc[test_index, :], fMRI_model,
                      'Site')
        test_sMRI_data = apply_neuroCombat_model(test_sMRI_data,
                      metadata.iloc[test_index, :], sMRI_model,
                      'Site')
    
    #recreate a stack of matrices for train and test
    # add extra singleton dimension for channels
    train_fMRI_matrices = np.reshape(train_fMRI_data, (train_size, n_fMRI_regions, n_fMRI_regions))
    test_fMRI_matrices = np.reshape(test_fMRI_data, (test_size, n_fMRI_regions, n_fMRI_regions))
    train_fMRI_matrices = train_fMRI_matrices[:, :, :, np.newaxis]
    test_fMRI_matrices = test_fMRI_matrices[:, :, :, np.newaxis]
    train_sMRI_matrices = np.reshape(train_sMRI_data, (train_size, n_sMRI_regions, n_sMRI_regions))
    test_sMRI_matrices = np.reshape(test_sMRI_data, (test_size, n_sMRI_regions, n_sMRI_regions))
    train_sMRI_matrices = train_sMRI_matrices[:, :, :, np.newaxis]
    test_sMRI_matrices = test_sMRI_matrices[:, :, :, np.newaxis]
    
    # take sMRI regions to match fMRI_regions
    train_sMRI_matrices = train_sMRI_matrices[:, fMRI_regions, :, :]
    train_sMRI_matrices = train_sMRI_matrices[:, :, fMRI_regions, :]
    test_sMRI_matrices = test_sMRI_matrices[:, fMRI_regions, :, :]
    test_sMRI_matrices = test_sMRI_matrices[:, :, fMRI_regions, :]
    
    # stack together fMRI and sMRI data for multichannel data
    train_matrices = np.concatenate((train_fMRI_matrices, train_sMRI_matrices), axis=3)
    test_matrices = np.concatenate((test_fMRI_matrices, test_sMRI_matrices), axis=3)
    
    
    # create and compile the model
    model = brainnetCNN_model_2((n_fMRI_regions, n_fMRI_regions, 2), n_filters, use_bias)
    # my custom
    #opt = SGD(lr=0.002, momentum=0.2, decay=0.01)
    # default
    opt = Adam()
    model.compile(loss='mean_absolute_error', optimizer=opt, metrics=['mean_absolute_error'])
    
    # fit and predict
    model.fit(train_matrices, train_targets ,epochs=100, batch_size=20, verbose=1)
    preds = model.predict(test_matrices)
    
    # store results
    predicted_followup_total_PANSS[start_ind:stop_ind, 0] = np.squeeze(preds)
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
pred_PANSS = subjects_response_results['predicted followup total PANSS'].values
true_PANSS = subjects_response_results['followup total PANSS'].values
PANSS_followup_ratio = subjects_response_results['total PANSS followup/baseline ratio'].values
mae = mean_absolute_error(pred_PANSS, true_PANSS)
r2 = pearsonr(pred_PANSS, true_PANSS)
std = np.std(true_PANSS)

# open file for storing results
date_string = datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")
results_file = open(results_dir + 'predict_treatment_response_results_brainnet_CNN_multichannel_MAE_' + data_type + '_' + network_size + '_' + site_correction + '_' + date_string + '.txt', "w") 

# accuracy of predicted response/nonresponse
pred_response = subjects_response_results['predicted response to treatment'].values
true_response = subjects_response_results['response to treatment'].values
bal_acc = balanced_accuracy_score(true_response, pred_response)
roc_auc = roc_auc_score(true_response, PANSS_followup_ratio)

print ('Overall results:')
print ('Total PANSS prediction - mae = ' + str(mae) + ' (std) = ' + str(std))
print ('Total PANSS prediction - correlation = ' + str(r2[0]))
print ('Response/non-response prediction - balanced accuracy = ' + str(bal_acc))
print ('Response/non-response prediction - AUC = ' + str(roc_auc))
results_file.write(str(mae) + '_(' + str(std) + ')\n')
results_file.write(str(r2[0]) + '\n')
results_file.write(str(bal_acc) + '\n')
results_file.write(str(roc_auc) + '\n')

# look at site breakdowns
unique_sites = list(set(site))

print ('Results by site:')
for site in unique_sites:
    
    print ('Site ' + str(site) + ' results:')
    site_results = subjects_response_results[subjects_response_results['Site'] == site]
    pred_PANSS = site_results['predicted followup total PANSS'].values
    true_PANSS = site_results['followup total PANSS'].values
    PANSS_followup_ratio = site_results['total PANSS followup/baseline ratio'].values
    mae = mean_absolute_error(pred_PANSS, true_PANSS)
    r2 = pearsonr(pred_PANSS, true_PANSS)
    std = np.std(true_PANSS)
    pred_response = site_results['predicted response to treatment'].values
    true_response = site_results['response to treatment'].values
    bal_acc = balanced_accuracy_score(true_response, pred_response)
    try :
    
        roc_auc = roc_auc_score(true_response, PANSS_followup_ratio)
        
    except ValueError :
        
        roc_auc = 'No_AUC_as_only_one_class_is_present_for_this_site'
    
    print ('Total PANSS prediction - mae = ' + str(mae) + ' (std) = ' + str(std))
    print ('Total PANSS prediction - correlation = ' + str(r2[0]))
    print ('Response/non-response prediction - balanced accuracy = ' + str(bal_acc))
    print ('Response/non-response prediction - AUC = ' + str(roc_auc))
    results_file.write(str(mae) + '_(' + str(std) + ')\n')
    results_file.write(str(r2[0]) + '\n')
    results_file.write(str(bal_acc) + '\n')
    results_file.write(str(roc_auc) + '\n')
    
# close the results file
results_file.close()    
