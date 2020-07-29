#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 11:31:45 2020

@author: k1511004
"""

# import what we need
import sys
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
site_correction = 'comBat_supervised'

# what timepoint do we want to predict?

timepoint = 'Month_06'

# excldue site 16?
exclude_site_16 = False

# anatomical covariance matrix parameters: atlas choice and number of histogram bins
atlas = 'Cambridge'
n_bins = 2
hemispheres = 'both'

print ('I am running!')

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

        #    try again, this this imposing the ROI labels
        subj_anatomical_covariance_matrix, lh_ROIs, rh_ROIs = anatomical_covariance_matrix(sMRI_data_dir + sMRI_subjects[i], atlas, n_bins, 'thickness', hemispheres, lh_ROI_labels = lh_ROIs_1, rh_ROI_labels = rh_ROIs_1)

    else :

        anatomical_covariance_data[i, :] = np.reshape(subj_anatomical_covariance_matrix, (1, n_covs))    

# pull out data, metadata and outcomes again
metadata = outcomes_metadata_data.iloc[:, :16]
fMRI_data = outcomes_metadata_data.iloc[:, 16:].as_matrix()
site = metadata['Site'].values

# process fMRI data: matrix logs and lower triangle
n_subjects, n_connections = np.shape(fMRI_data)
n_regions = int(np.sqrt(n_connections))
tril_inds = np.ravel_multi_index(np.tril_indices(n_regions, k=-1), (n_regions, n_regions))
n_connectivities = len(tril_inds)
logm_fMRI_connectivity_data = np.zeros((n_subjects, n_connectivities))
for i in range(n_subjects) :
    
    print(i)
    
    connectivity_vector = fMRI_data[i, :]
    connectivity_matrix = np.reshape(connectivity_vector, (n_regions, n_regions))
    logm_connectivity_matrix = logm(connectivity_matrix)
    logm_connectivity_vector = np.reshape(logm_connectivity_matrix, (n_regions * n_regions, ))
    logm_fMRI_connectivity_data[i, :] = logm_connectivity_vector[tril_inds]

# process sMRI data: lower triangle only
n_subjects, n_connections = np.shape(anatomical_covariance_data)
n_regions = int(np.sqrt(n_connections))
tril_inds = np.ravel_multi_index(np.tril_indices(n_regions, k=-1), (n_regions, n_regions))
n_connectivities = len(tril_inds)
anatomical_covariance_data = anatomical_covariance_data[:, tril_inds]

# do unsupervised comBat correction if that is selected
if site_correction == 'comBat_unsupervised' : 
       
    logm_fMRI_connectivity_data = neuroCombat(logm_fMRI_connectivity_data, metadata, batch_col = 'Site')
    anatomical_covariance_data = neuroCombat(anatomical_covariance_data, metadata, batch_col = 'Site')
    
# set up MCCV
n_repeats = 200
test_fraction = 0.2
train_inds_all, test_inds_all, test_size = ShuffleSplitFixed(logm_fMRI_connectivity_data, site, site, test_fraction, n_repeats)
train_size = n_subjects - test_size
ss = StratifiedShuffleSplit(n_splits = n_repeats, test_size = test_size,    )

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

# precompute kernel matrices
K_fMRI = np.dot(logm_fMRI_connectivity_data, np.transpose(logm_fMRI_connectivity_data))
K_sMRI = np.dot(anatomical_covariance_data, np.transpose(anatomical_covariance_data))

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
    
    # do supervised site correction?
    if site_correction == 'comBat_supervised' :
        
        # generate (outer) train and test data and metadata
        train_fMRI_data = logm_fMRI_connectivity_data[train_index, :]
        test_fMRI_data = logm_fMRI_connectivity_data[test_index, :]
        train_sMRI_data = anatomical_covariance_data[train_index, :]
        test_sMRI_data = anatomical_covariance_data[test_index, :]         
        train_metadata = metadata.iloc[train_index, :]
        
        # generate loo training prediction
        loo = LeaveOneOut()
        inner_fMRI_predictions = np.zeros((train_size,))
        inner_sMRI_predictions = np.zeros((train_size,))
        inner_train_indices, inner_test_indices = min_two_test_CV(site[train_index]) 
        #for j in range(len(inner_train_indices)) :
        #for inner_train_index, inner_test_index in loo.split(train_data) :
        skf = StratifiedKFold(n_splits=3)
        for inner_train_index, inner_test_index in skf.split(train_fMRI_data, site[train_index]):
        
#            print ('i = ' +str(i))
#            print ('j = ' +str(j))
#            
#            inner_train_index = inner_train_indices[j]
#            inner_test_index = inner_test_indices[j]
#        
            # re-split data
            inner_train_fMRI_data = train_fMRI_data[inner_train_index, :]
            inner_test_fMRI_data = train_fMRI_data[inner_test_index, :]
            inner_train_sMRI_data = train_sMRI_data[inner_train_index, :]
            inner_test_sMRI_data = train_sMRI_data[inner_test_index, :]
            inner_train_targets = train_targets[inner_train_index]
            
            # correct inner training data 
            #inner_train_data, LS_dict, v_pool = fit_transform_neuroCombat(inner_train_data, train_metadata.iloc[inner_train_index, :], 'Site', continuous_cols=[timepoint + '|total_panss'])
            inner_train_fMRI_data, model = fit_transform_neuroCombat(inner_train_fMRI_data, train_metadata.iloc[inner_train_index, :], 'Site', continuous_cols=[timepoint + '|total_panss'])
            # correct testing data
            inner_test_fMRI_data = apply_neuroCombat_model(inner_test_fMRI_data,
                                                train_metadata.iloc[inner_test_index, :],
                                                model,
                                                'Site')
            
            inner_train_sMRI_data, model = fit_transform_neuroCombat(inner_train_sMRI_data, train_metadata.iloc[inner_train_index, :], 'Site', continuous_cols=[timepoint + '|total_panss'])
            # correct testing data
            inner_test_sMRI_data = apply_neuroCombat_model(inner_test_sMRI_data,
                                                train_metadata.iloc[inner_test_index, :],
                                                model,
                                                'Site')
            
            # make inner predictions
            rgr_lin.fit(inner_train_fMRI_data, inner_train_targets)
            inner_fMRI_pred = rgr_lin.predict(inner_test_fMRI_data)
            inner_fMRI_predictions[inner_test_index] = inner_fMRI_pred
            rgr_lin.fit(inner_train_sMRI_data, inner_train_targets)
            inner_sMRI_pred = rgr_lin.predict(inner_test_sMRI_data)
            inner_sMRI_predictions[inner_test_index] = inner_sMRI_pred
            
        # split stacking vars and add inner predictions to the testing ones
        train_stacking_vars = stacking_vars[train_index, :]
        test_stacking_vars = stacking_vars[test_index, :]
        train_stacking_vars = np.hstack((train_stacking_vars, inner_fMRI_predictions[:, np.newaxis], inner_sMRI_predictions[:, np.newaxis]))
                
        # correct outer training data
        train_fMRI_data, model = fit_transform_neuroCombat(train_fMRI_data, metadata.iloc[train_index, :], 'Site', continuous_cols=[timepoint + '|total_panss'])
        
        # correct outer testing data
        test_fMRI_data = apply_neuroCombat_model(test_fMRI_data,
                      metadata.iloc[test_index, :],
                      model,
                      'Site')
        
        # correct outer training data
        train_sMRI_data, model = fit_transform_neuroCombat(train_sMRI_data, metadata.iloc[train_index, :], 'Site', continuous_cols=[timepoint + '|total_panss'])
        
        # correct outer testing data
        test_sMRI_data = apply_neuroCombat_model(test_sMRI_data,
                      metadata.iloc[test_index, :],
                      model,
                      'Site')
        
        # make base predictions and add them to test stacking vars
        rgr_lin.fit(train_fMRI_data, train_targets)
        base_fMRI_preds = rgr_lin.predict(test_fMRI_data) 
        rgr_lin.fit(train_sMRI_data, train_targets)
        base_sMRI_preds = rgr_lin.predict(test_sMRI_data) 
        test_stacking_vars = np.hstack((test_stacking_vars, base_fMRI_preds[:, np.newaxis], base_sMRI_preds[:, np.newaxis]))
    # no correction or unsupervised correction    
    else :
        
        # generate (outer) train and test kernels
        train_K_fMRI  = K_fMRI[train_index, :][:, train_index]
        cross_K_fMRI = K_fMRI[test_index, :][:, train_index]
        train_K_sMRI  = K_sMRI[train_index, :][:, train_index]
        cross_K_sMRI = K_sMRI[test_index, :][:, train_index]
        
        # generate loo training prediction
        loo = LeaveOneOut()
        inner_predictions_fMRI = np.zeros((train_size,))
        inner_predictions_sMRI = np.zeros((train_size,))
        for inner_train_index, inner_test_index in loo.split(train_K_fMRI) :
            
            # generate loo training prediction
            inner_train_K_fMRI = train_K_fMRI[inner_train_index, :][:, inner_train_index]
            inner_cross_K_fMRI = train_K_fMRI[inner_test_index, :][:, inner_train_index]
            inner_train_K_sMRI = train_K_sMRI[inner_train_index, :][:, inner_train_index]
            inner_cross_K_sMRI = train_K_sMRI[inner_test_index, :][:, inner_train_index]
            inner_train_targets = train_targets[inner_train_index]
            rgr.fit(inner_train_K_fMRI, inner_train_targets)
            inner_pred_fMRI = rgr.predict(inner_cross_K_fMRI)
            inner_predictions_fMRI[inner_test_index] = inner_pred_fMRI
            rgr.fit(inner_train_K_sMRI, inner_train_targets)
            inner_pred_sMRI = rgr.predict(inner_cross_K_sMRI)
            inner_predictions_sMRI[inner_test_index] = inner_pred_fMRI
        
        # split stacking vars and add inner predictions to the testing ones
        train_stacking_vars = stacking_vars[train_index, :]
        test_stacking_vars = stacking_vars[test_index, :]
        train_stacking_vars = np.hstack((train_stacking_vars, inner_predictions_fMRI[:, np.newaxis], inner_predictions_sMRI[:, np.newaxis]))
    
        # make base predictions and add them to test stacking vars
        rgr.fit(train_K_fMRI, train_targets)
        base_preds_fMRI = rgr.predict(cross_K_fMRI) 
        rgr.fit(train_K_sMRI, train_targets)
        base_preds_sMRI = rgr.predict(cross_K_sMRI) 
        test_stacking_vars = np.hstack((test_stacking_vars, base_preds_fMRI[:, np.newaxis], base_preds_fMRI[:, np.newaxis]))
    
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
pred_PANSS = subjects_response_results['predicted followup total PANSS'].values
true_PANSS = subjects_response_results['followup total PANSS'].values
PANSS_followup_ratio = subjects_response_results['total PANSS followup/baseline ratio'].values
mae = mean_absolute_error(pred_PANSS, true_PANSS)
r2 = pearsonr(pred_PANSS, true_PANSS)
std = np.std(true_PANSS)

# open file for storing results
date_string = datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")
results_file = open(results_dir + 'predict_treatment_response_results_fMRI_sMRI_stacking_' + date_string + '.txt', "w") 

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