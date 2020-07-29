#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 11:56:16 2020

@author: k1511004
"""

# import what we need
import sys
from scipy.linalg import logm
import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn import svm
#from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, classification_report, r2_score, mean_squared_error, recall_score, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression, Ridge
from neuroCombat import neuroCombat
from neuroCombatCV3 import fit_transform_neuroCombat, apply_neuroCombat_model, ShuffleSplitFixed
from sklearn.preprocessing import OneHotEncoder
from riem_mglm import mglm_spd, logmap_spd
from sklearn.metrics import mean_absolute_error, roc_auc_score, balanced_accuracy_score
from scipy.stats import pearsonr
from datetime import datetime
from anatomical_covariance import cortical_regional_means, anatomical_covariance_matrix
from os import listdir
from neuroHarmonize import harmonizationLearn, harmonizationApply
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

# set data type
data_type = 'sMRI_fMRI_subjects'

# excldue site 16?
exclude_site_16 = False

# save weights?
weights = True

# anatomical covariance matrix parameters: atlas choice and number of histogram bins
atlas = 'Cambridge'
n_bins = 2
hemispheres = 'both'

if data_type == 'fMRI' or data_type == 'sMRI_fMRI_subjects' :

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
        outcomes_metadata_data = outcomes_metadata_data.iloc[:, :16]
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
    
    # filter out rows where panss fpr baseline or prediction timepoint is missing/nan
    timepoint_panss_cols = list(filter(lambda x: timepoint + '|panss' in x, outcomes.columns))
    baseline_panss_cols = list(filter(lambda x: 'Baseline|panss' in x, outcomes.columns))
    outcomes = outcomes[outcomes[timepoint_panss_cols].notna().any(axis=1)]
    outcomes = outcomes[outcomes[baseline_panss_cols].notna().any(axis=1)]
    
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
tril_inds = np.ravel_multi_index(np.tril_indices(n_regions, k=-1), (n_regions, n_regions))
n_connectivities = len(tril_inds)
logm_connectivity_data = np.zeros((n_subjects, n_connectivities))

# do site correction
if data_type == 'fMRI' and (site_correction == 'comBat_unsupervised' or site_correction == 'comBat_supervised' or site_correction == 'None') :

    # take matrix logs at start
    for i in range(n_subjects) :
    
        print(i)
    
        connectivity_vector = data[i, :]
        connectivity_matrix = np.reshape(connectivity_vector, (n_regions, n_regions))
        logm_connectivity_matrix = logm(connectivity_matrix)
        logm_connectivity_vector = np.reshape(logm_connectivity_matrix, (n_regions * n_regions, ))
        logm_connectivity_data[i, :] = logm_connectivity_vector[tril_inds]
        
if data_type == 'sMRI' or data_type == 'sMRI_fMRI_subjects' :
    
    logm_connectivity_data = data[:, tril_inds]

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
        logm_connectivity_data[i, :] = mglm_residuals_vector[tril_inds]

# set up MCCV
n_repeats = 200
test_fraction = 0.2
train_inds_all, test_inds_all, test_size = ShuffleSplitFixed(logm_connectivity_data, site, site, test_fraction, n_repeats)
train_size = n_subjects - test_size
#ss = StratifiedShuffleSplit(n_splits = n_repeats, test_size = test_size,    )

# data structures to hold results
predicted_followup_total_PANSS = np.zeros((n_repeats * test_size, 1))
test_baseline_total_PANSS = np.zeros_like(predicted_followup_total_PANSS)
test_followup_total_PANSS = np.zeros_like(predicted_followup_total_PANSS)

if weights :
    
    weights = np.zeros((n_subjects))

# pull out targets (followup total PANSS), baseline total PANSS and subject ids
baseline_total_PANSS = metadata['Baseline|total_panss'].values
followup_total_PANSS = metadata[timepoint + '|total_panss'].values
subjectids = metadata['Subject ID'].to_list()

# initialise list of test subjects
test_subjects = []

# allocate memory for weights
weights = np.zeros((n_repeats, n_connectivities))

# initialise regressor
rgr = Ridge()

# do MCCV
for i in range(n_repeats) :
    
    train_index = train_inds_all[i]
    test_index = test_inds_all[i]
    
    train_data = logm_connectivity_data[train_index, :]
    test_data = logm_connectivity_data[test_index, :]

    print (i)    
    
    # calculate output indices
    start_ind = i * test_size
    stop_ind = start_ind + test_size
    
    train_targets = followup_total_PANSS[train_index]
    test_targets = followup_total_PANSS[test_index]
    test_subjects = test_subjects + list(np.array(subjectids)[test_index])
    
    # do supervised site correction?
    if site_correction == 'comBat_supervised' :
        
        # correct training data
        train_data, model = fit_transform_neuroCombat(train_data, metadata.iloc[train_index, :], 'Site', continuous_cols=[timepoint + '|total_panss'])
        
        # correct testing dara
        test_data = apply_neuroCombat_model(test_data,
                      metadata.iloc[test_index, :], model,
                      'Site')
        
    
    # train rge for followup PANSS and predict
    rgr.fit(train_data, train_targets)
    preds = rgr.predict(test_data)
    weights[i, :] = rgr.coef_
    
    # store results
    predicted_followup_total_PANSS[start_ind:stop_ind, 0] = preds
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
results_file = open(results_dir + 'predict_treatment_response_results_' + data_type + '_' + site_correction + '_' + date_string + '.txt', "w") 

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

# save the weights file
np.savetxt(results_dir + 'predict_treatment_response_weights_' + data_type + '_' + site_correction + '_'+ date_string + '.csv', weights, delimiter=',')
