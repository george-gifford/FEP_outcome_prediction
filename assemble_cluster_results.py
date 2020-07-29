#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 12:19:13 2020

@author: k1511004
"""

import pandas as pd
from glob import glob
from sklearn.metrics import mean_absolute_error, roc_auc_score, balanced_accuracy_score
from scipy.stats import pearsonr
from datetime import datetime
import numpy as np

# set directories
results_dir = '/home/k1511004/Data/PSYSCAN/WP5_data/FEP/full_cohort_results/'
cluster_results_dir = results_dir + 'CV_results/'

# set result parameters
data_type = 'fMRI'
site_correction = 'comBat_unsupervised'
network_size = 'medium'

# make base filename and find results files
base_filename = 'predict_treatment_response_results_brainnet_CNN_MSE_' + data_type + '_' + network_size + '_' + site_correction 
results_files = glob(cluster_results_dir + base_filename + '*')


# exit if results files not found
if len(results_files) == 0 :
    
    print ('results files not found!')
    exit

base_filename = base_filename + '_' + str(len(results_files))

# read in and concatenate the results files in a DF
result_dfs = []
for results_file in results_files:
    result_dfs.append(pd.read_csv(results_file))

# Concatenate all data into one DataFrame and remove index
subjects_response_results = pd.concat(result_dfs, ignore_index=True)
subjects_response_results = subjects_response_results.iloc[:, 1:]
subjects_response_results.rename(columns = {'0':'subjectid'}, inplace = True) 

# put results in a DF with subjects
pred_PANSS = subjects_response_results['predicted followup total PANSS'].values
true_PANSS = subjects_response_results['followup total PANSS'].values
PANSS_followup_ratio = subjects_response_results['total PANSS followup/baseline ratio'].values
mae = mean_absolute_error(pred_PANSS, true_PANSS)
r2 = pearsonr(pred_PANSS, true_PANSS)
std = np.std(true_PANSS)

# open file for storing results
date_string = datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")
results_file = open(results_dir + base_filename + '_' + date_string + '.txt', "w") 

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
site = subjects_response_results['Site'].values
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
