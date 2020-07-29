#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 17:11:15 2020

@author: k1511004
"""

import pandas as pd
import numpy as np
from statsmodels.stats.power import tt_ind_solve_power

# import the predictions
preds = pd.read_csv('/home/k1511004/Data/PSYSCAN/WP5_data/FEP/full_cohort_results/LOO_results_comBat_unsupervised.csv')

# allocate memory for smaple sizes
sample_sizes = np.zeros(20,)

# calculate standard deviation of change in PANSS
baseline_PANSS = preds['baseline total PANSS'].values
followup_PANSS = preds['followup total PANSS'].values
true_PANSS_ratio = followup_PANSS / baseline_PANSS
std = np.std(true_PANSS_ratio)
n_subjects = len(baseline_PANSS)

# extract predicted PANSS ratio
pred_PANSS_ratio = preds['followup total PANSS'].values

# loop through thresholds
for i in range(20) :
    
    # number of top subjects to take
    top_percentile = 5 * (i+1)
    top_n = int ((n_subjects * top_percentile) / 100)
    print (top_percentile)
    print (top_n)
    
    # find mean TRUE PANSS ratio of top n
    mean_nonresponders = np.mean(true_PANSS_ratio[:top_n])
    print (mean_nonresponders)
    
    # sample size calculation: set parameters
    # mean of non-treatment/placebo arm is mean_nonresponders
    # mean of treatment arm is 0.8 (defines responders)
    effect_size = np.abs(mean_nonresponders - 0.8) / std
    print (effect_size)
    
    # do the calculation
    # set alpha = 0.05, power = 80% as is standard and ratio to 1
    foo = tt_ind_solve_power(effect_size=effect_size, nobs1=None, alpha=0.05, power=0.8, ratio=1.0, alternative='two-sided')
    print (foo)