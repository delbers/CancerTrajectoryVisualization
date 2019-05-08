# -*- coding: utf-8 -*-
"""
Created on Thu May  2 09:39:35 2019

@author: Danne
"""

import pandas as pd
import data_wrangling as dw
import similarity_mapping as sm
import directed_acyclic_graph as dag
import dag_linearSVC as dagl

from sklearn.model_selection import train_test_split

#read data file
data = pd.read_csv('data/seer_data_cleaned.csv')

#clean data
data_t1, data_t2, data_t3, data_scaled_t1, data_scaled_t2, data_scaled_t3 = dw.clean(data)

data_list = [data_t1, data_t2, data_t3, data_scaled_t1, data_scaled_t2, data_scaled_t3]

#split in test (5%) and training (95%), only use indices! 
train, test = train_test_split(data, test_size=0.05)
train_index = train.index
test_index = test.index

#run clustering models
model_t1, labels_t1, score_t1 = sm.fit_model(data_scaled_t1.loc[train_index])
model_t2, labels_t2, score_t2 = sm.fit_model(data_scaled_t2.loc[train_index])

#get test case
test_case = test.index[0]

#run t1 for test case
#define patient cluster
patient_data_all = data_t3.loc[test_case]
patient_data = data_scaled_t1.loc[test_case]
patient_fit = sm.fit_patient(patient_data, model_t1)
#run generic dag
data_cluster = sm.get_cluster(patient_fit, labels_t1, data_t3.loc[train_index]) # use full dataset (t3) with t1 labels
#dag.dag_generic(data_cluster, patient_data_all, t = 't2')
#run LinearSVC dag
dagl.dag_linearSVC(data_cluster, patient_data_all, t = 't2')

#run t2 for test case
#define patient cluster
patient_data = data_scaled_t2.loc[test_case]
patient_fit = sm.fit_patient(patient_data, model_t2)
data_cluster = sm.get_cluster(patient_fit, labels_t2, data_t3.loc[train_index]) # use full dataset (t3) with t1 labels
#run generic dag
dag.dag_generic(data_cluster, patient_data_all, t = 't3')
