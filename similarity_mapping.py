# -*- coding: utf-8 -*-
"""
Created on Thu May  2 10:32:22 2019

@author: Danne
"""
import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift
from sklearn.metrics import  silhouette_score#, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns


def fit_model(data):
    '''returns the model, labels for the data and performance score of fit'''
    clustering = MeanShift().fit(data)
    labels = clustering.predict(data)
    
    sil_score = silhouette_score(data, labels)
    #dav_score = davies_bouldin_score(data, labels)   
    
    mean_shift = pd.DataFrame(labels)
    data.insert((data.shape[1]),'Mean Shift', mean_shift)
    
    mapping = {'Age' : 'Age at diagnosis','Subtype':'Histologic Type ICD-O-3', 'Tumor Size':'CS Tumor Size','Lymph Nodes':'CS Lymph Nodes' ,'Mets' : 'CS Mets at Dx','Stage':'Derived AJCC Stage Group','Survival months':'Survival months','Surgery':'RX Summ—Surg Prim Site'}  
    plot_col = ['Age at diagnosis','Derived AJCC Stage Group', 'Mean Shift', 'CS Mets at Dx', 'CS Lymph Nodes', 'CS Tumor Size', 'Histologic Type ICD-O-3']
    use_col = ['Age at diagnosis','Derived AJCC Stage Group', 'CS Mets at Dx', 'CS Lymph Nodes', 'CS Tumor Size', 'Histologic Type ICD-O-3']
    pp = sns.pairplot(data[plot_col], vars = use_col, hue='Mean Shift', size=1.8, aspect=1.8, 
                      plot_kws=dict(edgecolor="black", linewidth=0.5, alpha = 0.3))
    fig = pp.fig 
    fig.autofmt_xdate()
    fig.subplots_adjust(top=0.93, wspace=0.3)
    t = fig.suptitle('Mean Shift Clustering Using High Importance Columns', fontsize=14)

    return clustering, labels, sil_score #, dav_score

def fit_patient(patient, clustering):
    '''returns cluster patient belongs to'''
    patient_fit = clustering.predict(np.array(patient).reshape(1, -1))    
    return int(patient_fit)

def get_cluster(patient_label, labels, data):
    '''returns data set with all patients in the same cluster as the 'new' input patient'''
    data['labels'] = labels
    return data.loc[data['labels'] == patient_label]
    


