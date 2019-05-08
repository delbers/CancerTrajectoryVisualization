# -*- coding: utf-8 -*-
"""
Created on Sat May  4 16:27:29 2019

@author: Danne
"""
import pandas as pd
import numpy as np

#from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import LinearSVC

import directed_acyclic_graph as dag


def dag_linearSVC(data, patient_data, t = 't2'):
    '''returns graph model using outcomes of lasso linear regression'''
    
    if t == 't2':
        
        #design model
        
        features = ['Age at diagnosis', 'Histologic Type ICD-O-3', 'CS Tumor Size', 'CS Lymph Nodes', 'CS Mets at Dx', 'Derived AJCC Stage Group']
        train, test = train_test_split(data, test_size=0.1)
        train_features = train[features]
        train_obs = list(train['RX Summ—Surg Prim Site'])
        test_features = test[features]
        test_obs = list(test['RX Summ—Surg Prim Site'])
        
        #found out through Countvectorizer that features are sorted alphabetically. 
        clf = LinearSVC(random_state=0, tol=1e-5)
        clf.fit(train_features, train_obs)
        clf.predict(test_features)
        score = clf.score(test_features, test_obs)
        print(score)

        #create dictionary with weights for features
        weights = clf.coef_
        names = sorted(set(data['RX Summ—Surg Prim Site']))
        weight_names = {}
        for n in range(len(names)):
            weight_names[names[n]] = weights[n].tolist()
        
        patient_fit = clf.predict(np.array(patient_data[features]).reshape(1, -1))
        
        #create DAG
        mapping = {'Age' : 'Age at diagnosis','Subtype':'Histologic Type ICD-O-3', 'Tumor Size':'CS Tumor Size','Lymph Nodes':'CS Lymph Nodes' ,'Mets' : 'CS Mets at Dx','Stage':'Derived AJCC Stage Group','Survival months':'Survival months','Surgery':'RX Summ—Surg Prim Site'}      
        data_known = ['Age','Subtype', 'Tumor Size','Lymph Nodes','Mets','Stage']
        
        #putting nodes together
        surgery_data = list(data['RX Summ—Surg Prim Site'])
        surgery = patient_fit[0]
        sm_calc_m, sm_calc_s = dag.survival_mean_std(data.loc[data['RX Summ—Surg Prim Site'] == surgery]['Survival months'])
        sm_prevalence = surgery_data.count(surgery)/len(surgery_data)
        sm = 'SM_' + str(np.round(sm_calc_m, decimals = 2))
            
        nodes = data_known + [surgery] + [sm] #!!make sure to keep this sequence!!
        
        #size nodes based on Coefficients
        print(weight_names[surgery], score)
        size_nodes = weight_names[surgery]
        size_nodes.append(score)
        size_nodes.append(sm_prevalence)
              
        #putting edges together
        list_edges = [('Tumor Size', 'Stage'), 
                      ('Lymph Nodes','Stage'), 
                      ('Mets', 'Stage'),(surgery, sm), ('Tumor Size', surgery), 
                      ('Lymph Nodes', surgery), ('Stage', surgery), ('Age', surgery)
                      , ('Subtype', surgery)]

        #uncertainty of edge based on std if available. For higher interpretability scale and invert. 
        edge_uncertainty = []      
        for edge in list_edges:
            if edge[1] == surgery and edge[0] != sm:
                edge_uncertainty.append(1-score)
            else:
                edge_uncertainty.append(0)

        edge_uncertainty = normalize(np.array(edge_uncertainty).reshape(1, -1))
        edge_uncertainty = [1.0 - i for i in edge_uncertainty[0]]
        
        #creating positions
        positions = {'Age':(2,11),
         'Subtype':(2,13),
         'Tumor Size':(1,1),
         'Lymph Nodes':(1,3),
         'Mets':(1,5),
         'Stage':(2,9), surgery:(3,6), sm: (4,6)}
        
        #declare fig title
        title = 'LinearSVC DAG for T1 showing the prediction for T2 surgery.'
        
    dag.visualize_graph(nodes, list_edges, size_nodes, edge_uncertainty, positions, title)