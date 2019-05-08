# -*- coding: utf-8 -*-
"""
Created on Thu May  2 10:05:29 2019

@author: Danne
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def clean(data):
    '''fixes data issues in regards to weird values & generates six datasets one for each timepoint for both cleaned and cleaned, but scaled data.'''
    data.replace({'Survival months': {9999: np.nan}},inplace=True)
    mean_sm = np.mean(data['Survival months'])
    data.replace({'Survival months': {np.nan: mean_sm}},inplace=True)
    data.replace({'RX Summ—Surg Prim Site' : surgery_codes(detailed = 'short')}, inplace = True)
    
    data_scaled = scale(data)
    
    t1, t2, t3 = column_selection()    
    return data.loc[:,t1], data.loc[:,t2], data.loc[:,t3], data_scaled.loc[:,t1], data_scaled.loc[:,t2], data_scaled.loc[:,t3]

def scale(data):
    '''returns normalized dataframe'''    
    
    #change object columns to categorical integers
    obj_columns = data.select_dtypes(['object']).columns
    data_cat = pd.DataFrame()
    data_cat = data.copy(deep = True) #keep index the same for easy remapping
    data_cat[obj_columns] = data[obj_columns].apply(lambda x: x.astype("category").cat.codes)
    #scaling data for more appropriate assesment 
    scaler = StandardScaler()
    scaler.fit_transform(data_cat)
    
    return data_cat 

def surgery_codes(detailed = 'N'):
    ''' returns dictionary with surgery codes for 'RX Summ—Surg Prim Site', if detailed == 'Y' then the more detailed dictionary is returned, otherwise the generic.'''
    if detailed == 'Y':
    
        return {0 : 'None; no surgery of primary site',
         12: 'Laser ablation or cryosurgery',
         13: 'Electrocautery; fulguration',
         15: 'Local tumor destruction, NOS',
         19: 'Local tumor destruction or excision, NOS',
         20: 'Excision or resection of less than one lobe, NOS',
         21: 'Wedge resection',
         22: 'Segmental resection, including lingulectomy',
         23: 'Excision, NOS',
         24: 'Laser excision',
         25: 'Bronchial sleeve resection ONLY',
         30: 'Resection of [at least one] lobe or bilobectomy, but less than the whole lung (partial pneumonectomy, NOS)',
         33: 'Lobectomy WITH mediastinal lymph node dissection',
         45: 'Lobe or bilobectomy extended, NOS',
         46: 'WITH chest wall',
         47: 'WITH pericardium',
         48: 'WITH diaphragm',
         55: 'Pneumonectomy, NOS',
         56: 'WITH mediastinal lymph node dissection (radical pneumonectomy) ',
         65: 'Extended pneumonectomy',
         66: 'Extended pneumonectomy plus pleura or diaphragm',
         70: 'Extended radical pneumonectomy',
         80: 'Resection of lung, NOS ',
         90: 'Surgery, NOS',
         99: 'Unknown if surgery performed'}
        
    elif detailed == 'N':
        return {0 : 'None; no surgery of primary site',
         12: 'Local tumor destruction, NOS',
         13: 'Local tumor destruction, NOS',
         15: 'Local tumor destruction, NOS',
         19: 'Local tumor destruction, NOS',
         20: 'Excision or resection of less than one lobe, NOS',
         21: 'Excision or resection of less than one lobe, NOS',
         22: 'Excision or resection of less than one lobe, NOS',
         23: 'Excision or resection of less than one lobe, NOS',
         24: 'Excision or resection of less than one lobe, NOS',
         25: 'Excision or resection of less than one lobe, NOS',
         30: 'Resection of [at least one] lobe or bilobectomy, but less than the whole lung (partial pneumonectomy, NOS)',
         33: 'Resection of [at least one] lobe or bilobectomy, but less than the whole lung (partial pneumonectomy, NOS)',
         45: 'Lobe or bilobectomy extended, NOS',
         46: 'Lobe or bilobectomy extended, NOS',
         47: 'Lobe or bilobectomy extended, NOS',
         48: 'Lobe or bilobectomy extended, NOS',
         55: 'Pneumonectomy, NOS',
         56: 'Pneumonectomy, NOS',
         65: 'Extended pneumonectomy',
         66: 'Extended pneumonectomy',
         70: 'Extended radical pneumonectomy',
         80: 'Resection of lung, NOS ',
         90: 'Surgery, NOS',
         99: 'Unknown if surgery performed'}
        
    elif detailed == 'short':
        return {0 : 'No surgery',
         12: 'Local tumor destruction',
         13: 'Local tumor destruction',
         15: 'Local tumor destruction',
         19: 'Local tumor destruction',
         20: 'Resection <1 lobe',
         21: 'Resection <1 lobe',
         22: 'Resection <1 lobe',
         23: 'Resection <1 lobe',
         24: 'Resection <1 lobe',
         25: 'Resection <1 lobe',
         30: 'Resection of 1+ lobe',
         33: 'Resection of 1+ lobe',
         45: 'Lobe extended',
         46: 'Lobe extended',
         47: 'Lobe extended',
         48: 'Lobe extended',
         55: 'Pneumonectomy',
         56: 'Pneumonectomy',
         65: 'Extended pneumonectomy',
         66: 'Extended pneumonectomy',
         70: 'Extended radical pneumonectomy',
         80: 'Resection of lung ',
         90: 'Surgery',
         99: 'Unknown'}
        
    

def column_selection():
    '''return data columns that should be included in T1, T2 & T3'''
    
    analysis_col_t1 = ['Marital Status at DX', 'Race/Ethnicity',
     'NHIA Derived Hispanic Origin',
     'Sex',
     'Age at diagnosis',
     'Year of Birth',
     'Sequence Number—Central',
     'Month of diagnosis',
     'Year of diagnosis',
     'Primary Site',
     'Laterality',
     'Histology (92-00) ICD-O-2',
     'Behavior (92-00) ICD-O-2',
     'Histologic Type ICD-O-3',
     'Behavior Code ICD-O-3',
     'Grade',
     'Diagnostic Confirmation',
     'Type of Reporting Source',
     'Regional Nodes Positive',
     'Regional Nodes Examined',
     'CS Tumor Size',
     'CS Extension',
     'CS Lymph Nodes',
     'CS Mets at Dx',
     'CS Site-Specific Factor 1',
     'CS Site-Specific Factor 2',
     'CS Site-Specific Factor 25',
     'Derived AJCC T',
     'Derived AJCC N',
     'Derived AJCC M',
     'Derived AJCC Stage Group',
     'Derived SS1977',
     'Derived SS2000',
     'Derived AJCC—Flag',
     'CS Version Input Original',
     'CS Version Derived',
     'CS Version Input Current',
     'SEER Record Number',
     'SEER Type of Follow-up',
     'Age Recode <1 Year olds',
     'Site Recode ICD-O-3/WHO 2008',
     'Recode ICD-O-2 to 9',
     'Recode ICD-O-2 to 10',
     'ICCC site recode ICD-O-3/WHO 2008',
     'ICCC site rec extended ICD-O-3/WHO 2008',
     'Behavior Recode for Analysis',
     'Histology Recode—Broad Groupings',
     'Histology Recode—Brain Groupings',
     'CS Schema v0204+',
     'Race recode (White, Black, Other)',
     'Race recode (W, B, AI, API)',
     'Origin recode NHIA (Hispanic, Non- Hisp)',
     'SEER historic stage A',
     'First malignant primary indicator',
     'State-county recode',
     'COD to site rec KM',
     'Vital Status recode',
     'IHS Link',
     'Summary stage 2000 (1998+)',
     'AYA site recode/WHO 2008',
     'Lymphoma subtype recode/WHO 2008',
     'CS Tumor Size/Ext Eval',
     'CS Lymph Nodes Eval',
     'CS Mets Eval',
     'Primary by international rules',
     'ER Status Recode Breast Cancer (1990+)',
     'PR Status Recode Breast Cancer (1990+)',
     'CS Schema -AJCC 6th ed (previously called v1)',
     'Insurance recode (2007+)',
     'Derived AJCC-7 T',
     'Derived AJCC-7 N',
     'Derived AJCC-7 M',
     'Derived AJCC-7 Stage Grp',
     'Derived HER2 Recode (2010+)',
     'Breast Subtype (2010+)',
     'Lymphomas: Ann Arbor Staging (1983+)',
     'CS Mets at Dx-Bone',
     'CS Mets at Dx-Brain',
     'CS Mets at Dx-Liver',
     'CS Mets at Dx-Lung',
     'Total Number of In Situ/malignant\r\nTumors for Patient',
     'Total Number of Benign/Borderline\r\nTumors for Patient']
    
    analysis_col_t2 = ['Marital Status at DX', 'Race/Ethnicity',
     'NHIA Derived Hispanic Origin',
     'Sex',
     'Age at diagnosis',
     'Year of Birth',
     'Sequence Number—Central',
     'Month of diagnosis',
     'Year of diagnosis',
     'Primary Site',
     'Laterality',
     'Histology (92-00) ICD-O-2',
     'Behavior (92-00) ICD-O-2',
     'Histologic Type ICD-O-3',
     'Behavior Code ICD-O-3',
     'Grade',
     'Diagnostic Confirmation',
     'Type of Reporting Source',
     'Regional Nodes Positive',
     'Regional Nodes Examined',
     'CS Tumor Size',
     'CS Extension',
     'CS Lymph Nodes',
     'CS Mets at Dx',
     'CS Site-Specific Factor 1',
     'CS Site-Specific Factor 2',
     'CS Site-Specific Factor 25',
     'Derived AJCC T',
     'Derived AJCC N',
     'Derived AJCC M',
     'Derived AJCC Stage Group',
     'Derived SS1977',
     'Derived SS2000',
     'Derived AJCC—Flag',
     'CS Version Input Original',
     'CS Version Derived',
     'CS Version Input Current',
     'RX Summ—Surg Prim Site',
     'RX Summ—Scope Reg LN Sur',
     'RX Summ—Surg Oth Reg/Dis',
     'Reason for no surgery',
     'SEER Record Number',
     'SEER Type of Follow-up',
     'Age Recode <1 Year olds',
     'Site Recode ICD-O-3/WHO 2008',
     'Recode ICD-O-2 to 9',
     'Recode ICD-O-2 to 10',
     'ICCC site recode ICD-O-3/WHO 2008',
     'ICCC site rec extended ICD-O-3/WHO 2008',
     'Behavior Recode for Analysis',
     'Histology Recode—Broad Groupings',
     'Histology Recode—Brain Groupings',
     'CS Schema v0204+',
     'Race recode (White, Black, Other)',
     'Race recode (W, B, AI, API)',
     'Origin recode NHIA (Hispanic, Non- Hisp)',
     'SEER historic stage A',
     'First malignant primary indicator',
     'State-county recode',
     'COD to site rec KM',
     'Vital Status recode',
     'IHS Link',
     'Summary stage 2000 (1998+)',
     'AYA site recode/WHO 2008',
     'Lymphoma subtype recode/WHO 2008',
     'CS Tumor Size/Ext Eval',
     'CS Lymph Nodes Eval',
     'CS Mets Eval',
     'Primary by international rules',
     'ER Status Recode Breast Cancer (1990+)',
     'PR Status Recode Breast Cancer (1990+)',
     'CS Schema -AJCC 6th ed (previously called v1)',
     'Insurance recode (2007+)',
     'Derived AJCC-7 T',
     'Derived AJCC-7 N',
     'Derived AJCC-7 M',
     'Derived AJCC-7 Stage Grp',
     'Derived HER2 Recode (2010+)',
     'Breast Subtype (2010+)',
     'Lymphomas: Ann Arbor Staging (1983+)',
     'CS Mets at Dx-Bone',
     'CS Mets at Dx-Brain',
     'CS Mets at Dx-Liver',
     'CS Mets at Dx-Lung',
     'Total Number of In Situ/malignant\r\nTumors for Patient',
     'Total Number of Benign/Borderline\r\nTumors for Patient']
    
    analysis_col_t3 = ['Marital Status at DX', 'Race/Ethnicity',
     'NHIA Derived Hispanic Origin',
     'Sex',
     'Age at diagnosis',
     'Year of Birth',
     'Sequence Number—Central',
     'Month of diagnosis',
     'Year of diagnosis',
     'Primary Site',
     'Laterality',
     'Histology (92-00) ICD-O-2',
     'Behavior (92-00) ICD-O-2',
     'Histologic Type ICD-O-3',
     'Behavior Code ICD-O-3',
     'Grade',
     'Diagnostic Confirmation',
     'Type of Reporting Source',
     'Regional Nodes Positive',
     'Regional Nodes Examined',
     'CS Tumor Size',
     'CS Extension',
     'CS Lymph Nodes',
     'CS Mets at Dx',
     'CS Site-Specific Factor 1',
     'CS Site-Specific Factor 2',
     'CS Site-Specific Factor 25',
     'Derived AJCC T',
     'Derived AJCC N',
     'Derived AJCC M',
     'Derived AJCC Stage Group',
     'Derived SS1977',
     'Derived SS2000',
     'Derived AJCC—Flag',
     'CS Version Input Original',
     'CS Version Derived',
     'CS Version Input Current',
     'RX Summ—Surg Prim Site',
     'RX Summ—Scope Reg LN Sur',
     'RX Summ—Surg Oth Reg/Dis',
     'Reason for no surgery',
     'SEER Record Number',
     'SEER Type of Follow-up',
     'Age Recode <1 Year olds',
     'Site Recode ICD-O-3/WHO 2008',
     'Recode ICD-O-2 to 9',
     'Recode ICD-O-2 to 10',
     'ICCC site recode ICD-O-3/WHO 2008',
     'ICCC site rec extended ICD-O-3/WHO 2008',
     'Behavior Recode for Analysis',
     'Histology Recode—Broad Groupings',
     'Histology Recode—Brain Groupings',
     'CS Schema v0204+',
     'Race recode (White, Black, Other)',
     'Race recode (W, B, AI, API)',
     'Origin recode NHIA (Hispanic, Non- Hisp)',
     'SEER historic stage A',
     'First malignant primary indicator',
     'State-county recode',
     'Cause of Death to SEER site recode',
     'COD to site rec KM',
     'Vital Status recode',
     'IHS Link',
     'Summary stage 2000 (1998+)',
     'AYA site recode/WHO 2008',
     'Lymphoma subtype recode/WHO 2008',
     'SEER Cause-Specific Death\r\nClassification',
     'SEER Other Cause of Death Classification',
     'CS Tumor Size/Ext Eval',
     'CS Lymph Nodes Eval',
     'CS Mets Eval',
     'Primary by international rules',
     'ER Status Recode Breast Cancer (1990+)',
     'PR Status Recode Breast Cancer (1990+)',
     'CS Schema -AJCC 6th ed (previously called v1)',
     'Survival months',
     'Survival months flag',
     'Insurance recode (2007+)',
     'Derived AJCC-7 T',
     'Derived AJCC-7 N',
     'Derived AJCC-7 M',
     'Derived AJCC-7 Stage Grp',
     'Derived HER2 Recode (2010+)',
     'Breast Subtype (2010+)',
     'Lymphomas: Ann Arbor Staging (1983+)',
     'CS Mets at Dx-Bone',
     'CS Mets at Dx-Brain',
     'CS Mets at Dx-Liver',
     'CS Mets at Dx-Lung',
     'Total Number of In Situ/malignant\r\nTumors for Patient',
     'Total Number of Benign/Borderline\r\nTumors for Patient']
    
    return analysis_col_t1, analysis_col_t2, analysis_col_t3