# -*- coding: utf-8 -*-
"""
Created on Thu May  2 14:52:32 2019

@author: Danne
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import normalize
#from graphviz import Digraph -> maybe better pictures
#https://networkx.github.io/documentation/networkx-2.2/auto_examples/drawing/plot_directed.html

def dag_generic(data, patient_data, t = 't2'):
    '''returns a graph with probabilities for predicted time element (t) and certainties for specific patient.''' 
    
    list_edges = [
                  ('Tumor Size', 'Stage'), 
                  ('Lymph Nodes','Stage'), 
                  ('Mets', 'Stage'), 
                  ]

    mapping = {'Age' : 'Age at diagnosis','Subtype':'Histologic Type ICD-O-3', 'Tumor Size':'CS Tumor Size','Lymph Nodes':'CS Lymph Nodes' ,'Mets' : 'CS Mets at Dx','Stage':'Derived AJCC Stage Group','Survival months':'Survival months','Surgery':'RX Summ—Surg Prim Site'}
 
    if t == 't2':
        
        data_known = ['Age','Subtype', 'Tumor Size','Lymph Nodes','Mets','Stage']
        prob_data_known = calc_prob_data_known(data_known, mapping, data, patient_data)
        
        #calculating probability of surgery
        surgery_prob = prob_of_surgery(data)
        
        #putting nodes together
        nodes_sm = []
        for surgery in surgery_prob.keys():
            sm = 'SM_' + str(np.round(surgery_prob[surgery][1], decimals = 2))
            nodes_sm.append(sm) #assumes uniques of means.. could break
            
        nodes = data_known + list(surgery_prob.keys()) + nodes_sm #!!make sure to keep this sequence!!
        #size nodes based on probabilty
        size_surgery_nodes = []
        for surgery in surgery_prob.keys():
            size_surgery_nodes.append(surgery_prob[surgery][0])

        size_nodes = prob_data_known + size_surgery_nodes + size_surgery_nodes
        
        
        #putting edges together
        for surgery in range(len(surgery_prob.keys())):
            list_edges.append(('Tumor Size', list(surgery_prob.keys())[surgery]))
            list_edges.append(('Lymph Nodes', list(surgery_prob.keys())[surgery]))
            list_edges.append(('Stage', list(surgery_prob.keys())[surgery]))
            list_edges.append(('Age', list(surgery_prob.keys())[surgery]))
            list_edges.append(('Subtype', list(surgery_prob.keys())[surgery]))
            list_edges.append((list(surgery_prob.keys())[surgery], nodes_sm[surgery])) 
        
        #uncertainty of edge based on std if available. For higher interpretability scale and invert. 
        edge_uncertainty = []      
        for edge in list_edges:
            x = True            
            for surgery in surgery_prob.keys():
                if surgery == edge[0]:
                    edge_uncertainty.append(surgery_prob[surgery][2])
                    x = False
            if x:
                edge_uncertainty.append(0)
        
          
        edge_uncertainty = normalize(np.array(edge_uncertainty).reshape(1, -1))
        edge_uncertainty = [1.0 - i for i in edge_uncertainty[0]]
        
        #creating positions
        positions = {'Age':(2,11),
         'Subtype':(2,13),
         'Tumor Size':(1,1),
         'Lymph Nodes':(1,3),
         'Mets':(1,5),
         'Stage':(2,9)}
        
        x = 3.5
        for surgery in surgery_prob.keys(): #use set since SM can overlap
            x += 1
            positions[surgery] = (3, x)
        x = 3.5
        for sm in nodes_sm:
            x += 1
            positions[sm] = (4, x) 
            
        #declare fig title
        title = 'generic DAG for T1 showing the likelihood of T2 values'
        
    elif t == 't3':
        data_known = ['Age','Subtype', 'Stage','Surgery']
        prob_data_known = calc_prob_data_known(data_known, mapping, data, patient_data)

        #putting nodes together
        surgery_data = list(data['RX Summ—Surg Prim Site'])
        surgery = patient_data['RX Summ—Surg Prim Site']
        sm_calc_m, sm_calc_s = survival_mean_std(data.loc[data['RX Summ—Surg Prim Site'] == surgery]['Survival months'])
        sm_prevalence = surgery_data.count(surgery)/len(surgery_data)
        sm = 'SM_' + str(np.round(sm_calc_m, decimals = 2))
            
        nodes = data_known + [sm] #!!make sure to keep this sequence!!

        #size nodes based on probabilty
        size_nodes = prob_data_known + [sm_prevalence]
        
        #putting edges together
        list_edges = [('Stage', sm),
                  ('Age', sm),
                  ('Subtype', sm), ('Surgery', sm)]
        
        #uncertainty of edge based on std if available. For higher interpretability scale and invert. 
        edge_uncertainty = []      
        for edge in list_edges:
            if edge == ('Surgery', sm): 
                edge_uncertainty.append(sm_calc_s)
            else:
                edge_uncertainty.append(0)
        edge_uncertainty = [1 - (i/100) for i in edge_uncertainty]

        #creating positions
        positions = {'Age':(2,11),
         'Subtype':(2,13),
         'Stage':(2,9), 'Surgery':(3,6), sm: (4,6)}
        
        #declare fig title
        title = 'generic DAG for T2 showing the likelihood of T3 values'
        
    visualize_graph(nodes, list_edges, size_nodes, edge_uncertainty, positions, title)
    
def prob_of_surgery(data):
    '''returns probability of each surgery in a dictionary order by [prob of surgery, mean sm, std sm]'''
    
    surgery_data = list(data['RX Summ—Surg Prim Site'])
    probability_surgery = {}
    for surgery in set(surgery_data):
        probability_surgery[surgery] = [surgery_data.count(surgery)/len(surgery_data)] + survival_mean_std(data.loc[data['RX Summ—Surg Prim Site'] == surgery]['Survival months'])
    
    return probability_surgery

def survival_mean_std(surgery_survival_data):
    '''returns mean and std for each surgery in dict'''
    return [np.mean(list(surgery_survival_data)), np.std(list(surgery_survival_data))]

def calc_prob_data_known(data_known, mapping, data, patient_data):
    '''return probablities in list'''
    l =[]
    for d in data_known:
        l.append(len(data.loc[data[mapping[d]] == patient_data[mapping[d]]])/len(data))
    return l


def visualize_graph(nodes, edges, size_nodes, edge_uncertainty, positions, title, div =[50,600]):
    '''visualizes graph'''
    #do crazy sorting thing because graph sorts nodes and edges alphabetically
    nodes_info = zip(nodes, size_nodes)
    nodes_sorted = sorted(nodes_info)
    nodes, size_nodes = zip(*nodes_sorted)
    
    edges_info = zip(edges, edge_uncertainty)
    edges_sorted = sorted(edges_info)
    edges, edge_uncertainty = zip(*edges_sorted)
    
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    pos = positions
    pos_labels = {}
    #put the labels above the nodes
    for p in pos.keys():
        pos_labels[p] = (pos[p][0] , pos[p][1]+ 0.35)
    
    node_sizes = [div[0] + div[1] * i for i in size_nodes]
    M = G.number_of_edges()
    edge_colors = edge_uncertainty
    edge_alphas = [i/2 for i in edge_uncertainty]
    
    plt.figure(figsize=(11, 9))  
    
    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='deepskyblue')
    edges = nx.draw_networkx_edges(G, pos, node_size=node_sizes, arrowstyle='->',
                                   arrowsize=20, width=2, edge_color=edge_colors,edge_cmap=plt.cm.Blues, edge_vmin = min(edge_colors) - 0.2) 
                                   
    labels = nx.draw_networkx_labels(G, pos_labels, font_size = 9, font_weight = 'bold')
    # set alpha value for each edge
    for i in range(M):
        edges[i].set_alpha(edge_alphas[i])
    
    pc = mpl.collections.PatchCollection(edges, cmap=plt.cm.Blues)
    pc.set_array(edge_colors)
    plt.colorbar(pc)
    

    ax = plt.gca()
    ax.set_axis_off()
    plt.title(title)
    plt.savefig('dag_generic.png', bbox = 'tight')
    plt.show()
    

