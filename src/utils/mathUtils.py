#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 11:35:37 2017

@author: Massimo De Mauri
"""

from numpy import *
import casadi as cs


def cartesianProduct(sets):
    if len(sets) == 0:
        return []
    elif len(sets) == 1:
        return [[i] for i in sets[0]] 
    s1 = cartesianProduct(sets[1:])
    return [j+[i] for j in s1 for i in sets[0]]
    




def dm_argmax(matrix,dim=None):
    if matrix.numel() == 0:
        return (cs.DM(),None)
  
    elif dim is None:
        
            
        if matrix.nnz() == 0:
            return (cs.DM(1,1),(0,0))
        
        indxs = matrix.sparsity().get_triplet()
        cnd_i = [-1]*len(indxs)
        for d,i in enumerate(indxs):
            cnd_i[d] = i[0] 
        
        max_i = tuple(cnd_i)
        
        for k in range(1,len(indxs[0])):
            for d,i in enumerate(indxs):
                cnd_i[d] = i[k]     
            
            tmp = tuple(cnd_i)
            if matrix[max_i] < matrix[tmp]:
                max_i = tmp
                
        return (matrix[max_i],max_i)
    
    
    elif dim is 0:
        
        
        if matrix.nnz() == 0:
            return (cs.DM(1,matrix.shape[1]),[0]*matrix.shape[1])
        
        
        max_v = cs.DM.inf(1,matrix.shape[1])
        max_i = [-1]*matrix.shape[1]
        
        for k in range(max_v.shape[1]):
            max_v[k], tmp = dm_argmax(matrix[:,k])
            max_i[k] = tmp[0]
        return (max_v,max_i)
    
    elif dim is 1:
        
        if matrix.nnz() == 0:
            return (cs.DM(matrix.shape[0],1),[0]*matrix.shape[0])
        
        max_v = cs.DM.inf(matrix.shape[0],1)
        max_i = [-1]*matrix.shape[0]
        
        for k in range(max_v.shape[0]):
            max_v[k], tmp = dm_argmax(matrix[k,:])
            max_i[k] = tmp[1]
        return (max_v,max_i)
                
    
    
        
    
def dm_max(matrix,dim=None):
    return dm_argmax(matrix,dim)[0]


def dm_argmin(matrix,dim=None):
    tmp = dm_argmax(-matrix,dim)
    return (-tmp[0],tmp[1])

def dm_min(matrix,dim=None):
    return -dm_max(-matrix,dim)



def dm_round(num,decimals=0):
    return floor(num*10**decimals + 0.5)*10**-decimals

     
    
def dependency_dfs(sparsity,row_list):
    
    num_nz = len(sparsity[0])
    row_set = set(row_list)
    cols_to_check = set([sparsity[1][i] for i in range(num_nz) if sparsity[0][i] in row_set])
    col_set = cols_to_check
    while True:
        rows_to_check = set([sparsity[0][i] for i in range(num_nz) if sparsity[1][i] in cols_to_check]).difference(row_set)
        cols_to_check = set([sparsity[1][i] for i in range(num_nz) if sparsity[0][i] in rows_to_check]).difference(col_set)
        row_set = row_set.union(rows_to_check)
        if len(cols_to_check) == 0:
            return(col_set)
        
    return 0




def dm_quicksort(x,I=None,reverse = False):
    
    if reverse:
        return dm_quicksort_bw(x,I)
    else:
        return dm_quicksort_fw(x,I)
    
    
    
    
def dm_quicksort_fw(x,I=None):
    
    if I is None: 
        I = list(range(x.numel()))
    else:
        assert len(I) == x.numel()
        
        
    x = cs.reshape(x,-1,1)
    
    
    
    if x.numel() <= 1:
        return [x,I]
    else:
        
        pivot = x[0]
        i = 0
        for j in range(x.numel()-1):
            
            if x[j+1] < pivot:
                I[j+1],I[i+1] = I[i+1], I[j+1]
                x[j+1],x[i+1] = x[i+1], x[j+1]
                i += 1
        I[0],I[i] = I[i],I[0]       
        x[0],x[i] = x[i],x[0]
        
        
        [x_fp, I_fp] = dm_quicksort_fw(x[:i],I[:i])
        [x_sp, I_sp] = dm_quicksort_fw(x[i+1:],I[i+1:])
        
        
        return [cs.vertcat(x_fp,x[i],x_sp), I_fp + [I[i]] + I_sp]
        
    
def dm_quicksort_bw(x,I=None):
    
    if I is None: 
        I = list(range(x.numel()))
    else:
        assert len(I) == x.numel()
        
        
    x = cs.reshape(x,-1,1)
    
    
    
    if x.numel() <= 1:
        return [x,I]
    else:
        
        pivot = x[0]
        i = 0
        for j in range(x.numel()-1):
            
            if x[j+1] > pivot:
                I[j+1],I[i+1] = I[i+1], I[j+1]
                x[j+1],x[i+1] = x[i+1], x[j+1]
                i += 1
        I[0],I[i] = I[i],I[0]       
        x[0],x[i] = x[i],x[0]
        
        
        [x_fp, I_fp] = dm_quicksort_bw(x[:i],I[:i])
        [x_sp, I_sp] = dm_quicksort_bw(x[i+1:],I[i+1:])
        
        
        return [cs.vertcat(x_fp,x[i],x_sp), I_fp + [I[i]] + I_sp]