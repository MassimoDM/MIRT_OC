#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 14:12:12 2017

@author: Massimo De Mauri
"""
import MIRT_OC as oc

class sos1_constraint:
    def __init__(self,group,weights=None):
        self.group = group
        self.weights = weights

        
    def __str__(self):
        out = 'SOS1: group = ' + str(self.group)
        
        if not self.weights is None:
            out += ' weights = ' + str(self.weights)
        return out
    
    def copy(self):
        return sos1_constraint(self.group,self.weights)
    
    def deepcopy(self):
        return sos1_constraint(self.group,self.weights)