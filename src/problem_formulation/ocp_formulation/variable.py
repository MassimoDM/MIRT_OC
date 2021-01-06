#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 12:40:29 2017

@author: Massimo De Mauri
"""
from casadi import *
from numpy import inf, amin, amax
class variable:
    def __init__(self,ref ='v',lob=-inf,upb=inf,val=None,dsc=False,lbl = ''):

        if ref is None:
            self.nme = None
            self.sym = None
        elif type(ref) == str:
            self.nme = ref
            self.sym = MX.sym(ref)
        else:
            self.nme = ref.name
            self.sym = ref


        self.lob = lob
        self.upb = upb
        self.lbl = lbl

        if val is None:
            self.val = (upb + lob)/2
            if lob > -inf and upb < inf:
               self.val = (upb + lob)/2
            elif lob > -inf:
                self.val = lob
            elif upb < inf:
                self.val = upb
            else:
                self.val = 0


        else:
            self.val = val




        if type(dsc) == type(1):
            dsc = dsc == 1
        assert type(dsc) == type(True)

        self.dsc = dsc

    def __str__(self):


        if self.dsc:
            out = str(self.lob) + " <= [" + self.nme + "] <= " + str(self.upb)
        else:
            out = str(self.lob) + " <= " + self.nme + " <= " + str(self.upb)


        if DM(self.val).numel() > 1:
            out += '    (val = ' + str(DM(self.val).numel()) + ' elements in [' + str(amin(DM(self.val))) + ',' +str(amax(DM(self.val))) + '])'
        else:
            out += '    (val = ' + str(self.val) + ')'

        return out

    def copy(self):
        return variable(self.nme,self.lob,self.val,self.dsc)

    def deepcopy(self):
        return variable(self.nme,self.lob,self.val,self.dsc)
