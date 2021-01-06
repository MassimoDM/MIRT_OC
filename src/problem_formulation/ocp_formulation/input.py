#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 13:25:32 2017

@author: Massimo De Mauri
"""
from MIRT_OC import MX, DM, is_numeric, is_symbolic

class input:

    def __init__(self,ref = None ,val = None):

        if ref is None:
            self.nme = None
            self.sym = None
        elif type(ref) == str:
            self.nme = ref
            self.sym = MX.sym(ref)
        else:
            self.nme = ref.name
            self.sym = ref

        self.val = val

    def __str__(self):

        if is_numeric(self.val):
            if self.val.numel() < 10:
                return self.nme + ' = ' + str(self.val)
            else:
                return self.nme + ' = ' + str(self.val[:4])[:-1] + ', ... ,' + str(self.val[-4:])[1:]
        elif is_symbolic(self.val):
            return self.nme + ' = ' + self.val.name()
        else:
            print('ooh')
