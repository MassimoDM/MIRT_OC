#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 12:42:52 2017

@author: Massimo De Mauri
"""
import MIRT_OC as oc


class constraint:

    def __init__(self,expression,lob = -oc.inf, upb = 0, typ = None, lin = None, lbl=''):

        if expression.numel() >1:
            raise NameError('Multidimensional constraints currently not allowed')


        self._exp = expression
        self.lob = lob
        self.upb = upb
        self.typ = typ
        self.lin = lin
        self.lbl = lbl


        if typ is None or lin is None:
            self.classify()


    @property
    def exp(self):
        return self._exp

    @exp.setter
    def exp(self,exp):
        self._exp = exp
        self.classify()



    def __str__(self):

        if self.ord == 0:
            out = 'Constant constraint?  :   '
        elif self.ord == 1:
            out = 'Linear constraint     :   '
        elif self.ord == 2:
            out = 'Quadratic constraint  :   '
        else:
            out = 'Non-linear constraint :   '


        if self.lob == self.upb:
            out += str(self._exp) + ' = ' + str(float(self.lob))
        else:
            if not self.lob == -oc.inf:
                out += str(float(self.lob)) + ' <= '

            out += str(self._exp)

            if self.upb != oc.inf:
                out += ' <= ' + str(float(self.upb))


        return out


    def standardize(self):

        if  self.lob == self.upb:
            out = [constraint(self._exp - self.lob,0,0,'equality',self.ord,self.lbl)]
        else:
            out = []
            if self.lob != -oc.inf:
                out += [constraint(self.lob - self._exp,-oc.inf,0,'inequality',self.ord,self.lbl)]

            if self.upb != oc.inf:
                out += [constraint(self._exp - self.upb,-oc.inf,0,'inequality',self.ord,self.lbl)]

        return out



    def classify(self,variable_list = []):

        if self.lob != self.upb:
             self.typ = 'inequality'
        else:
            self.typ = 'equality'

        if variable_list == []: variable_list = oc.symvar(self._exp)
        
        self.ord = oc.amax(oc.get_order(self._exp,variable_list))
