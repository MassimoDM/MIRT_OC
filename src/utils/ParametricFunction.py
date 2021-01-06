# @Author: Massimo De Mauri <massimo>
# @Date:   2021-01-04T12:35:39+01:00
# @Email:  massimo.demauri@protonmail.com
# @Filename: ParametricFunction.py
# @Last modified by:   massimo
# @Last modified time: 2021-01-04T12:35:42+01:00
# @License: LGPL-3.0

# Define a parametric NLP solver class
class ParametricFunction:
    def __init__(self,function,par_values):
        self.function = function
        self.par_values = par_values

    def call(self,input_data_list):
        return self.function.call({'expar':self.par_values,**input_data_list})
