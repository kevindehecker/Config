# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 23:29:02 2018

@author: guido
"""

import pickle
import evaluation_utils

DVE_info = pickle.load( open( "./pkl_rawerror/DVE_info_2.pkl", "rb" ) )
evaluation_utils.plot_dve_info(DVE_info);