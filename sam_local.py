# -*- coding: utf-8 -*-
"""
Created on Sun May 17 14:43:35 2020

@author: sbcur
"""
import pandas as pd
import os
os.chdir("C:/Users/Sam/Documents/GitHub/QBUS6840")

raw_df = pd.read_csv("C:/Users/Sam/Documents/GitHub/QBUS6840/UnemploymentRateJan1986-Dec2018.csv")

from pandas_profiling import ProfileReport
prof = ProfileReport(raw_df)
