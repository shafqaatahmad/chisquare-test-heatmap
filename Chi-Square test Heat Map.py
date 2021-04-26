# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 19:57:55 2021

@author: shafqaat.ahmad
"""

from pandas import Series, DataFrame
import pandas as pd
import numpy as np
from collections import Counter
import os 
from datetime import date
from sklearn.feature_selection import chi2
from scipy import stats
import seaborn as sns
import matplotlib.pylab as plt
from numpy import percentile
from sklearn.feature_selection import SelectKBest


path="E:\Python WD"
os.chdir(path) 
studentdf = pd.read_csv("xAPI-Edu-Data.csv",low_memory='False') # Loading the file

######### Chi Square testing Matrix ###########
## Testing association bewteen indepnedent variables######


column_names=studentdf.columns

chisqmatrix=pd.DataFrame(studentdf,columns=column_names,index=column_names)

outercnt=0
innercnt=0
for icol in column_names:
    
    for jcol in column_names:
        
       mycrosstab=pd.crosstab(studentdf[icol],studentdf[jcol])
       #print (mycrosstab)
       stat,p,dof,expected=stats.chi2_contingency(mycrosstab)
       chisqmatrix.iloc[outercnt,innercnt]=round(p,3)
       cntexpected=expected[expected<5].size
       perexpected=((expected.size-cntexpected)/expected.size)*100
      
       #print (icol)
       #print (jcol)
       if perexpected<20:
            chisqmatrix.iloc[outercnt,innercnt]=2
       #print (perexpected) 
       if icol==jcol:
           chisqmatrix.iloc[outercnt,innercnt]=0.00
       #print (expected) 
       innercnt=innercnt+1
    #print (outercnt) 
    outercnt=outercnt+1
    innercnt=0
    


sns.heatmap(chisqmatrix.astype(np.float64), annot=True,linewidths=0.1, 
            cmap='coolwarm')





