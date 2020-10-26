# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 12:32:18 2020

@author: Dell
"""

import numpy as np 
import pandas as pd
import pickle
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import datetime
from tqdm import tqdm
from statistics import mean 
import math


    
testdf=pd.read_csv("../../data/test_data.csv")
traindf= pd.read_csv("../../data/train_data.csv")
valdf=pd.read_csv("../../data/val_data.csv")


def calcVolatility(retValues):
    
    try:
        avg=mean(retValues)
        variance = sum([((x - avg) ** 2) for x in retValues]) / (len(retValues)) 
        stddev = variance ** 0.5
        volatility=np.log(stddev)
    except:
        volatility='NA'
    
    return volatility


days=[3,7,15,30]
error=[]
error_date=[]

count=0

for file in tqdm(os.listdir('../../data/AllRetPrices/')):
    # print(file)
    compRetdf=pd.read_csv("../../data/AllRetPrices/"+file)
    for day in days:
        # print("Day="+str(day))
        
        volatility_list=[]
        for index,row in compRetdf.iterrows():
            anchor_date_ind=index
            try:
                tempPriordays=list(compRetdf.iloc[anchor_date_ind+1:anchor_date_ind+day+1]['ReturnPrice'])
                vol=calcVolatility(tempPriordays)
            except:
                vol='NA'
            volatility_list.append(vol)
            
        vol_col_name='Past_Volatility_{}'.format(day)
        # print(vol_col_name)
        compRetdf[vol_col_name] = volatility_list

    if not os.path.exists('../../data/AllVolatilities/'):
        os.mkdir('../../data/AllVolatilities/')
    save_path='../../data/AllVolatilities/'+file
    
    compRetdf.to_csv(save_path,index=False)
    

def GenerateRegresFeatures(df,day):

    X=[]  
    for index,row in tqdm(df.iterrows()):
        ticker=df.iloc[index]['ticker']
        
        call_date = datetime.datetime(df.iloc[index]['year'], df.iloc[index]['month'], df.iloc[index]['day'])
        
        vol_filename='../../data/AllVolatilities/'+ticker+'.csv'
        
        vol_df=pd.read_csv(vol_filename)
        
        vol_df['Date'] = vol_df['Date'].map(lambda date: datetime.datetime.strptime(date, '%Y-%m-%d'))
        
        vol_col_name='Past_Volatility_{}'.format(day)
        
        
        temp=vol_df.loc[vol_df['Date'] == call_date]
        anchor_date_ind=temp.index[0]
        
        vol_col=vol_df[vol_col_name]
        
        last_ind= vol_col.last_valid_index() 
        
        maxind=min(last_ind,anchor_date_ind+30+1)
        
        X_temp=list(vol_df.iloc[anchor_date_ind+1:anchor_date_ind+30+1][vol_col_name])
        
        avg=np.nanmean(X_temp)
        
        X_temp = [avg if math.isnan(x) else x for x in X_temp]
        
        
        X.append(X_temp)
    
    X=np.array(X)
    return X

    
#3 Days

X_train3days=GenerateRegresFeatures(traindf,3)
X_test3days=GenerateRegresFeatures(testdf,3)
X_val3days=GenerateRegresFeatures(valdf,3)


#7 Days
X_train7days=GenerateRegresFeatures(traindf,7)
X_test7days=GenerateRegresFeatures(testdf,7)
X_val7days=GenerateRegresFeatures(valdf,7)

X_train15days=GenerateRegresFeatures(traindf,15)
X_test15days=GenerateRegresFeatures(testdf,15)
X_val15days=GenerateRegresFeatures(valdf,15)

X_train30days=GenerateRegresFeatures(traindf,30)
X_test30days=GenerateRegresFeatures(testdf,30)
X_val30days=GenerateRegresFeatures(valdf,30)

#3
pickle.dump( X_train3days, open( "../../data/financial_features/X_train3days.pkl", "wb" ) )
pickle.dump( X_test3days, open( "../../data/financial_features/X_test3days.pkl", "wb" ) )
pickle.dump( X_val3days, open( "../../data/financial_features/X_val3days.pkl", "wb" ) )

#7
pickle.dump( X_train7days, open( "../../data/financial_features/X_train7days.pkl", "wb" ) )
pickle.dump( X_test7days, open( "../../data/financial_features/X_test7days.pkl", "wb" ) )
pickle.dump( X_val7days, open( "../../data/financial_features/X_val7days.pkl", "wb" ) )

#15
pickle.dump( X_train15days, open( "../../data/financial_features/X_train15days.pkl", "wb" ) )
pickle.dump( X_test15days, open( "../../data/financial_features/X_test15days.pkl", "wb" ) )
pickle.dump( X_val15days, open( "../../data/financial_features/X_val15days.pkl", "wb" ) )

#30
pickle.dump( X_train30days, open( "../../data/financial_features/X_train30days.pkl", "wb" ) )
pickle.dump( X_test30days, open( "../../data/financial_features/X_test30days.pkl", "wb" ) )
pickle.dump( X_val30days, open( "../../data/financial_features/X_val30days.pkl", "wb" ) )

