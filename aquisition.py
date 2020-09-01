#Get data from API: 

#INPUT:  'symbol'
#OUTPUT: ---
import requests
import sys
import pandas as pd
from pandas.io.json import json_normalize
import json

def getData(symbol):
    
    token="pk_f21b1474e546423f9a39275cb33eed97"
    URL= "https://cloud.iexapis.com/v1/stock/"+symbol+"/chart/max?token="+token;
    #request the data from provider
    r = requests.get(url = URL);
   
    data = r.json() 
    #put the data in a dataframe
    df=pd.json_normalize(data);

    #make sure the columns in the following order
    dffile=df[[	'date',	'uClose','uOpen','uHigh','uLow','uVolume','close','open','high','low','volume','currency','change','changePercent','label','changeOverTime']].copy()
    #save data frame to the   correspoding cvs file
    dffile.to_csv('C:\\Users\\aalex\\Stocks_Project\\historic_data\\'+symbol+'.csv'); 
    print('C:\\\\Users\\\\aalex\\\\Stocks_Project\\\\historic_data\\\\'+symbol+'.csv')
   

if __name__ == "__main__":
    arguments= sys.argv;
    getData(sys.argv[1]);

    
    
    