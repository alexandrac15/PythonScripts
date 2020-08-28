import sys

import pandas as pd
from pandas.io.json import json_normalize
import json

import datetime



def loadData(path, n):
    df1=pd.read_csv(path)
    df1.drop(df1.columns.values[0], axis=1, inplace=True)
    df2=df1[['date','close']].copy()
   # print(df2)
    df3=df2.tail(n)
    pd.options.mode.chained_assignment = None    #WHY THE WARINGIN APPEARS? SLICING VS COPY????????? 
  
    #attempt to get rid of the warining , by introducing intermidiary steps;
    ggg=pd.to_datetime(df3.date).copy();
    rrr=ggg.dt.strftime('%m/%d/%Y').copy();
    df3['date']=rrr.copy();

   # pd.to_datetime(df3['date'].astype(str), format='%m/%d/%Y')
    print(df3.to_string(header=None,index=False))





if __name__ == "__main__":
    arguments= sys.argv;
    loadData(sys.argv[1],int(sys.argv[2]));
#def loadData():
#   #incarca din cvs in dataframe 
#    df1=pd.read_csv("C:\\Users\\aalex\\Stocks Project\\historic_data\\FB.csv")
#    df1.drop(df1.columns.values[0], axis=1, inplace=True)
#    print(df1.head())
   