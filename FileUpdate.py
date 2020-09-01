import sys
import requests
import pandas as pd
from pandas.io.json import json_normalize
import json

def loadData(symbol):
    URL= "https://cloud.iexapis.com/stable/stock/"+symbol+"/previous?token=pk_f21b1474e546423f9a39275cb33eed97"
    r = requests.get(url = URL);
 
    data1 = r.json() 
    df=pd.json_normalize(data1);
    print(df);
    with open("C:\\Users\\aalex\\Stocks_Project\\historic_data\\"+symbol+".csv",'a') as fd:
       
        line='';
        #line =line+"index"+','+ str( df['date'].iloc[0])+','+str(df['open'].iloc[0])+','+str(df['close'].iloc[0])+','+str(df['high'].iloc[0])+','+str(df['low'].iloc[0])+','+str(df['volume'].iloc[0])+','+str(df['uOpen'].iloc[0])+','+str(df['uClose'].iloc[0])+','+str(df['uHigh'].iloc[0])+','+str(df['uLow'].iloc[0])+','+str(df['uVolume'].iloc[0])+','+df['change'].head().to_string(index=False)+','+str(df['changePercent'].iloc[0])+','+'---'+','+str(df['changeOverTime'].iloc[0])+'\n'
        line1 =line+"index"+','+ str( df['date'].iloc[0])+','+str(df['uClose'].iloc[0])+','+str(df['uOpen'].iloc[0])+','+str(df['uHigh'].iloc[0])+','+str(df['uLow'].iloc[0])+','+str(df['uVolume'].iloc[0])+','+str(df['close'].iloc[0])+','+str(df['open'].iloc[0])+','+str(df['high'].iloc[0])+','+str(df['low'].iloc[0])+','+str(df['volume'].iloc[0])+','+' '+','+df['change'].head().to_string(index=False)+','+str(df['changePercent'].iloc[0])+','+'---'+','+str(df['changeOverTime'].iloc[0])+'\n'
       
        fd.write(line1)

    df1=pd.read_csv("C:\\Users\\aalex\\Stocks_Project\\historic_data\\"+symbol+".csv")
    df1.drop(df1.columns.values[0], axis=1, inplace=True)
    fd.close();
   



if __name__ == "__main__":
    arguments= sys.argv;
    loadData(sys.argv[1]);
    n=sys.argv[1];
    print(n)


 