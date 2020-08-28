import sys
import requests
import pandas as pd
from pandas.io.json import json_normalize
import json

def loadData(symbol):
    URL= "https://sandbox.iexapis.com/stable/stock/"+symbol+"/chart/5d?token=Tpk_fb4e7bfffded4a93b5ef66eed6aeac2a"
    r = requests.get(url = URL);
    print (r);
    data1 = r.json() 
    df=pd.json_normalize(data1);
    df.to_csv('C:\\Users\\aalex\\Desktop\\'+symbol+'.csv'); 
    print(df);
   
   



if __name__ == "__main__":
    arguments= sys.argv;
    loadData(sys.argv[1]);
   
   


 
