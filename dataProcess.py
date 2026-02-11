import akshare as ak
import os
import pandas as pd


from config import Param

param = Param()

def get_save_data(code,cover=True):
    filePath = os.path.join(param.stockDataPath,f"{code}.csv")
    if cover:
        data = ak.stock_zh_a_hist(symbol=code,period="daily",start_date="20000101",end_date="20250718",adjust='hfq')
        
        if data is not None:
            print(data)

            data.to_csv(filePath, index=False)
            print(f"Data saved to {filePath}")
    elif os.path.exists(filePath):
        print(f"File {filePath} already exists. Skipping save.")
    else:
        data = ak.stock_zh_a_hist(symbol=code,period="daily",start_date="20000101",end_date="20250718")
        if data is not None:
            data.to_csv(filePath, index=False)
            print(f"Data saved to {filePath}")

def process_data(code):
    filePath = os.path.join(param.stockDataPath,f"{code}.csv")
    if os.path.exists(filePath):
        data = pd.read_csv(filePath)
        print(data)
        train_data = data[:int(0.8*len(data))]
        test_data = data[int(0.8*len(data)):]
        train_data.to_csv(filePath.replace("stockData","trainDataset"), index=False)
        test_data.to_csv(filePath.replace("stockData","testDataset"), index=False)
    else:
        print(f"File {filePath} does not exist.")
            
if __name__ == "__main__":

    get_save_data(param.code)
    process_data(param.code)



    pass

