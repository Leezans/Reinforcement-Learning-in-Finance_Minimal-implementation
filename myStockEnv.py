import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
import config
param = config.Param()




class SingleStockDayEnv:
    def __init__(self, param:config.Param):
        self.csv_path = os.path.join(param.trainDatasetPath, f"{param.code}.csv") if param.trainMode else os.path.join(param.testDatasetPath, f"{param.code}.csv")
        self.seq_length = param.seq_length
        self.date = param.seq_length

        self.df = self.read_file()
        self.initial_balance = param.initial_balance
        self.initial_stockOwned = param.initial_stock_owned
        self.balance = param.initial_balance
        self.stock_owned = param.initial_stock_owned
        self.asset = self.balance + self.stock_owned* self.get_price(self.date - 1)[2] # self.get_price(self.date - 1)[2]是最高价
        self.transaction_fee = param.transaction_fee                                   # self.get_price(self.date - 1)[1]是收盘价
        self.endDate = self.df.shape[0] - 1  
        self.observation_space = self.df.iloc[0].to_numpy()
        self.action_space = np.arange(21) 
        self.fixed_quantity = param.fixed_quantity

    def reset(self):
        self.date = self.seq_length
        self.balance = self.initial_balance
        self.asset = self.balance + self.stock_owned* self.get_price(self.date - 1)[2] # 这里是用了最高价做了初始资产计算，不过不影响后续计算
        self.stock_owned = self.initial_stockOwned
        state = self.get_state(self.date)
        info = {"asset": self.asset, "balance": self.balance, "stock_owned": self.stock_owned}
        return state, info

    def step(self, action: int):
        assert 0 <= action < 21, "Invalid action. Action must be in range [0, 20]."
        
        current_price = self.get_price(self.date - 1)[1]
        temp_asset = self.asset
        punishment = 0

        if self.fixed_quantity:
            fixed_quantity = 10000 
            if action == 0: 
                pass

            elif 1 <= action <= 10:  
                buy_percentage = action * 0.1  
                max_buyable = self.balance // (current_price * (1 + self.transaction_fee))  
                buy_amount = int(fixed_quantity * buy_percentage)  

                actual_buy_amount = min(max_buyable, buy_amount)

                if actual_buy_amount > 0:  
                    self.stock_owned += actual_buy_amount
                    self.balance -= current_price * actual_buy_amount * (1 + self.transaction_fee)
                else:
                    punishment = 100 

            elif 11 <= action <= 20:
                sell_percentage = (action - 10) * 0.1
                max_sellable = self.stock_owned  
                sell_amount = int(fixed_quantity * sell_percentage) 

                actual_sell_amount = min(max_sellable, sell_amount)

                if actual_sell_amount > 0: 
                    self.stock_owned -= actual_sell_amount
                    self.balance += current_price * actual_sell_amount * (1 - self.transaction_fee)
                else:
                    punishment = 100 
        else:
            if action == 0:  
                pass
            elif 1 <= action <= 10:  
                buy_percentage = action * 0.1 
                max_buyable = self.balance // (current_price * (1 + self.transaction_fee))
                buy_amount = int(max_buyable * buy_percentage)
                if buy_amount > 0:
                    self.stock_owned += buy_amount
                    self.balance -= current_price * buy_amount * (1 + self.transaction_fee)
                else:
                    punishment = 100  

            elif 11 <= action <= 20:  
                sell_percentage = (action - 10) * 0.1 
                sell_amount = int(self.stock_owned * sell_percentage)
                if sell_amount > 0:
                    self.stock_owned -= sell_amount
                    self.balance += current_price * sell_amount * (1 - self.transaction_fee)
                else:
                    punishment = 500 


        self.asset = self.balance + current_price * self.stock_owned
        reward = self.asset - temp_asset - punishment
        self.date += 1

        terminal = self.date == self.endDate
        truncated = False 

        info = {"asset": self.asset, "balance": self.balance, "stock_owned": self.stock_owned, "current_price": current_price,}

        next_state = self.get_state(self.date) if not terminal else np.zeros_like(self.get_state(self.date))
        return next_state, reward, terminal, truncated, info

    def get_state(self, n):
        state = self.df.iloc[n-self.seq_length:n].copy()
        scaler_minmax = MinMaxScaler()
        state[['成交额']] = scaler_minmax.fit_transform(state[['成交额']])+0.1
        state[['成交量']] = scaler_minmax.fit_transform(state[['成交量']])+0.1
        state[['收盘']] = scaler_minmax.fit_transform(state[['收盘']])+0.1
        state[['开盘']] = scaler_minmax.fit_transform(state[['开盘']])+0.1
        state[['最高']] = scaler_minmax.fit_transform(state[['最高']])+0.1
        state[['最低']] = scaler_minmax.fit_transform(state[['最低']])+0.1
        state = state.to_numpy()
        return state
    
    def get_price(self, n):
        return self.df.iloc[n].to_numpy()

    def read_file(self):
        df = pd.read_csv(self.csv_path)
        df = df.drop(columns=["日期", "股票代码"])
        return df


if __name__ == "__main__":
    env = SingleStockDayEnv(param)
    state, info = env.reset()
    print("Initial State:", state,"\n")
    print("Initial Info:", info,"\n")

    for _ in range(10):  
        action = np.random.choice(np.arange(21)) 
        next_state, reward, terminal, truncated, info = env.step(action)
        print(f"Action: {action},\n Next State: {next_state}, \nReward: {reward}, \nTerminal: {terminal}, \nInfo: {info}\n")
        if terminal or truncated:
            break
        
        
    pass
