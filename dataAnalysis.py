import pandas as pd
import plotly.graph_objects as go
import config
import numpy as np



def saveResultAsHTML(targetpath,episode,para):
    df = pd.read_csv(targetpath)

    df = df[para.seq_length-1:-para.RewardDay-2]

    fig_asset = go.Figure()

    fig_asset.add_trace(go.Scatter(x=df.index, y=df['asset'],
                                mode='lines',
                                name='Asset',
                                line=dict(color='blue', width=2)))

    fig_asset.update_layout(
        title='Asset Over Time',
        xaxis_title='Date',
        yaxis_title='Asset Value',
        xaxis_rangeslider_visible=False 
    )

    fig_asset.write_html(f'{para.log_dir}/asset_{episode}.html')

    fig_close_price = go.Figure()

    fig_close_price.add_trace(go.Scatter(x=df.index, y=df['收盘'],
                                        mode='lines',
                                        name='Close Price',
                                        line=dict(color='green', width=2)))

    fig_close_price.update_layout(
        title='Close Price Over Time',
        xaxis_title='Date',
        yaxis_title='Close Price',
        xaxis_rangeslider_visible=False  
    )

    fig_close_price.write_html(f'{para.log_dir}/close_price_{episode}.html')


    
    
    
    
def furtherAnalysis(targetPath, episode,para):

    df = pd.read_csv(targetPath)

    df = df[para.seq_length-1:-para.RewardDay+1]
    def update_action(action):
        if 5 <= action <= 8:
            return -(action - 4)  
        else:
            return action 

    df['updated_action'] = df['action'].apply(update_action)

    fig = go.Figure(data=[go.Candlestick(x=df.index,
                                        open=df['开盘'],
                                        high=df['最高'],
                                        low=df['最低'],
                                        close=df['收盘'],
                                        name='Candlesticks')])

    close_trace = []

    for i, row in df.iterrows():
        if row['updated_action'] == 0:
            color = 'blue'
        elif 1 <= row['updated_action'] <= 4:
            color = 'green'
        elif -4 <= row['updated_action'] < 0:
            color = 'red'
        else:
            color = 'gray'  

        hover_text = (
                    f"reward: {row['reward']}<br>"
                    f"Stock Owned: {row['stock_owned']}<br>"
                    f"Balance: {row['balance']}<br>"
                    f"Asset: {row['asset']}<br>"
                    f"action:{row['updated_action']}")

        close_trace.append(go.Scatter(x=[i], y=[row['收盘']],
                                    mode='markers',
                                    marker=dict(color=color, size=8),
                                    text=[hover_text],  
                                    hoverinfo='text',  
                                    name='Close Action'))


    for trace in close_trace:
        fig.add_trace(trace)

    fig.update_layout(
        title='Stock Price K-Line with Updated Action, Close Price and Asset Information',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,  
        legend_title="Agent Actions",
        legend=dict(x=0.01, y=0.99)  
    )
    fig.write_html(f'{para.log_dir}/action_{episode}.html')

    

    
    
    
    
def winRate(df):
    df['do'] = 0
    df['shouldDo'] = 0
    df.loc[(df['futureAvgPrice'] >= df['收盘']) , 'shouldDo'] = 1
    df.loc[(df['收盘'] > df['futureAvgPrice']) , 'shouldDo'] = 2
    
    df.loc[(df['action'].between(0, 4)), 'do'] = 1
    df.loc[(df['action'].between(5, 8)), 'do'] = 2
    buySum = df['do'].eq(1).sum()
    sellSum = df['do'].eq(2).sum()  
    
    shouldBuy = df['shouldDo'].eq(1).sum()
    shouldSell = df['shouldDo'].eq(2).sum()    
    buyRight = ((df['do'] == 1) & (df['shouldDo'] == 1)).sum()
    sellRight = ((df['do'] == 2) & (df['shouldDo'] == 2)).sum()
    right_rate =  round((buyRight + sellRight)/(shouldBuy+shouldSell)*100,3)
    # print("buy right num:",buyRight)
    # print("buy action sum:",buySum)
    # print("should buy num:",shouldBuy)
    # print("sell rihgt num:",sellRight)
    # print("sell action sum:",sellSum)
    # print("should sell num:",shouldSell)
    buyRight_rate = round(buyRight*100 / buySum,3) if buySum > 0 else 0
    sellRight_rate = round(sellRight*100 / sellSum,3) if sellSum > 0 else 0
    result = {
        "buyRightN": buyRight,
        "buyActionS": buySum,
        "shouldBuyN": shouldBuy,
        "sellRightN": sellRight,
        "sellActionS": sellSum,
        "shouldSellN": shouldSell,
        "winRate": right_rate,
        "buyWinRate": buyRight_rate,
        "sellWinRate": sellRight_rate
    }

    return result
    
    
    
    
def winRateAnalysis(targetpath,episode,para,n=100):
    df = pd.read_csv(targetpath)
    df = df[para.seq_length-1:-para.RewardDay-2]
    df['group'] = pd.cut(df.index, bins=n, labels=False)
    dfs = [group for _, group in df.groupby('group')]

    ret_df = None
    for i in range(n):
        ret = winRate(dfs[i])
        if ret_df is None:
            ret_df = pd.DataFrame([ret.values()],columns=ret.keys())
        else:
            newRow = pd.DataFrame([ret])
            ret_df = pd.concat([ret_df,newRow],ignore_index=True)
            
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=ret_df.index,
        y=ret_df['winRate'],
        mode='lines+markers',
        name='Win Rate',
        line=dict(color='blue'),
        marker=dict(size=8)
    ))

    fig.add_trace(go.Scatter(
        x=ret_df.index,
        y=ret_df['buyWinRate'],
        mode='lines+markers',
        name='Buy Win Rate',
        line=dict(color='rgba(0, 128, 0, 0.2)'),
        marker=dict(size=8),
        hovertemplate=(
            "<b>Buy Win Rate</b><br>" +
            "Rate: %{y:.3f}<br>" +
            "Buy Right Num: %{customdata[0]}<br>" +
            "Buy Action Sum: %{customdata[1]}<br>" +
            "Should Buy Num: %{customdata[2]}<br>" +
            "<extra></extra>"
        ),
        customdata=ret_df[['buyRightN', 'buyActionS', 'shouldBuyN']].values
    ))

    fig.add_trace(go.Scatter(
        x=ret_df.index,
        y=ret_df['sellWinRate'],
        mode='lines+markers',
        name='Sell Win Rate',
        line=dict(color='rgba(128, 0, 0, 0.2)'),
        marker=dict(size=8),
        hovertemplate=(
            "<b>Sell Win Rate</b><br>" +
            "Rate: %{y:.3f}<br>" +
            "Sell Right Num: %{customdata[0]}<br>" +
            "Sell Action Sum: %{customdata[1]}<br>" +
            "Should Sell Num: %{customdata[2]}<br>" +
            "<extra></extra>"
        ),
        customdata=ret_df[['sellRightN', 'sellActionS', 'shouldSellN']].values
    ))

    fig.update_layout(
        title='Win Rate Analysis',
        yaxis_title='Rate (%)',
        legend=dict(x=0, y=1.2, orientation='h'),
        template='plotly_white',  
        hovermode='x unified' 
    )
    fig.write_html(f'{para.log_dir}/win_rate_analysis_{episode}.html')
        

def winRateTest(df):
    df['do'] = 0
    df['shouldDo'] = 0
    df.loc[(df['futureAvgPrice'] >= df['收盘']) , 'shouldDo'] = 1
    df.loc[(df['收盘'] > df['futureAvgPrice']) , 'shouldDo'] = 2
    
    df.loc[(df['action'].between(0, 10)), 'do'] = 1
    df.loc[(df['action'].between(11, 20)), 'do'] = 2
    buySum = df['do'].eq(1).sum()
    sellSum = df['do'].eq(2).sum()  
    
    shouldBuy = df['shouldDo'].eq(1).sum()
    shouldSell = df['shouldDo'].eq(2).sum()    
    buyRight = ((df['do'] == 1) & (df['shouldDo'] == 1)).sum()
    sellRight = ((df['do'] == 2) & (df['shouldDo'] == 2)).sum()
    right_rate =  round((buyRight + sellRight)/(shouldBuy+shouldSell)*100,3)
    buyRight_rate = round(buyRight*100 / buySum,3) if buySum > 0 else 0
    sellRight_rate = round(sellRight*100 / sellSum,3) if sellSum > 0 else 0
    result = {
        "buyRightN": buyRight,
        "buyActionS": buySum,
        "shouldBuyN": shouldBuy,
        "sellRightN": sellRight,
        "sellActionS": sellSum,
        "shouldSellN": shouldSell,
        "winRate": right_rate,
        "buyWinRate": buyRight_rate,
        "sellWinRate": sellRight_rate
    }
    return result

def winRateAnalysisTest(targetpath,episode,para,n=100):
    df = pd.read_csv(targetpath)
    df = df[para.seq_length-1:-para.RewardDay-2]
    df['group'] = pd.cut(df.index, bins=n, labels=False)
    dfs = [group for _, group in df.groupby('group')]

    ret_df = None
    for i in range(n):
        ret = winRateTest(dfs[i])
        if ret_df is None:
            ret_df = pd.DataFrame([ret.values()],columns=ret.keys())
        else:
            newRow = pd.DataFrame([ret])
            ret_df = pd.concat([ret_df,newRow],ignore_index=True)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=ret_df.index,
        y=ret_df['winRate'],
        mode='lines+markers',
        name='Win Rate',
        line=dict(color='blue'),
        marker=dict(size=8)
    ))

    fig.add_trace(go.Scatter(
        x=ret_df.index,
        y=ret_df['buyWinRate'],
        mode='lines+markers',
        name='Buy Win Rate',
        line=dict(color='rgba(0, 128, 0, 0.2)'),
        marker=dict(size=8),
        hovertemplate=(
            "<b>Buy Win Rate</b><br>" +
            "Rate: %{y:.3f}<br>" +
            "Buy Right Num: %{customdata[0]}<br>" +
            "Buy Action Sum: %{customdata[1]}<br>" +
            "Should Buy Num: %{customdata[2]}<br>" +
            "<extra></extra>"
        ),
        customdata=ret_df[['buyRightN', 'buyActionS', 'shouldBuyN']].values
    ))

    fig.add_trace(go.Scatter(
        x=ret_df.index,
        y=ret_df['sellWinRate'],
        mode='lines+markers',
        name='Sell Win Rate',
        line=dict(color='rgba(128, 0, 0, 0.2)'),
        marker=dict(size=8),
        hovertemplate=(
            "<b>Sell Win Rate</b><br>" +
            "Rate: %{y:.3f}<br>" +
            "Sell Right Num: %{customdata[0]}<br>" +
            "Sell Action Sum: %{customdata[1]}<br>" +
            "Should Sell Num: %{customdata[2]}<br>" +
            "<extra></extra>"
        ),
        customdata=ret_df[['sellRightN', 'sellActionS', 'shouldSellN']].values
    ))

    fig.update_layout(
        title='Win Rate Analysis',
        yaxis_title='Rate (%)',
        legend=dict(x=0, y=1.2, orientation='h'),
        template='plotly_white',  
        hovermode='x unified' 
    )

    fig.write_html(f'./win_rate_analysis_{episode}.html')

if __name__ == "__main__": 
    import os
    para = config.Param()
    env_name = para.envName
    log_dir = log_dir = os.path.join("log", env_name)
    
    episode = para.episodeNum-1
    csvPath = os.path.join(log_dir, f"{env_name}_Eval_episode{episode}.csv")
    df = pd.read_csv(csvPath)
    df['futureAvgPrice'] = df['收盘'].shift(-2).rolling(window=2).mean()
    df.to_csv(csvPath)
    saveResultAsHTML(csvPath,episode,para)
    furtherAnalysis(csvPath,episode,para)
    winRateAnalysisTest(csvPath,999,para)
    
    
    
    pass
    