import pandas as pd
import matplotlib.pyplot as plt

def upd_df(df):
    #df = pd.read_csv(f'C:\Stock Price Prediction\df_{ticker}.csv')
    Added_changes = []
    for i in range(len(df)):
      Added_changes.append(df.Close_actual[0] + df.Close_prediction_change[1] + df.Close_prediction_change[1:i].sum())

    df['Added_changes'] = Added_changes
    df['Added_changes'] = df['Added_changes'].shift(-1)
    return df

def plot_results(ticker, df, change):
    plt.figure(figsize=(12, 6))
    plt.plot(df.Close_actual_change, color='green', label='Real Price')
    plt.plot(df.Close_prediction_change, color='purple', label='Predicted Price')
    plt.title(f'{ticker}')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig(f'C:/Users/AI_BootCamp_06/Desktop/LSTMM/plot_{ticker}_daily.png')

    # plt.show()

    if change == 'absolute':
        plt.figure(figsize=(12, 6))
        plt.plot(pd.concat([df['Close_actual'], df['Added_changes']], axis=1))
        plt.title('Close Absolute Change Prediction (only adding changes)')
        plt.savefig(f'C:/Users/AI_BootCamp_06/Desktop/LSTMM/absolute_change_{ticker}.png')
        plt.close()

    else:
        pass

def plot_loss(my_model, ticker):
    plt.figure(figsize=(10, 6))
    plt.plot(my_model.history_['loss'])
    plt.savefig(f'C:/Users/AI_BootCamp_06/Desktop/LSTMM/loss_plot/plot_loss_{ticker}.png')


# for stock in ['NFLX', 'MSFT', 'V', 'AMZN', 'TWTR', 'AAPL', 'GOOG', 'TSLA', 'FB', 'NVDA', 'JNJ', 'UNH', 'XOM', 'JPM', 'PG', 'CVX', 'MA', 'WMT', 'HD', 'PFE', 'BAC', 'LLY', 'KO', 'ABBV']:
#     df1 = upd_df(stock)
#     plot_results(stock, df1, change='absolute')