from feature_engineering import *
from data_engineering import *
from modeling import *
from prediction import *
from visualisation import *
from helper_functions import *
import os

def final_pred(ticker, change='absolute'):
    data = download_data(ticker, '2018-01-01', '2022-01-01', '1d')
    data = all_indicators(data)
    data.dropna(inplace = True)
    data_tr = data_transform(data, change)
    X_train, y_train, X_test, y_test, scaler = data_split(data_tr.iloc[:, :-1], division='by date',
                                                          split_criteria='2021-01-01', scale='yes', step_size=30)
    grid_model = build_model(X_train, loss='mse', optimizer='adam')
    grid_model = KerasRegressor(build_fn=grid_model, verbose=1)

    my_model = best_model(X_train, y_train, grid_model, cv=2)
    # dataset_test = data.iloc[:, :-1].loc['2021-01-01':]
    # y_test_change = data_tr.loc['2021-01-01':]
    # y_test_change = np.array(y_test_change.iloc[30:,3])

    #y_test_close = np.array(data.loc['2021-01-01':, 'Close'][30:])
    y_test_close_change = np.array(data_tr.loc['2021-01-01':, 'Close_abs_change'][30:])

    preds, score = prediction(my_model, y_test_close_change, X_test, scaler, loss='mse') # y_test_close_change
    d = {'Close_actual_change': y_test_close_change, 'Close_prediction_change': preds} # y_test_close_change
    data_pred = pd.DataFrame(data=d, index=data[-len(preds):].index)
    df_preds, classification_accuracy = classification(data_pred, data, change=change)
    df_preds_abs = upd_df(df_preds)
    plot_results(ticker, df_preds_abs, change=change)

    return df_preds, df_preds_abs, classification_accuracy

# data = download_data('NFLX', '2018-01-01', '2022-01-01', '1d')
# data_tr = data_transform(data, 'absolute')
# X_train, y_train, X_test, y_test, scaler = data_split(data_tr.iloc[:, :-1], division = 'by date', split_criteria = '2021-01-01', scale = 'yes', step_size = 30)
# grid_model = build_model(X_train, loss = 'mse', optimizer = 'adam')
# grid_model = KerasRegressor(build_fn = grid_model,verbose=1)
#
#
# my_model = best_model(X_train, y_train, grid_model, cv = 2)
# # dataset_test = data.iloc[:, :-1].loc['2021-01-01':]
# # y_test_change = data_tr.loc['2021-01-01':]
# # y_test_change = np.array(y_test_change.iloc[30:,3])
#
# y_test_close = np.array(data.loc['2021-01-01':, 'Close'][30:])
# y_test_close_change = np.array(data_tr.loc['2021-01-01':, 'Close_abs_change'][30:])
#
# prediction, score = prediction(my_model, y_test_close_change, X_test, scaler, loss = 'mse')
# d = {'Close_actual_change': y_test_close_change, 'Close_prediction_change': prediction}
# data_pred = pd.DataFrame(data=d, index = data[-len(prediction):].index)
# df, classification_accuracy = classification(data_pred, data, change = 'absolute')
#
# # plot_results('NFLX', df, data, data_tr, prediction, change = 'absolute')


def makemydir(df, stock, folder_name ):
    dir = os.path.join("C:\\", folder_name)
    if not os.path.exists(dir):
        os.makedirs(dir)
    os.chdir(dir)
    df.to_csv(f'df_{stock}_change_TI.csv')

for stock in ['NFLX', 'MSFT', 'V', 'AMZN', 'TWTR', 'AAPL', 'GOOG', 'TSLA', 'FB', 'NVDA']: #, 'JNJ', 'UNH', 'XOM', 'JPM', 'PG', 'CVX', 'MA', 'WMT', 'HD', 'PFE', 'BAC', 'LLY', 'KO', 'ABBV']:
    df_preds, df_preds_abs, clf_acc = final_pred(stock, change='absolute')
    makemydir(df_preds, stock, "Stock Price Prediction with TI (Close change only)")
    makemydir(df_preds_abs, stock, "Stock Price Prediction with TI (with added changes) (Close change only)")
    # plt.close()
    print(f'{stock} done')

