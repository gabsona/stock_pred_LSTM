from feature_engineering import *
from data_engineering import *
from modeling import *
from prediction import *
from visualisation import *
from helper_functions import *
import os
from csv import DictWriter

import joblib

def final_pred(ticker, change='absolute'):
    data = download_data(ticker, '2018-01-01', '2022-01-01', '1d')
    # data = all_indicators(data) # adds TIs
    data.dropna(inplace = True)
    data_tr = data_transform(data, change)
    X_train, y_train, X_test, y_test, scaler = data_split(data_tr.iloc[:, :-1], division='by date',
                                                          split_criteria='2021-01-01', scale='yes', step_size=30)
    grid_model, grid_model_reg = build_model(X_train, loss='mse', optimizer='adam')
    # grid_model_reg = KerasRegressor(build_fn=grid_model, verbose=1)

    my_model, grid_result = best_model(X_train, y_train, grid_model_reg, ticker, cv=3)
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

    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    mse_train = mean_squared_error(y_train, grid_result.predict(X_train))
    mse_test = mean_squared_error(y_test, grid_result.predict(X_test))
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    print("Train MSE : {}".format(mean_squared_error(y_train, grid_result.predict(X_train))))
    print("Test  MSE : {}".format(mean_squared_error(y_test, grid_result.predict(X_test))))

    print("\nTrain R^2 : {}".format(grid_result.score(X_train, y_train)))
    print("Test  R^2 : {}".format(grid_result.score(X_test, y_test)))


    plot_results(ticker, df_preds_abs, change=change)
    plot_loss(my_model, ticker)
    best_score = grid_result.best_score_
    best_params = grid_result.best_params_

    return df_preds, df_preds_abs, classification_accuracy, best_score, best_params, mse_train, mse_test

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


def makemydir(df, stock, folder_name):
    dir = os.path.join("C:/Users/AI_BootCamp_06/Desktop/LSTMM/", folder_name)
    if not os.path.exists(dir):
        os.makedirs(dir)
    # os.chdir(dir)
    df.to_csv(dir + f'\\df_{stock}_change.csv')

acc_list = []
scores = []
best_params_ = []
mse_train_ = []
mse_test_ = []
dict_acc = {'Stock': [], 'Accuracy': [], 'Score': [], 'MSE train': [], 'MSE test': [], 'Best Parameters': []}
df_acc = pd.DataFrame(dict_acc)
df_acc.to_csv('dict_15.07.csv', index = False)

# stocks = ['NFLX', 'MSFT', 'V', 'AMZN', 'TWTR', 'AAPL', 'GOOG', 'TSLA', 'FB', 'NVDA', 'JNJ', 'UNH', 'XOM', 'JPM', 'CVX', 'MA', 'WMT', 'HD', 'PFE', 'BAC', 'LLY', 'KO', 'ABBV']
# stocks = ['JNJ', 'XOM', 'JPM', 'CVX', 'MA', 'WMT', 'HD', 'PFE', 'BAC', 'LLY', 'KO']
# stocks = ['CVX', 'MA', 'WMT', 'HD']
stocks = ['XOM', 'JPM'] # no PG

for stock in stocks:
    df_preds, df_preds_abs, clf_acc, score, best_params, mse_train, mse_test = final_pred(stock, change='absolute')
    makemydir(df_preds, stock, "Stock Price Prediction (absolute change) 15.07")
    makemydir(df_preds_abs, stock, "Stock Price Prediction(with added changes) (absolute change) 15.07")
    dict_append = {'Stock': stock, 'Accuracy':clf_acc, 'Score': score, 'MSE train': mse_train, 'MSE test':mse_test, 'Best Parameters':best_params}
    # Open your CSV file in append mode
    # Create a file object for this file
    with open('dict_15.07.csv', 'a', newline='') as f_object:

        fieldnames = ['Stock', 'Accuracy', 'Score', 'MSE train', 'MSE test', 'Best Parameters']
        dictwriter_object = DictWriter(f_object, fieldnames = dict_append)

        # Passing the dictionary as an argument to the Writerow()
        dictwriter_object.writerow(dict_append)

        # Closing the file object
        f_object.close()
    # acc_list.append(clf_acc)
    # scores.append(score)
    # mse_train_.append(mse_train)
    # mse_test_.append(mse_test)
    # best_params_.append(best_params)

    # plt.close()
    # print(f'{stock} done')
#
# dict_acc = {'Stock': stocks, 'Accuracy':acc_list, 'Score': scores, 'MSE train': mse_train_, 'MSE test':mse_test_, 'Best Parameters':best_params_}
# df_acc = pd.DataFrame(dict_acc)
#
# print(df_acc)
# df_acc.to_csv('accuracy_15.07.csv')
