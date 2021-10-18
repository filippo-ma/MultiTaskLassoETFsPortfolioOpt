import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from scipy import stats
from sklearn.linear_model import MultiTaskLasso
import config

import os

if not os.path.exists("images"):
    os.mkdir("images")


eq_etf_symbol = config.eq_etf
fi_etf_symbol = config.fi_etf

def adj_date_index(obj_ind):
    as_list = obj_ind.index.tolist()
    list_ind_el = as_list[-1]
    idx = as_list.index(list_ind_el)
    as_list[idx] = end_date
    obj_ind.index = as_list


# chart returns
def make_line(rets_df, column, alt_name=None):
    data = rets_df[[column]]
    name = column
    if alt_name is not None:
        name = f"{alt_name} ({column})"

    return go.Scatter(x=data.index, y=data[column], name=name)


def make_chart(rets_df, emphasize=None):
    alt_names = {eq_etf_symbol: '100% Equities', fi_etf_symbol: '100% Bonds'}
    data = []
    for column in rets_df:
        alt_name = None
        if column in alt_names:
            alt_name = alt_names[column]
        chart = make_line(rets_df, column, alt_name)

        if emphasize is not None:
            if type(emphasize) != list:
                emphasize = [emphasize]
            if column not in emphasize:
                chart.line.width = 1
                chart.mode = 'lines'
            else:
                chart.line.width = 2
                chart.mode = 'lines+markers'
        data.append(chart)
    return data

def returns_chart(rets_df, emphasize_line_list, chart_title):

    data = make_chart(rets_df, emphasize_line_list)

    layout = {'template': 'plotly_dark',
            'title': chart_title,
            'xaxis': {'title': {'text': 'Date'}},
            'yaxis': {'title': {'text': 'Cumulative Total Return'},
                        'tickformat': '.0%'}}

    figure1 = go.Figure(data=data, layout=layout)
    figure1.write_image(f"images/{chart_title}.png")


def log_returns_chart(rets_df, emphasize_line_list, chart_title):

    data = make_chart(rets_df, emphasize_line_list)

    layout = ({'template': 'plotly_dark',
           'xaxis': {'title': {'text': 'Date'}},
           'yaxis': {'title': {'text': 'Cumulative Total Return'},
                     'type': 'log', 'tickformat': '$.3s'},
            'title': f'{chart_title} - Logarithmic Scale'})

    figure1 = go.Figure(data=data, layout=layout)
    figure1.write_image(f"images/{chart_title}.png")


def forecast_returns(return_time_series_data, non_return_data=None, window_size=5, num_test_dates=90):
    """
    Use a given dataset and the MultiTaskLasso object from sklearn to 
    generate a DataFrame of predicted returns
    
    Args:
    ================================
    return_time_series_data (pandas.DataFrame):
        pandas DataFrame of an actual return time series for a set of given indices.
        Must be in the following format:
        
         Period     |    
         Ending     |    Ticker_1    Ticker_2     ...    Ticker_N
       -----------  |   ----------  ----------   -----  ----------
       YYYY-MM-DD   |      0.01        0.03       ...     -0.05
                    |
       YYYY-MM-DD   |     -0.05       -0.01       ...      0.04
       
       
    non_return_data (pandas.DataFrame):
        pandas DataFrame of an actual time series of non-return data
        for a set of given indices. Must be in the same format, same
        ticker order, and have the same periodicity as the return_time_series_data above
        
        
    window_size (int):
        Number of periods used to predict the next value.
        Example: if window_size = 5, look 5 periods back to predict the next value
        Default = 5
        
    
    num_test_dates (int):
        Number of periods for which to generate forecasts
        Example: 120 = 10 years of monthly predictions, or 30 years of quarterly predicitons
        depending on the periodicity of the input data in return_time_series_data and non_return_data
        Default = 120
        
        
    Returns:
    ================================
    pandas.DataFrame
        Output is a DataFrame of expected returns in the same format as return_time_series_data
    
    """
    
    # descriptive variables for later use
    names = list(return_time_series_data.columns)
    dates = [f'{date.year}-{date.month}-{date.day}' for date in list(pd.to_datetime(return_time_series_data.index))]
    
    # transform pandas to numpy arrays
    X_returns = return_time_series_data.to_numpy()
    X_input = X_returns
    max_iter = 7500
    
    # concatenate non_return_data if it exists
    if non_return_data is not None:
        max_iter = 3000
        X_non_rtn = non_return_data.to_numpy()
        X_input =  np.concatenate((X_returns, X_non_rtn), axis=1)
    
    # number of time series (tickers) to model
    n_series = X_returns.shape[1]
    # number of features at each date; equal to n_series * number of features (return, oas_spread, etc.)
    n_features_per_time_point = X_input.shape[1]
    
    num_features = window_size * n_features_per_time_point
    num_training_points = X_returns.shape[0] - window_size
    X_train = np.zeros((num_training_points, num_features))
    Y_train = X_returns[window_size:,:]
    
    for i in range(num_training_points-1):
        X_train[i,:] = np.matrix.flatten(X_input[i : window_size + i,:])
    
    # establish empty arrays & variables for use in training each model
    mtl_list=[]
    alpha= 0.001
    Y_pred = np.zeros((num_test_dates, n_series))
    delta_Y = np.zeros((num_test_dates, n_series))
    dY_percent = np.zeros((num_test_dates, n_series))
    mse_pred = np.zeros(num_test_dates)
    predict_dates=[]    
    
    # loop through dates & predict returns
    for i in range(num_test_dates):
        X_i = X_train[:num_training_points - num_test_dates + (i-1)]
        Y_i = Y_train[:num_training_points - num_test_dates + (i-1)]
        print("X shape: ", X_i.shape, "Y shape: ", Y_i.shape)
        print("number of points in training data:", X_i.shape[0] )
        mtl = MultiTaskLasso(alpha=alpha, max_iter=max_iter, warm_start=True).fit(X_i, Y_i)
        mtl_list.append(mtl)
        
        print(f"using X from {dates[num_training_points - num_test_dates + (i-1) + window_size]}\
        to predict {dates[num_training_points - num_test_dates + (i-1) + 1 + window_size]}")
        
        predict_dates.append(dates[num_training_points - num_test_dates + (i-1) + window_size])
        
        X_i_plus_1 = X_train[num_training_points - num_test_dates + (i-1) + 1]
        
        Y_pred[i,:] = mtl.predict([X_i_plus_1])
        Y_act =  Y_train[num_training_points - num_test_dates + (i-1) + 1]
        delta_Y[i] = (Y_pred[i,:] - Y_act)
        mse_pred[i] = np.sqrt(np.sum((Y_pred[i,:] - Y_act)**2))/len(Y_act)
        print("mse", mse_pred[i])
    
    predictions = pd.DataFrame(Y_pred, index=predict_dates, columns=names)
    predictions.index = [pd.Timestamp(i).strftime('%Y-%m-%d') for i in predictions.index]
    
    return predictions




# error plot 
def SetColor(y):
    if(y < 0):
        return "red"
    elif(y >= 0):
        return "green"

def error_plot(err_df, equity_or_fixed_inc):
    layout = ({'template': 'plotly_dark',
                'xaxis': {'title': {'text': 'Date'}},
                'yaxis': {'title': {'text': 'Avg Error %'}},
                'title': f"Average_{equity_or_fixed_inc}_Prediction_Error"})

    fig = go.Figure(layout=layout)

    fig.add_trace(go.Bar(
        x=err_df.index,
        y=err_df.iloc[:,0],
        marker=dict(color = list(map(SetColor, err_df.iloc[:,0])))
        ))

    fig.write_image(f"images/{layout['title']}.png")




def allocate_portfolio(expected_eq_returns, 
                       expected_fi_returns, 
                       actual_eq_returns,
                       actual_fi_returns,
                       for_period_ending,
                       total_equity_weight=0.6,
                       n_equity_funds=5,
                       n_bond_funds=5):
    
    """
    Allocate a portfolio by picking the top n_equity_funds & top n_bond_funds for the period
    ending on for_period_ending
    
    """
    
    fi_wgt = 1 - total_equity_weight
    eq_fund_wgt = total_equity_weight / n_equity_funds
    fi_fund_wgt = fi_wgt / n_bond_funds
    for_period_ending = pd.Timestamp(for_period_ending).strftime('%Y-%m-%d')
    
    eq_returns = pd.DataFrame(expected_eq_returns.loc[for_period_ending])
    eq_returns.columns = ['Expected Return']
    eq_returns['Type'] = ['Equity'] * len(eq_returns)
    eq_returns['Weight'] = [eq_fund_wgt] * len(eq_returns)
    eq_returns = eq_returns.sort_values(by='Expected Return', ascending=False).head(n_equity_funds)

    fi_returns = pd.DataFrame(expected_fi_returns.loc[for_period_ending])
    fi_returns.columns = ['Expected Return']
    fi_returns['Type'] = ['Fixed Income'] * len(fi_returns)
    fi_returns['Weight'] = [fi_fund_wgt] * len(fi_returns)
    fi_returns = fi_returns.sort_values(by='Expected Return', ascending=False).head(n_bond_funds)
    
    holdings_df = pd.concat([eq_returns, fi_returns], axis=0)
    holdings_df.index.name = 'Index'
    
    actual_returns = []
    for i in range(len(holdings_df)):
        index_type = holdings_df['Type'].iloc[i]
        index_name = holdings_df.index[i]
        if index_type == 'Equity':
            actual_returns.append(actual_eq_returns[index_name].loc[for_period_ending])
        elif index_type == 'Fixed Income':
            actual_returns.append(actual_fi_returns[index_name].loc[for_period_ending])
    holdings_df['Actual Return'] = actual_returns
    
    holdings_df.index = pd.MultiIndex.from_tuples([(for_period_ending, i) for i in holdings_df.index], names=['For Period Ending', 'Fund Ticker'])
    holdings_df = holdings_df[['Type', 'Weight', 'Expected Return', 'Actual Return']]
    
    return holdings_df



def get_historical_portfolio_holdings(expected_eq_returns, 
                                      expected_fi_returns, 
                                      actual_eq_returns, 
                                      actual_fi_returns, 
                                      total_equity_weight):
    """
    Loop over the time frame given in expected_fi_returns 
    and run allocate_portfolio at each date
    
    """

    holdings = []
    for date in expected_fi_returns.index:
        holdings_at_date = allocate_portfolio(expected_eq_returns=expected_eq_returns, 
                                              expected_fi_returns=expected_fi_returns, 
                                              actual_eq_returns=actual_eq_returns,
                                              actual_fi_returns=actual_fi_returns,
                                              for_period_ending=date, 
                                              total_equity_weight=total_equity_weight)
        holdings.append(holdings_at_date)
    return pd.concat(holdings)



def get_portfolio_returns(portfolio_holdings_df, port_name='Optimized Portfolio'):
    weighted_returns = portfolio_holdings_df['Actual Return'] * portfolio_holdings_df['Weight']
    returns_df = pd.DataFrame(weighted_returns.groupby(level=[0]).sum())
    returns_df.columns = [port_name]
    return returns_df


