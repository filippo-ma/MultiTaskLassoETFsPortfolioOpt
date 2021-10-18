import time
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import helpers
from datetime import datetime, timedelta

import config




# fetch data: equity etf - fixed income etf
years = config.years_tot
years_in_days = years * 365

start_date = (datetime.today() - timedelta(days=years_in_days)).strftime("%Y-%m-%d")
end_date = datetime.strptime(((datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d 00:00:00")), "%Y-%m-%d %H:%M:%S")

eq_etf_symbol = config.eq_etf
fi_etf_symbol = config.fi_etf

eq_data = yf.download(eq_etf_symbol, start=start_date)
fi_data = yf.download(fi_etf_symbol, start=start_date)

eq_adj_close = eq_data['Adj Close']
fi_adj_close = fi_data['Adj Close']

eq_daily_rets = eq_adj_close.pct_change().rename(eq_etf_symbol)
fi_daily_rets = fi_adj_close.pct_change().rename(fi_etf_symbol)

eq_monthly_rets = eq_adj_close.resample('MS').ffill().pct_change().rename(eq_etf_symbol)
fi_monthly_rets = fi_adj_close.resample('MS').ffill().pct_change().rename(fi_etf_symbol)

daily_rets_df = pd.concat([eq_daily_rets, fi_daily_rets], axis=1)
monthly_rets_df = pd.concat([eq_monthly_rets, fi_monthly_rets], axis=1)


# add 60/40 & 70/30 portfolios base case
daily_rets_df['60/40 Portfolio'] =  sum([daily_rets_df[eq_etf_symbol]*0.6, daily_rets_df[fi_etf_symbol]*0.4])
daily_rets_df['70/30 Portfolio'] =  sum([daily_rets_df[eq_etf_symbol]*0.7, daily_rets_df[fi_etf_symbol]*0.3])

monthly_rets_df['60/40 Portfolio'] =  sum([monthly_rets_df[eq_etf_symbol]*0.6, monthly_rets_df[fi_etf_symbol]*0.4])
monthly_rets_df['70/30 Portfolio'] =  sum([monthly_rets_df[eq_etf_symbol]*0.7, monthly_rets_df[fi_etf_symbol]*0.3])

# cumulative returns base case
cum_daily_rets = daily_rets_df.apply(lambda x: x.add(1, fill_value=0).cumprod() - 1)
cum_monthly_rets = monthly_rets_df.apply(lambda x: x.add(1, fill_value=0).cumprod() - 1)
# (plus)
invested_amount = 100
tot_daily_rets_inv = daily_rets_df.apply(lambda x: x.add(invested_amount, fill_value=0).cumprod())
tot_monthly_rets_inv = monthly_rets_df.apply(lambda x: x.add(invested_amount, fill_value=0).cumprod())
# log rets 
log_cum_monthly_rets = monthly_rets_df.apply(lambda x: x.add(1, fill_value=0).cumprod() * 100)


# plot cumulative returns
helpers.returns_chart(cum_monthly_rets, emphasize_line_list=['60/40 Portfolio', '70/30 Portfolio'], chart_title='60_40_70_30_Base_Case')

# plot log cumulative returns
helpers.log_returns_chart(log_cum_monthly_rets, emphasize_line_list=['60/40 Portfolio', '70/30 Portfolio'], chart_title='60_40_70_30_Base_Case_log')




