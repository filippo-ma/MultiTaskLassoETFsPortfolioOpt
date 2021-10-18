#import time
#import requests
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import helpers
from datetime import datetime, timedelta
import config
import portfolio_basecase 

import quantstats as qs

qs.extend_pandas()

# ETFs universe (equity and fixed income)


equity_etfs5 = ['EUE.MI','EXSI.MI', 'DAXX.MI', 'CSSX5E.MI', 'DJMC.MI', 'IDVY.MI', 'DJSC.MI', 
                'ISF.MI', 'FXC.MI', 'INFR.MI', 'ETFMIB.MI', 'IUKD.MI', 'IFFF.MI', 'CSCA.MI',
                'IEEM.MI', 'CSEMAS.MI', 'IMEU.MI', 'XMJP.MI', 'IKOR.MI', 'INAA.MI', 'CSUS.MI', 
                'LUSA.MI', 'CSUSS.MI', 'SWDA.MI', 'EQQQ.MI', 'CSNDX.MI', 'CSSPX.MI', 'INRG.MI', 
                'IH2O.MI', 'EXSA.MI', 'ENER.MI', 'WAT.MI', 'AGED.MI', 'GLDV.MI', 'RBOT.MI', 
                'SXLF.MI', 'XQUI.MI', 'MVOL.MI', 'R2US.MI', 'ROBO.MI', 'XDWH.MI', 'XDWT.MI']

fixinc_etfs5 = ['SEGA.MI', 'IEAC.MI', 'SE15.MI', 'IBGS.MI', 'IBGX.MI', 'IBCI.MI', 'EMI.MI', 
                'EM13.MI', 'ITPS.MI', 'XGIN.MI', 'IBTS.MI', 'IEMB.MI', 'SEML.MI', 'IBCX.MI', 
                'IHYG.MI', 'LQDE.MI', 'CORP.MI']


equity_etfs10 = ['CSSPX.MI','EQQQ.MI','CSUSS.MI', 'ENER.MI', 'EUE.MI', 'IDVY.MI', 'IEEM.MI', 'IH2O.MI', 'ISF.MI', 'WAT.MI', 'CSNDX.MI', 'DAXX.MI', 'INRG.MI', 'FXC.MI'] 
fixinc_etfs10 = ['IBCX.MI','IBGS.MI','IBGX.MI','IBTS.MI', 'IEAC.MI', 'IEMB.MI', 'IHYG.MI', 'LQDE.MI', 'XGIN.MI', 'ITPS.MI', 'IBCI.MI', 'SEML.MI']



years = config.years_tot
years_in_days = years * 365

start_date = (datetime.today() - timedelta(days=years_in_days)).strftime("%Y-%m-%d")
end_date = datetime.strptime(((datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d 00:00:00")), "%Y-%m-%d %H:%M:%S")

eq_etfs_data = yf.download(equity_etfs10, start=start_date)
fi_etfs_data = yf.download(fixinc_etfs10, start=start_date)

eq_etfs_adj_close = eq_etfs_data['Adj Close']
fi_etfs_adj_close = fi_etfs_data['Adj Close']

eq_monthly_rets = eq_etfs_adj_close.resample('MS').ffill().pct_change()[1:]
fi_monthly_rets = fi_etfs_adj_close.resample('MS').ffill().pct_change()[1:]

cum_eq_monthly_rets = eq_monthly_rets.apply(lambda x: x.add(1, fill_value=0).cumprod() - 1)
cum_fi_monthly_rets = fi_monthly_rets.apply(lambda x: x.add(1, fill_value=0).cumprod() - 1)


## plot rets
helpers.returns_chart(cum_eq_monthly_rets, emphasize_line_list=None, chart_title='Equity_etfs_rets')
helpers.returns_chart(cum_fi_monthly_rets, emphasize_line_list=None, chart_title='Fixed_income_etfs_rets')



## returns predictions 
eq_predictions = helpers.forecast_returns(eq_monthly_rets.dropna())
fi_predictions = helpers.forecast_returns(fi_monthly_rets.dropna())


# average equity prediction error
average_equity_return_error = eq_predictions.subtract(eq_monthly_rets).mean(axis=1).dropna()
equity_avg_error_plot_df = pd.DataFrame({'Avg Error': average_equity_return_error}, index=average_equity_return_error.index)
helpers.error_plot(equity_avg_error_plot_df, 'Equity')

# average fixed income prediction error
average_fi_return_error = fi_predictions.subtract(fi_monthly_rets).mean(axis=1).dropna()
fi_avg_error_plot_df = pd.DataFrame({'Avg Error': average_fi_return_error}, index=average_fi_return_error.index)
helpers.error_plot(fi_avg_error_plot_df, 'Fixed_Income')



# allocate strategy portfolio
params = {'expected_eq_returns': eq_predictions,
          'expected_fi_returns': fi_predictions,
          'actual_eq_returns': eq_monthly_rets,
          'actual_fi_returns': fi_monthly_rets}

portfolio_holdings_60 = helpers.get_historical_portfolio_holdings(**params, total_equity_weight=0.6)
portfolio_holdings_70 = helpers.get_historical_portfolio_holdings(**params, total_equity_weight=0.7)
bond_only_holdings = helpers.get_historical_portfolio_holdings(**params, total_equity_weight=0)
equity_only_holdings = helpers.get_historical_portfolio_holdings(**params, total_equity_weight=1)


# returns
new_60_40_returns = helpers.get_portfolio_returns(portfolio_holdings_60, 'Optimized 60/40')
new_70_30_returns = helpers.get_portfolio_returns(portfolio_holdings_70, 'Optimized 70/30')
bond_strategy_rtns = helpers.get_portfolio_returns(bond_only_holdings, 'Optimized Bond Strategy')
equity_strategy_rtns = helpers.get_portfolio_returns(equity_only_holdings, 'Optimized Equity Strategy')

new_rets = pd.concat([new_60_40_returns, new_70_30_returns, bond_strategy_rtns, equity_strategy_rtns], axis=1) #.dropna()
new_rets.index = pd.DatetimeIndex(new_rets.index)

all_rets = pd.concat([portfolio_basecase.monthly_rets_df, new_rets], axis=1).dropna()

cum_all_rets = all_rets.apply(lambda x: x.add(1, fill_value=0).cumprod() - 1)

helpers.returns_chart(cum_all_rets, emphasize_line_list=None, chart_title='All_strategies')


# reports (open in browser)
qs.reports.html(all_rets['Optimized 70/30'], all_rets['70/30 Portfolio'], title='70/30', output='output_reports/opt_vs_nopt_7030.html')
qs.reports.html(all_rets['Optimized 60/40'], all_rets['60/40 Portfolio'], title='60/40', output='output_reports/opt_vs_nopt_6040.html')
qs.reports.html(all_rets['Optimized Equity Strategy'], all_rets[config.eq_etf], title='100% Equity', output='output_reports/opt_vs_nopt_Equity.html')



