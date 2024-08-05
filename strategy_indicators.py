import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
from MACDBacktester import MACDBacktester as MACD
from RSIBacktester import RSIBacktester as RSI
from PPBacktester import PPBacktester as PP
from BBBacktester import BBBacktester as BB
from FRLBacktester import FRLBacktester as FRL
from SMABacktester import SMABacktester as SMA
from EMABacktester import EMABacktester as EMA
from SOBacktester import SOBacktester as SO
# Volume indicator

plt.style.use("seaborn")

months = ['06','07','08','09','10','11'] # Define here the inputs of the months within the year (initialized as 2023) that you want to use for the backscoring.
currency_pair = 'AUDJPY' # Define the currency pair
time_interval = '30min' # Define time interval
ptc = 0.000035 # Define costs of trading

#weeks = [["01","04"],["07","11"],["14","18"],["21","25"],["28","31"]] # Only applicable when backscoring is performed for 1 month

all_months_output = pd.DataFrame()
for month in months:
    # for week in weeks:
    # Initialize start date and end date
    #start_date = f'2023-{month}-{week[0]}'
    #end_date = f'2023-{month}-{week[1]}'
    start_date = f'2023-{month}-01'
    end_date = f'2023-{month}-30'

    # MACD class initialization - V
    ema_s_macd = 12
    ema_l_macd = 26
    signal_mw = 9
    macd = MACD(currency_pair, ema_s_macd, ema_l_macd, signal_mw, start_date, end_date, month, time_interval, ptc)
    # Optimize parameters + save plot of results
    print('Optimal parameters MACD: [ema_s, ema_l, signal_mw]', macd.optimize_parameters((5, 20, 1),(21,50,1),(5,20,1)))
    macd.test_strategy()
    macd.plot_results()
    
    # RSI class initialization - V
    periods_rsi = 20
    rsi_upper = 70
    rsi_lower = 30
    rsi = RSI(currency_pair, periods_rsi, rsi_lower, rsi_upper, start_date, end_date, month, time_interval, ptc)
    # Optimize parameters + save plot of results
    print('Optimal parameters RSI: [periods_rsi, rsi_lower, rsi_upper]', rsi.optimize_parameters((5, 20, 1),(20,35,1),(65,80,1))) # higher votality
    #print(rsi.optimize_parameters((20, 50, 1),(15,25,1),(75,85,1))) # lower votality
    rsi.test_strategy()
    rsi.plot_results()

    # SMA class initialization - V
    sma_s = 30
    sma_l = 80
    sma = SMA(currency_pair, sma_s, sma_l, start_date, end_date, month, time_interval, ptc)
    print('Optimal parameters SMA: [sma_s, sma_l, yadano ]', sma.optimize_parameters((20, 50, 1),(51,150,1),(5,20,1)))
    sma.test_strategy()
    sma.plot_results()

    # EMA class initialization - V
    ema_s = 20
    ema_l = 60
    ema = EMA(currency_pair, ema_s, ema_l, start_date, end_date, month, time_interval, ptc)
    # Optimize parameters + save plots
    print(ema.optimize_parameters((10, 40, 1),(41,80,1)))
    ema.test_strategy()
    ema.plot_results()

    # SO class inititialization - V
    periods_so = 14
    D_mw = 3
    so = SO(currency_pair, periods_so, D_mw, start_date, end_date, month, time_interval, ptc)
    # Optimize parameters + save plots
    print(so.optimize_parameters((10, 100, 1),(3, 50, 1)))
    so.test_strategy()
    so.plot_results()

    # Fibonacci Retracement Level class initialization
    fib1 = 0.236
    fib2 = 0.386
    fib3 = 0.618
    frl = FRL(currency_pair, fib1, fib2, fib3, start_date, end_date, month, time_interval, ptc)
    # Optimize parameters + save plots
    print(frl.optimize_parameters((5, 20, 1),(21,50,1),(5,20,1)))
    frl.test_strategy()
    frl.plot_results()

    # Bollinger Bands class initialization
    sma_bb = 20
    dev = 2
    bb = BB(currency_pair, sma_bb, dev, start_date, end_date, month, time_interval, ptc)
    # Optimize parameters + save plots
    print(bb.optimize_parameters((5, 20, 1),(21,50,1),(5,20,1)))
    bb.test_strategy()
    bb.plot_results()

    # Pivot Point class initialization
    indicator = 'value'
    pp = PP(currency_pair, indicator, start_date, end_date, month, time_interval, ptc)
    # Optimize parameters + save plots
    print(pp.optimize_parameters((5, 20, 1),(21,50,1),(5,20,1)))
    pp.test_strategy()
    pp.plot_results()

    comb = macd.results.loc[:,["returns","position"]].copy()
    comb.rename(columns = {"position":"position_MACD"}, inplace=True)
    comb["position_RSI"] = rsi.results.position.astype("int")
    comb["position_comb"] = np.where(comb.position_MACD == comb.position_RSI, comb.position_MACD, 0)
    
    comb["strategy"] = comb["position_comb"].shift(1) * comb["returns"]
    comb.dropna(inplace=True)
    
    comb["trades"] = comb.position_comb.diff().fillna(0).abs()
    comb.strategy = comb.strategy - comb.trades * ptc
    
    comb["creturns"] = comb["returns"].cumsum().apply(np.exp)
    comb["cstrategy"] = comb["strategy"].cumsum().apply(np.exp)

    title = "{} | EMA_S = {} | EMA_L = {} | SIGNAL_MW = {} | PERIODS = {} | RSI_LOWER = {} | RSI_UPPER = {} |TC = {}".format(macd.symbol, macd.ema_s, macd.ema_l, macd.signal_mw, rsi.periods, rsi.rsi_lower, rsi.rsi_upper, rsi.tc)
    comb[["creturns", "cstrategy"]].plot(title=title, figsize = (12,8), fontsize=12)
    plt.savefig(f'{month}_AUDJPY.png')
    # plt.show()

    absolute_performance = comb[["returns", "strategy"]].sum().apply(np.exp) # absolute performance
    print(absolute_performance.strategy)
    trades = (comb["trades"] == 2.0).sum()

    table = {
    'Optimal MACD EMA_S' : macd.ema_s,
    'Optimal MACD EMA_L' : macd.ema_l,
    'Optimal MACD Signal MW' : macd.signal_mw,
    'Optimal RSI periods' : rsi.periods,
    'Optimal RSI lower bound' : rsi.rsi_lower,
    'Optimal RSI upper bound' : rsi.rsi_upper,
    'Returns' : absolute_performance.returns,
    'Strategy' : absolute_performance.strategy,
    'Trades' : trades,
    } 

    # Create a DataFrame
    output = pd.DataFrame(table, index=[0])

    all_months_output = pd.concat([all_months_output, output])
    print(all_months_output)

all_months_output.to_csv('indicator_Backtesting_JuneNovember.csv', sep=';', decimal=',', index=False)
