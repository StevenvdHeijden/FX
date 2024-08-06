import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from MACDBacktester import MACDBacktester as MACD
from RSIBacktester import RSIBacktester as RSI
from BBBacktester import BBBacktester as BB
from FRLBacktester import FRLBacktester as FRL
from SMABacktester import SMABacktester as SMA
from EMABacktester import EMABacktester as EMA
from SOBacktester import SOBacktester as SO

plt.style.use("seaborn")

# Define configurations
months = ['06','07','08','09','10','11']
currency_pair = 'AUDJPY'
time_interval = '30min'
ptc = 0.000035

# Initialize an empty DataFrame to store all results
all_months_output = pd.DataFrame()

def optimize_and_test_strategy(strategy_class, params, start_date, end_date, month, time_interval, ptc):
    ''' Optimize and test a trading strategy. '''
    strategy = strategy_class(currency_pair, *params, start_date, end_date, month, time_interval, ptc)
    optimal_params = strategy.optimize_parameters(*params[1:])
    strategy.test_strategy()
    strategy.plot_results()
    
    return optimal_params, strategy.results

for month in months:
    start_date = f'2023-{month}-01'
    end_date = f'2023-{month}-30'

    # MACD
    macd_params = (12, 26, 9)
    macd_opt_params, macd_results = optimize_and_test_strategy(MACD, (12, 26, 9, start_date, end_date, month, time_interval, ptc), start_date, end_date, month, time_interval, ptc)
    print('Optimal parameters MACD:', macd_opt_params)

    # RSI
    rsi_params = (20, 30, 70)
    rsi_opt_params, rsi_results = optimize_and_test_strategy(RSI, (20, 30, 70, start_date, end_date, month, time_interval, ptc), start_date, end_date, month, time_interval, ptc)
    print('Optimal parameters RSI:', rsi_opt_params)

    # SMA
    sma_params = (30, 80)
    sma_opt_params, sma_results = optimize_and_test_strategy(SMA, (30, 80, start_date, end_date, month, time_interval, ptc), start_date, end_date, month, time_interval, ptc)
    print('Optimal parameters SMA:', sma_opt_params)

    # EMA
    ema_params = (20, 60)
    ema_opt_params, ema_results = optimize_and_test_strategy(EMA, (20, 60, start_date, end_date, month, time_interval, ptc), start_date, end_date, month, time_interval, ptc)
    print('Optimal parameters EMA:', ema_opt_params)

    # SO
    so_params = (14, 3)
    so_opt_params, so_results = optimize_and_test_strategy(SO, (14, 3, start_date, end_date, month, time_interval, ptc), start_date, end_date, month, time_interval, ptc)
    print('Optimal parameters SO:', so_opt_params)

    # Fibonacci Retracement Levels
    frl_params = (0.236, 0.386, 0.618)
    frl_opt_params, frl_results = optimize_and_test_strategy(FRL, (0.236, 0.386, 0.618, start_date, end_date, month, time_interval, ptc), start_date, end_date, month, time_interval, ptc)
    print('Optimal Fibonacci strategy:', frl_opt_params)

    # Bollinger Bands
    bb_params = (20, 2)
    bb_opt_params, bb_results = optimize_and_test_strategy(BB, (20, 2, start_date, end_date, month, time_interval, ptc), start_date, end_date, month, time_interval, ptc)
    print('Optimal Bollinger Bands parameters:', bb_opt_params)

    # Combine results
    comb = macd_results.loc[:, ["returns", "position"]].copy()
    comb.rename(columns={"position": "position_MACD"}, inplace=True)
    comb["position_RSI"] = rsi_results.position.astype("int")
    comb["position_comb"] = np.where(comb.position_MACD == comb.position_RSI, comb.position_MACD, 0)
    
    comb["strategy"] = comb["position_comb"].shift(1) * comb["returns"]
    comb.dropna(inplace=True)
    
    comb["trades"] = comb.position_comb.diff().fillna(0).abs()
    comb["strategy"] = comb["strategy"] - comb["trades"] * ptc
    
    comb["creturns"] = comb["returns"].cumsum().apply(np.exp)
    comb["cstrategy"] = comb["strategy"].cumsum().apply(np.exp)

    title = f"{currency_pair} | MACD: {macd_opt_params} | RSI: {rsi_opt_params} | SMA: {sma_opt_params} | EMA: {ema_opt_params} | SO: {so_opt_params} | Fibonacci: {frl_opt_params} | BB: {bb_opt_params} | TC = {ptc}"
    comb[["creturns", "cstrategy"]].plot(title=title, figsize=(12, 8), fontsize=12)
    plt.savefig(f'{month}_{currency_pair}.png')

    # Compute performance metrics
    absolute_performance = comb[["returns", "strategy"]].sum().apply(np.exp)
    trades = (comb["trades"] == 2.0).sum()

    table = {
        'Month': month,
        'MACD EMA_S': macd_opt_params[0],
        'MACD EMA_L': macd_opt_params[1],
        'MACD Signal MW': macd_opt_params[2],
        'RSI Periods': rsi_opt_params[0],
        'RSI Lower Bound': rsi_opt_params[1],
        'RSI Upper Bound': rsi_opt_params[2],
        'SMA Short': sma_opt_params[0],
        'SMA Long': sma_opt_params[1],
        'EMA Short': ema_opt_params[0],
        'EMA Long': ema_opt_params[1],
        'SO Periods': so_opt_params[0],
        'SO D_MW': so_opt_params[1],
        'Fib 1': frl_opt_params[0],
        'Fib 2': frl_opt_params[1],
        'Fib 3': frl_opt_params[2],
        'BB SMA': bb_opt_params[0],
        'BB Deviation': bb_opt_params[1],
        'Returns': absolute_performance.returns,
        'Strategy': absolute_performance.strategy,
        'Trades': trades
    }

    output = pd.DataFrame(table, index=[0])
    all_months_output = pd.concat([all_months_output, output], ignore_index=True)
    print(all_months_output)

# Save all results to a CSV file
all_months_output.to_csv('indicator_Backtesting_JuneNovember.csv', sep=';', decimal=',', index=False)
