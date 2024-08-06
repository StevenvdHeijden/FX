import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brute
plt.style.use("seaborn")
import period_converter

class PPBacktester():
    ''' Class for the vectorized backtesting of a Pivot Point-based trading strategy.

    Attributes
    ==========
    symbol : str
        Ticker symbol to work with.
    start : str
        Start date for data retrieval.
    end : str
        End date for data retrieval.
    month : str
        The month to extract the data from.
    time_interval : str
        Time interval for candlestick data.
    tc : float
        Proportional transaction costs per trade.
        
    Methods
    =======
    get_data:
        Retrieves and prepares the data.
        
    calculate_pivot_points:
        Calculates pivot points and support/resistance levels.
        
    test_strategy:
        Runs the backtest for the Pivot Point-based strategy.
        
    plot_results:
        Plots the performance of the strategy compared to buy and hold.
        
    update_and_run:
        Updates Pivot Point levels and returns the negative absolute performance (for minimization algorithm).
        
    optimize_parameters:
        Implements a brute force optimization for the Pivot Point levels.
    '''

    def __init__(self, symbol, start, end, month, time_interval, tc):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.month = month
        self.time_interval = time_interval
        self.tc = tc
        self.results = None
        self.get_data()
        
    def __repr__(self):
        return f"PPBacktester(symbol={self.symbol}, start={self.start}, end={self.end}, month={self.month})"
        
    def get_data(self):
        ''' Retrieves and prepares the data.
        '''
        column_names = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        folder_path = f"C:\\Users\\steve\\AlgoTradingFrx\\csv files\\{self.symbol}\\HISTDATA_COM_MT_{self.symbol}_M12023{self.month}\\"
        raw = pd.read_csv(f"{folder_path}DAT_MT_{self.symbol}_M1_2023{self.month}.csv", sep=',', header=None)
        raw.columns = column_names
        
        # Combine date and time columns for datetime index
        raw['DateTime'] = pd.to_datetime(raw['Date'] + ' ' + raw['Time'])
        raw.set_index('DateTime', inplace=True)
        raw = raw[self.start:self.end]
        raw.rename(columns={"Close": "price"}, inplace=True)
        
        # Convert 1 min chart into other timeframes
        raw = period_converter.PeriodConverter(raw).convert_to_timeframe(self.time_interval)
        raw["returns"] = np.log(raw.price / raw.price.shift(1))
        raw.dropna(inplace=True)
        
        self.data = raw
    
    def calculate_pivot_points(self):
        ''' Calculates pivot points and support/resistance levels.
        '''
        data = self.data.copy()
        
        data = data.tz_localize("UTC")
        data = data.tz_convert("US/Eastern")
        
        close = data.price.to_frame().copy()
        daily_close = close.resample("D", offset="17H").last().dropna()
        
        agg_dict = {"Open": "first", "High": "max", "Low": "min", "price": "last"}
        daily_data = data.resample("D", offset="17H").agg(agg_dict).dropna()
        daily_data.columns = ["Open_d", "High_d", "Low_d", "Close_d"]
        daily_data = daily_data.shift().dropna()
        
        data = pd.concat([data, daily_data], axis=1).ffill().dropna()
        
        data["PP"] = (data.High_d + data.Low_d + data.Close_d) / 3
        data["S1"] = data.PP * 2 - data.High_d
        data["S2"] = data.PP - (data.High_d - data.Low_d)
        data["R1"] = data.PP * 2 - data.Low_d
        data["R2"] = data.PP + (data.High_d - data.Low_d)
        
        self.data = data
    
    def test_strategy(self):
        ''' Backtests the Pivot Point-based strategy.
        '''
        self.calculate_pivot_points()
        data = self.data.copy().dropna()
        
        data["position"] = np.where(data.Open > data.PP, 1, -1)
        data["position"] = np.where(data.Open >= data.R1, 0, data.position)
        data["position"] = np.where(data.Open <= data.S1, 0, data.position)
        data["position"] = data.position.fillna(0)
        
        data["returns"] = np.log(data.Open.shift(-1).div(data.Open))
        data["strategy"] = data.position * data["returns"]
        
        data["trades"] = data.position.diff().fillna(0).abs()
        data["strategy"] = data["strategy"] - data["trades"] * self.tc
        
        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        
        self.results = data
        
        if not data.empty:
            perf = data["cstrategy"].iloc[-1]
            outperf = perf - data["creturns"].iloc[-1]
        else:
            perf = 1
            outperf = 0
        
        return round(perf, 6), round(outperf, 6)

    def plot_results(self):
        ''' Plots the cumulative performance of the trading strategy compared to buy and hold.
        '''
        if self.results is None:
            print("No results to plot yet. Run a strategy.")
        else:
            title = f"{self.symbol} | TC = {self.tc}"
            self.results[["creturns", "cstrategy"]].plot(title=title, figsize=(12, 8))
            plt.savefig(f'{self.month}_PP_{self.symbol}.png')

    def update_and_run(self, params):
        ''' Updates strategy parameters and returns the negative absolute performance (for minimization algorithm).

        Parameters
        ==========
        params : tuple
            Strategy parameters tuple
        '''
        self.tc = params
        return -self.test_strategy()[0]

    def optimize_parameters(self, tc_range):
        ''' Finds the optimal transaction cost parameter.

        Parameters
        ==========
        tc_range : tuple
            Tuple of the form (start, end, step size)
        '''
        opt = brute(self.update_and_run, (tc_range,), finish=None)
        return opt, -self.update_and_run(opt)
