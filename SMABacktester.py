import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brute
import period_converter
plt.style.use("seaborn")


class SMABacktester:
    ''' Class for the vectorized backtesting of SMA-based trading strategies.

    Attributes
    ==========
    symbol : str
        Ticker symbol to work with.
    SMA_S : int
        Time window in days for the shorter SMA.
    SMA_L : int
        Time window in days for the longer SMA.
    start : str
        Start date for data retrieval.
    end : str
        End date for data retrieval.
    month : str
        Month for the data file.
    time_interval : str
        Time interval for the data conversion.
    ptc : float
        Proportional transaction cost.
        
        
    Methods
    =======
    get_data:
        Retrieves and prepares the data.
        
    set_parameters:
        Sets one or two new SMA parameters.
        
    test_strategy:
        Runs the backtest for the SMA-based strategy.
        
    plot_results:
        Plots the performance of the strategy compared to buy and hold.
        
    update_and_run:
        Updates SMA parameters and returns the negative absolute performance (for minimization algorithm).
        
    optimize_parameters:
        Implements a brute force optimization for the two SMA parameters.
    '''

    def __init__(self, symbol, SMA_S, SMA_L, start, end, month, time_interval, ptc):
        self.symbol = symbol
        self.SMA_S = SMA_S
        self.SMA_L = SMA_L
        self.start = start
        self.end = end
        self.month = month
        self.time_interval = time_interval
        self.ptc = ptc
        self.results = None
        self.get_data()

    def __repr__(self):
        return f"SMABacktester(symbol={self.symbol}, SMA_S={self.SMA_S}, SMA_L={self.SMA_L}, start={self.start}, end={self.end})"

    def get_data(self):
        ''' Retrieves and prepares the data.
        '''
        column_names = ["Date", "Time", "Open", "High", "Low", "Close", "Volume"]
        folder_path = f"C:{folderpath_name}\\{self.symbol}\\HISTDATA_COM_MT_{self.symbol}_M12023{self.month}\\"
        raw = pd.read_csv(f"{folder_path}DAT_MT_{self.symbol}_M1_2023{self.month}.csv", sep=',', header=None)
        raw.columns = column_names

        # Convert 1 min chart into other timeframes
        raw = period_converter.PeriodConverter(raw).convert_to_timeframe(self.time_interval)
        raw = raw[self.start:self.end]
        raw.dropna(inplace=True)

        # Calculate logarithmic returns
        raw["returns"] = np.log(raw["price"] / raw["price"].shift(1))

        # Calculate the SMAs
        raw["SMA_S"] = raw["price"].rolling(self.SMA_S).mean()
        raw["SMA_L"] = raw["price"].rolling(self.SMA_L).mean()

        self.data = raw

    def set_parameters(self, SMA_S=None, SMA_L=None):
        ''' Updates SMA parameters and respective time series.
        '''
        if SMA_S is not None:
            self.SMA_S = SMA_S
            self.data["SMA_S"] = self.data["price"].rolling(self.SMA_S).mean()
        if SMA_L is not None:
            self.SMA_L = SMA_L
            self.data["SMA_L"] = self.data["price"].rolling(self.SMA_L).mean()

    def test_strategy(self):
        ''' Backtests the trading strategy.
        '''
        data = self.data.copy().dropna()
        data["position"] = np.where(data["SMA_S"] > data["SMA_L"], 1, -1)  # long if SMA_S > SMA_L, short otherwise
        data["strategy"] = data["position"].shift(1) * data["returns"]
        data.dropna(inplace=True)

        # Calculate cumulative returns
        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)

        self.results = data

        # Absolute performance of the strategy
        perf = data["cstrategy"].iloc[-1]
        # Out-/underperformance of strategy
        outperf = perf - data["creturns"].iloc[-1]

        return round(perf, 6), round(outperf, 6)

    def plot_results(self):
        ''' Plots the cumulative performance of the trading strategy
        compared to buy and hold.
        '''
        if self.results is None:
            print("No results to plot yet. Run a strategy.")
        else:
            title = f"{self.symbol} | SMA_S = {self.SMA_S} | SMA_L = {self.SMA_L}"
            self.results[["creturns", "cstrategy"]].plot(title=title, figsize=(12, 8))
            plt.savefig(f'{self.month}_SMA_{self.symbol}.png')

    def update_and_run(self, SMA):
        ''' Updates SMA parameters and returns the negative absolute performance (for minimization algorithm).

        Parameters
        ==========
        SMA : tuple
            SMA parameter tuple
        '''
        self.set_parameters(int(SMA[0]), int(SMA[1]))
        return -self.test_strategy()[0]

    def optimize_parameters(self, SMA1_range, SMA2_range):
        ''' Finds global maximum given the SMA parameter ranges.

        Parameters
        ==========
        SMA1_range, SMA2_range : tuple
            Tuples of the form (start, end, step size)
        '''
        opt = brute(self.update_and_run, (SMA1_range, SMA2_range), finish=None)
        return opt, -self.update_and_run(opt)
