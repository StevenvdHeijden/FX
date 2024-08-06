import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brute
plt.style.use("seaborn")
import period_converter

class RSIBacktester:
    ''' Class for the vectorized backtesting of RSI-based trading strategies.

    Attributes
    ==========
    symbol : str
        Ticker symbol to work with.
    periods : int
        Time periods for calculating the RSI.
    rsi_lower : int
        Lower bound for the RSI to signal a buy.
    rsi_upper : int
        Upper bound for the RSI to signal a sell.
    start : str
        Start date for data retrieval.
    end : str
        End date for data retrieval.
    month : str
        The month to extract the data from.
    ti : str
        Time interval for candlestick data.
    tc : float
        Proportional transaction costs per trade.
        
        
    Methods
    =======
    get_data:
        Retrieves and prepares the data.
        
    set_parameters:
        Sets new RSI parameters.
        
    test_strategy:
        Runs the backtest for the RSI-based strategy.
        
    plot_results:
        Plots the performance of the strategy compared to buy and hold.
        
    update_and_run:
        Updates RSI parameters and returns the negative absolute performance (for minimization algorithm).
        
    optimize_parameters:
        Implements a brute force optimization for the RSI parameters.
    '''

    def __init__(self, symbol, periods, rsi_lower, rsi_upper, start, end, month, ti, tc):
        self.symbol = symbol
        self.periods = periods
        self.rsi_lower = rsi_lower
        self.rsi_upper = rsi_upper
        self.start = start
        self.end = end
        self.month = month
        self.ti = ti
        self.tc = tc
        self.results = None
        self.get_data()

    def __repr__(self):
        return f"RSIBacktester(symbol={self.symbol}, periods={self.periods}, rsi_lower={self.rsi_lower}, rsi_upper={self.rsi_upper}, start={self.start}, end={self.end}, month={self.month})"

    def get_data(self):
        ''' Retrieves and prepares the data.
        '''
        column_names = ["Date", "Time", "Open", "High", "Low", "Close", "Volume"]
        folder_path = f"C:\\Users\\steve\\AlgoTradingFrx\\csv files\\{self.symbol}\\HISTDATA_COM_MT_{self.symbol}_M12023{self.month}\\"
        raw = pd.read_csv(f"{folder_path}DAT_MT_{self.symbol}_M1_2023{self.month}.csv", sep=',', header=None)
        raw.columns = column_names

        # Convert 1 min chart into other timeframes
        raw = period_converter.PeriodConverter(raw).convert_to_timeframe(self.ti)
        raw = raw[self.start:self.end]
        raw.dropna(inplace=True)

        raw.rename(columns={"Close": "price"}, inplace=True)

        # Calculate logarithmic returns
        raw["returns"] = np.log(raw.price / raw.price.shift(1))

        # Calculate RSI
        raw["U"] = np.where(raw.price.diff() > 0, raw.price.diff(), 0)
        raw["D"] = np.where(raw.price.diff() < 0, raw.price.diff(), 0)
        raw["MA_U"] = raw.U.rolling(self.periods).mean()
        raw["MA_D"] = raw.D.rolling(self.periods).mean()
        raw["RSI"] = raw.MA_U / (raw.MA_U + raw.MA_D) * 100

        self.data = raw

    def set_parameters(self, periods=None, rsi_lower=None, rsi_upper=None):
        ''' Updates RSI parameters and respective time series.
        '''
        if periods is not None:
            self.periods = periods
            self.data["MA_U"] = self.data.U.rolling(self.periods).mean()
            self.data["MA_D"] = self.data.D.rolling(self.periods).mean()
            self.data["RSI"] = self.data.MA_U / (self.data.MA_U + self.data.MA_D) * 100

        if rsi_lower is not None:
            self.rsi_lower = rsi_lower

        if rsi_upper is not None:
            self.rsi_upper = rsi_upper

    def test_strategy(self):
        ''' Backtests the trading strategy.
        '''
        data = self.data.copy().dropna()
        data["position"] = np.where(data.RSI > self.rsi_upper, -1, np.nan)
        data["position"] = np.where(data.RSI < self.rsi_lower, 1, np.nan)
        data.position = data.position.fillna(0)
        data["strategy"] = data["position"].shift(1) * data["returns"]
        data.dropna(inplace=True)
        
        # Determine when a trade takes place
        data["trades"] = data.position.diff().fillna(0).abs()
        
        # Subtract transaction costs from return when trade takes place
        data.strategy = data.strategy - data.trades * self.tc
        
        # Calculate cumulative returns
        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)

        self.results = data

        if not data.empty:
            # Absolute performance of the strategy
            perf = data["cstrategy"].iloc[-1]
            # Out-/underperformance of strategy
            outperf = perf - data["creturns"].iloc[-1]
        else:
            perf = 1  # Absolute performance of the strategy
            outperf = 0  # Out-/underperformance of strategy
        
        return round(perf, 6), round(outperf, 6)

    def plot_results(self):
        ''' Plots the cumulative performance of the trading strategy
        compared to buy and hold.
        '''
        if self.results is None:
            print("No results to plot yet. Run a strategy.")
        else:
            title = f"{self.symbol} | PERIODS = {self.periods} | RSI_LOWER = {self.rsi_lower} | RSI_UPPER = {self.rsi_upper} | TC = {self.tc}"
            self.results[["creturns", "cstrategy"]].plot(title=title, figsize=(12, 8))
            plt.savefig(f'{self.month}_RSI_{self.symbol}.png')

    def update_and_run(self, RSI):
        ''' Updates RSI parameters and returns the negative absolute performance (for minimization algorithm).

        Parameters
        ==========
        RSI : tuple
            RSI parameter tuple
        '''
        self.set_parameters(int(RSI[0]), int(RSI[1]), int(RSI[2]))
        return -self.test_strategy()[0]

    def optimize_parameters(self, periods_range, rsi_lower_range, rsi_upper_range):
        ''' Finds global maximum given the RSI parameter ranges.

        Parameters
        ==========
        periods_range, rsi_lower_range, rsi_upper_range : tuple
            Tuples of the form (start, end, step size)
        '''
        opt = brute(self.update_and_run, (periods_range, rsi_lower_range, rsi_upper_range), finish=None)
        return opt, -self.update_and_run(opt)
