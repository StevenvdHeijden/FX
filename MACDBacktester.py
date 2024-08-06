import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brute
plt.style.use("seaborn")
import period_converter

class MACDBacktester:
    ''' Class for the vectorized backtesting of MACD-based trading strategies.

    Attributes
    ==========
    symbol : str
        Ticker symbol to work with.
    ema_s : int
        Time window for the fast period EMA.
    ema_l : int
        Time window for the slow period EMA.
    signal_mw : int
        Time window for the signal EMA.
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
        Sets new MACD parameters.
        
    test_strategy:
        Runs the backtest for the MACD-based strategy.
        
    plot_results:
        Plots the performance of the strategy compared to buy and hold.
        
    update_and_run:
        Updates MACD parameters and returns the negative absolute performance (for minimization algorithm).
        
    optimize_parameters:
        Implements a brute force optimization for the MACD parameters.
    '''

    def __init__(self, symbol, ema_s, ema_l, signal_mw, start, end, month, ti, tc):
        self.symbol = symbol
        self.ema_s = ema_s
        self.ema_l = ema_l
        self.signal_mw = signal_mw
        self.start = start
        self.end = end
        self.month = month
        self.ti = ti
        self.tc = tc
        self.results = None
        self.get_data()

    def __repr__(self):
        return f"MACDBacktester(symbol={self.symbol}, ema_s={self.ema_s}, ema_l={self.ema_l}, signal_mw={self.signal_mw}, start={self.start}, end={self.end}, month={self.month})"

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

        # Calculate MACD
        raw["EMA_S"] = raw["price"].ewm(span=self.ema_s, min_periods=self.ema_s).mean()
        raw["EMA_L"] = raw["price"].ewm(span=self.ema_l, min_periods=self.ema_l).mean()
        raw["MACD"] = raw["EMA_S"] - raw["EMA_L"]
        raw["MACD_Signal"] = raw["MACD"].ewm(span=self.signal_mw, min_periods=self.signal_mw).mean()

        self.data = raw

    def set_parameters(self, ema_s=None, ema_l=None, signal_mw=None):
        ''' Updates MACD parameters and respective time series.
        '''
        if ema_s is not None:
            self.ema_s = ema_s
            self.data["EMA_S"] = self.data["price"].ewm(span=self.ema_s, min_periods=self.ema_s).mean()
        
        if ema_l is not None:
            self.ema_l = ema_l
            self.data["EMA_L"] = self.data["price"].ewm(span=self.ema_l, min_periods=self.ema_l).mean()
        
        self.data["MACD"] = self.data["EMA_S"] - self.data["EMA_L"]

        if signal_mw is not None:
            self.signal_mw = signal_mw
        
        self.data["MACD_Signal"] = self.data["MACD"].ewm(span=self.signal_mw, min_periods=self.signal_mw).mean()

    def test_strategy(self):
        ''' Backtests the trading strategy.
        '''
        data = self.data.copy().dropna()
        data["position"] = np.where(data.MACD - data.MACD_Signal > 0, 1, -1)
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
            title = f"{self.symbol} | EMA_S = {self.ema_s} | EMA_L = {self.ema_l} | SIGNAL_MW = {self.signal_mw} | TC = {self.tc}"
            self.results[["creturns", "cstrategy"]].plot(title=title, figsize=(12, 8))
            plt.savefig(f'{self.month}_MACD_{self.symbol}.png')

    def update_and_run(self, MACD):
        ''' Updates MACD parameters and returns the negative absolute performance (for minimization algorithm).

        Parameters
        ==========
        MACD : tuple
            MACD parameter tuple
        '''
        self.set_parameters(int(MACD[0]), int(MACD[1]), int(MACD[2]))
        return -self.test_strategy()[0]

    def optimize_parameters(self, ema_s_range, ema_l_range, signal_mw_range):
        ''' Finds global maximum given the MACD parameter ranges.

        Parameters
        ==========
        ema_s_range, ema_l_range, signal_mw_range : tuple
            Tuples of the form (start, end, step size)
        '''
        opt = brute(self.update_and_run, (ema_s_range, ema_l_range, signal_mw_range), finish=None)
        return opt, -self.update_and_run(opt)
