import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brute
plt.style.use("seaborn")
import period_converter
from scipy.signal import argrelextrema

class FRLBacktester():
    ''' Class for the vectorized backtesting of Fibonacci-based trading strategies.

    Attributes
    ==========
    symbol : str
        Ticker symbol to work with.
    order : int
        Order parameter for identifying local highs and lows.
    fibonacci1 : float
        First Fibonacci retracement level (e.g., 0.236).
    fibonacci2 : float
        Second Fibonacci retracement level (e.g., 0.382).
    fibonacci3 : float
        Third Fibonacci retracement level (e.g., 0.618).
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
        
    highs_lows:
        Identifies local highs and lows and determines trend.
        
    test_strategy:
        Runs the backtest for the Fibonacci-based strategy.
        
    plot_results:
        Plots the performance of the strategy compared to buy and hold.
        
    update_and_run:
        Updates Fibonacci retracement levels and returns the negative absolute performance (for minimization algorithm).
        
    optimize_parameters:
        Implements a brute force optimization for the Fibonacci retracement levels.
    '''

    def __init__(self, symbol, order, fibonacci1, fibonacci2, fibonacci3, start, end, month, time_interval, tc):
        self.symbol = symbol
        self.order = order
        self.fibonacci1 = fibonacci1
        self.fibonacci2 = fibonacci2
        self.fibonacci3 = fibonacci3
        self.start = start
        self.end = end
        self.month = month
        self.time_interval = time_interval
        self.tc = tc
        self.results = None
        self.get_data()
        
    def __repr__(self):
        return f"FRLBacktester(symbol={self.symbol}, order={self.order}, fibonacci1={self.fibonacci1}, fibonacci2={self.fibonacci2}, fibonacci3={self.fibonacci3}, start={self.start}, end={self.end}, month={self.month})"
        
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
    
    def highs_lows(self):
        ''' Identifies local highs and lows and determines trend.
        '''
        self.data["hh"] = np.nan
        self.data["hh_date"] = np.nan
        self.data["ll"] = np.nan
        self.data["ll_date"] = np.nan

        for bar in range(len(self.data)):
            date = self.data.index[bar]
            hh = self.data.iloc[:bar+1].High
            ll = self.data.iloc[:bar+1].Low
            
            local_max = argrelextrema(hh.values, np.greater_equal, order=self.order)
            local_min = argrelextrema(ll.values, np.less_equal, order=self.order)
            
            if len(local_max[0]) > 0:
                self.data.loc[date, "hh"] = hh[local_max][-1]
                self.data.loc[date, "hh_date"] = self.data.index[local_max][-1]
            
            if len(local_min[0]) > 0:
                self.data.loc[date, "ll"] = ll[local_min][-1]
                self.data.loc[date, "ll_date"] = self.data.index[local_min][-1]

        self.data["Trend"] = np.where(self.data.hh_date > self.data.ll_date, "Up", "Down")
        self.data.drop(columns=["hh_date", "ll_date"], inplace=True)

        self.data[f"R{self.fibonacci1 * 100}"] = np.where(
            self.data.Trend == "Up", 
            self.data.hh - (self.data.hh - self.data.ll) * self.fibonacci1, 
            self.data.hh - (self.data.hh - self.data.ll) * (1 - self.fibonacci1)
        )
        self.data[f"R{self.fibonacci2 * 100}"] = np.where(
            self.data.Trend == "Up", 
            self.data.hh - (self.data.hh - self.data.ll) * self.fibonacci2, 
            self.data.hh - (self.data.hh - self.data.ll) * (1 - self.fibonacci2)
        )
        self.data[f"R{self.fibonacci3 * 100}"] = np.where(
            self.data.Trend == "Up", 
            self.data.hh - (self.data.hh - self.data.ll) * self.fibonacci3, 
            self.data.hh - (self.data.hh - self.data.ll) * (1 - self.fibonacci3)
        )
    
    def test_strategy(self):
        ''' Backtests the Fibonacci retracement strategy.
        '''
        self.highs_lows()
        data = self.data.copy().dropna()

        data["position"] = np.nan

        # Neutral when reaching new Highs/lows
        data["position"] = np.where((data.hh != data.hh.shift()) | (data.ll != data.ll.shift()), 0, data.position)

        # Downtrend Decisions
        data["position"] = np.where(
            (data.Trend == "Down") & (data.price.shift() < data[f"R{self.fibonacci1 * 100}"].shift()) & (data.price > data[f"R{self.fibonacci1 * 100}"]),
            1, data.position
        )  # Long when Price breaks R23.6

        data["position"] = np.where(
            (data.Trend == "Down") & (data.price.shift() < data[f"R{self.fibonacci2 * 100}"].shift()) & (data.price >= data[f"R{self.fibonacci2 * 100}"]),
            0, data.position
        )  # Neutral when Price reaches/breaks R38.2 (Take Profit)

        data["position"] = np.where(
            (data.Trend == "Down") & (data.price.shift() > data.ll.shift()) & (data.price <= data.ll),
            0, data.position
        )  # Neutral when Price reaches/breaks R0 (Stop Loss)

        # Uptrend Decisions
        data["position"] = np.where(
            (data.Trend == "Up") & (data.price.shift() > data[f"R{self.fibonacci1 * 100}"].shift()) & (data.price < data[f"R{self.fibonacci1 * 100}"]),
            -1, data.position
        )  # Short when Price breaks R23.6

        data["position"] = np.where(
            (data.Trend == "Up") & (data.price.shift() > data[f"R{self.fibonacci2 * 100}"].shift()) & (data.price <= data[f"R{self.fibonacci2 * 100}"]),
            0, data.position
        )  # Neutral when Price reaches/breaks R38.2 (Take Profit)

        data["position"] = np.where(
            (data.Trend == "Up") & (data.price.shift() < data.hh.shift()) & (data.price >= data.hh),
            0, data.position
        )  # Neutral when Price reaches/breaks R0 (Stop Loss)

        data["position"] = data.position.ffill().fillna(0)
        data["strategy"] = data["position"].shift(1) * data["returns"]
        
        data["trades"] = data["position"].diff().fillna(0).abs()
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
        ''' Plots the cumulative performance of the trading strategy
        compared to buy and hold.
        '''
        if self.results is None:
            print("No results to plot yet. Run a strategy.")
        else:
            title = f"{self.symbol} | Order = {self.order} | Fibo Levels = {self.fibonacci1}, {self.fibonacci2}, {self.fibonacci3} | TC = {self.tc}"
            self.results[["creturns", "cstrategy"]].plot(title=title, figsize=(12, 8))
            plt.savefig(f'{self.month}_FRL_{self.symbol}.png')

    def update_and_run(self, levels):
        ''' Updates Fibonacci retracement levels and returns the negative absolute performance (for minimization algorithm).

        Parameters
        ==========
        levels : tuple
            Fibonacci levels tuple
        '''
        self.fibonacci1, self.fibonacci2, self.fibonacci3 = levels
        return -self.test_strategy()[0]

    def optimize_parameters(self, fibo1_range, fibo2_range, fibo3_range):
        ''' Finds global maximum given the Fibonacci level ranges.

        Parameters
        ==========
        fibo1_range, fibo2_range, fibo3_range : tuple
            Tuples of the form (start, end, step size)
        '''
        opt = brute(self.update_and_run, (fibo1_range, fibo2_range, fibo3_range), finish=None)
        return opt, -self.update_and_run(opt)
