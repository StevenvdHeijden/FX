import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brute
plt.style.use("seaborn")
import period_converter
from scipy.signal import argrelextrema

'''
In Fibonacci there are no distinct parameters to optimize. There are just different strategies to apply. 
This backtesting class will check over a longer period of time which strategies work the best. Here it will take into account whic retracement levels to use and which ones correspond the best.

A Fibonnacci Retracement (23.6%) Breakout Strategy
    data["position"] = np.where((data.hh != data.hh.shift()) | (data.ll != data.ll.shift()), 0, np.nan) # Go Neutral when reaching new Highs/lows (e.g. when Trend reverses)
    
    # Downtrend Decisions
    data["position"] = np.where((data.Trend == "Down") & (data.Close.shift() < data["R23.6"].shift()) &  (data.Close > data["R23.6"]), 1, data.position) # Go Long when Price breaks R23.6
    
    data["position"] = np.where((data.Trend == "Down") & (data.Close.shift() < data["R38.2"].shift()) &  (data.Close >= data["R38.2"]), 0, data.position) # Go Neutral when Price reaches/breaks R38.2 (Take Profit)
    
    data["position"] = np.where((data.Trend == "Down") & (data.Close.shift() > data.ll.shift()) &  (data.Close <= data.ll), 0, data.position) # Go Neutral when Price reaches/breaks R0 (Stop Loss)
    
    # Uptrend Decisions
    data["position"] = np.where((data.Trend == "Up") & (data.Close.shift() > data["R23.6"].shift()) &  (data.Close < data["R23.6"]), -1, data.position) # Go Short when Price breaks R23.6
    
    data["position"] = np.where((data.Trend == "Up") & (data.Close.shift() > data["R38.2"].shift()) &  (data.Close <= data["R38.2"]), 0, data.position) # Go Neutral when Price reaches/breaks R38.2 (Take Profit)
    
    data["position"] = np.where((data.Trend == "Up") & (data.Close.shift() < data.hh.shift()) &  (data.Close >= data.hh), 0, data.position) # Go Neutral when Price reaches/breaks R0 (Stop Loss)
    
    data["position"] = data.position.ffill()
'''

class FRLBacktester():
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
        rep = "FRLBacktester(symbol = {}, start = {}, end = {})"
        return rep.format(self.symbol, self.start, self.end)
        
    def get_data(self):
        ''' Retrieves and prepares the data.
        '''
        column_names = ['Date','Open','High','Close','Low','Vol']
        raw = pd.read_csv(f'XRF\\EURUSD\\DAT_MT_{self.symbol}_M1_2023{self.month}.csv', header=None, names=column_names)
        raw['Date'] = pd.to_datetime(raw['Date'])
        raw.set_index('Date', inplace=True)
        raw = raw.loc[self.start:self.end]
        raw.rename(columns={"Close": "price"}, inplace=True)
        raw["returns"] = np.log(raw.price / raw.price.shift(1))
        self.data = raw
        return raw
    
    def highs_lows(self):
        self.data["hh"] = np.nan
        self.data["hh_date"] = np.nan

        self.data["ll"] = np.nan
        self.data["ll_date"] = np.nan

        for bar in range(len(self.data)): # iterating over the bars
            date = self.data.index[bar] # determine the current bar's date

            hh = self.data.iloc[:bar+1].High # get the high column until current bar
            ll = self.data.iloc[:bar+1].Low # get the low column until current bar
            
            # determine all local highs/lows until current bar
            local_max = argrelextrema(hh.values, np.greater_equal, order= self.order)
            local_min = argrelextrema(ll.values, np.less_equal)
            
            # determine the most recent local high/low (price) and add to "hh" column
            self.data.loc[date, "hh"] = self.data.High.values[local_max][-1]
            self.data.loc[date, "ll"] = self.data.Low.values[local_min][-1]
            
            # determine the most recent local high/low (date) and add to "hh_date" column
            self.data.loc[date, "hh_date"] = self.data.index[local_max][-1]
            self.data.loc[date, "ll_date"] = self.data.index[local_min][-1]

            # Identifying Trend (Uptrend / Downtrend)
            self.data["Trend"] = np.where(self.data.hh_date > self.data.ll_date, "Up", "Down")
            self.data.drop(columns = ["hh_date", "ll_date"], inplace= True)

            # Adding Fibonacci Retracement Levels
            self.data[f"R{self.fibonacci1 * 100}"] = np.where(self.data.Trend == "Up", self.data.hh - (self.data.hh-self.data.ll) * self.fibonacci1, self.data.hh - (self.data.hh - self.data.ll) * (1-self.fibonacci1))
            self.data[f"R{self.fibonacci2 * 100}"] = np.where(self.data.Trend == "Up", self.data.hh - (self.data.hh-self.data.ll) * self.fibonacci2, self.data.hh - (self.data.hh - self.data.ll) * (1-self.fibonacci2))
            self.data[f"R{self.fibonacci3 * 100}"] = np.where(self.data.Trend == "Up", self.data.hh - (self.data.hh-self.data.ll) * self.fibonacci3, self.data.hh - (self.data.hh - self.data.ll) * (1-self.fibonacci3))

        

    #def fib_236_strategy(self):
        