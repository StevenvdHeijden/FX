'''
First strategy that is truly a Price Action-strategy using support and resistance lines. Which is also the challenging part here. Many traders
believe that this is primarily driven by traders intuition and judgement.
a. Data Preparation (advanced Pandas)
Pivot Point is an Intraday Price Action Strategy
It uses (Open), High, Low and Close Prices of the previous day
i. Time zone Conversion
print(data.index.tz) # print the current time zone applied to the current dataset
data = data.tz_localize("UTC") # localize to UTC time
data = data.tz_convert("US/Eastern") # convert to US/Eastern (NY) time
close = data.Close.to_frame().copy()
close.resample("D", offset = "17H").last().dropna() # resample to daily
ii. OHLC Resampling
agg_dict = {"Open":"first",
"High":"max",
"Low":"min",
"Close":"last"
}
daily_data = data.resample("D", offset = "17H").agg(agg_dict).dropna()
daily_data.columns = ["Open_d", "High_d", "Low_d", "Close_d"]
daily_data.shift().dropna()
pd.concat([data, daily_data.shift().dropna()], axis = 1).ffill().dropna()
iii. Merging Intraday and Daily Data:

b. Adding Pivot Point and Support and Resistance Lines:
Pivot Point Line: The average of the previous day's High, Low and Close price
data["PP"] = (data.High_d + data.Low_d + data.Close_d) / 3
S1 and S2 Support Lines
data["S1"] = data.PP * 2 - data.High_d
data["S2"] = data.PP - (data.High_d - data.Low_d)
R1 and R2 Resistance Lines
data["R1"] = data.PP * 2 - data.Low_d
data["R2"] = data.PP + (data.High_d - data.Low_d)
c. Defining a simple Pivot Point Strategy:
Frx
Tuesday, 12 December 2023 15:12

Random notes Page 1

Defining a simple Pivot Point Strategy:
There is not the one Pivot Point Strategy that is set in stone.
Working with Price Action and Support and Resistance Lines is highly subjective and case specific.
Subject to human intuition, judgement and experience -> Can be used for Algorithmic Trading?
But the consensus is that
□ a price above PP signals bullish sentiment -> go long until resistance (R1) line met
□ a price below PP signals bearish sentiment -> go short until support (S1) line met
data["position"] = np.where(data.Open > data.PP, 1, -1)
data["position"] = np.where(data.Open >= data.R1, 0, data.position)
data["position"] = np.where(data.Open <= data.S1, 0, data.position)
data["position"] = data.position.fillna(0)
c.

d. Vectorized Strategy Backtesting:
Note that we used the Open prices instead of the Close prices when defining the positions
data["returns"] = np.log(data.Open.shift(-1).div(data.Open)) # use Open instead of Close
data["strategy"] = data.position * data["returns"] # no shift needed here
data.dropna(inplace = True)
'''
class PPBacktester:
    {


}