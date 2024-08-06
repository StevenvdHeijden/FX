# Forex Trading Strategy Tester and Optimizer

## Overview

This Python code is designed to test various trading strategies for the forex market by optimizing their parameters and performing backtesting on user-defined historical data intervals. The user can select which currency pair to use, with the current data sourced from [histdata.com](https://www.histdata.com). It is recommended to use similar data for consistency. The optimization is based on maximizing net returns.

## Features

- **period_converter**: Allows for the user to convert the minute chart loaded from histdata, into whatever period desired
- **MACDBacktester**: Initializes a MACD strategy and optimizes the short EMA, long EMA, and signal line.
- **RSIBacktester**: Initializes an RSI strategy and optimizes the period, lower RSI, and upper RSI levels.
- **PPBacktester**: Initializes a pivot point strategy and optimizes the relevant parameters .
- **BBBacktester**: Initializes a Bollinger Bands strategy and optimizes the Bollinger Bands parameters.
- **FRLBacktester**: Initializes a Fibonacci Retracement strategy and optimizes the Fibonacci levels.
- **SMABacktester**: Initializes a Simple Moving Average (SMA) strategy and optimizes the short SMA and long SMA.
- **EMABacktester**: Initializes an Exponential Moving Average (EMA) strategy and optimizes the short EMA and long EMA.
- **SOBacktester**: Initializes a Stochastic Oscillator strategy and optimizes the stochastic parameters.
- **strategy_indicators**: Assigns limits to the training sets and gathers the optimal parameters based on return for each of abovementioned indicators/strategies.

## Installation

To use this code, you need to have Python installed on your system. You can install the required libraries using the following command:

    pip install -r requirements.txt

## Usage

1. Prepare the Data: Download historical forex data from histdata.com and save it in a compatible format (e.g., CSV).

2. Load the Data: Load your historical data into the code. Ensure that the name of the path matches the one in the code:

        def get_data(self):
        ''' Retrieves and prepares the data.
        '''
            column_names = ["Date","Time","Open","High","Low","Close","Volume"]
            folder_path = f"C:{folderpath_name}\\HISTDATA_COM_MT_{self.symbol}_M12023{self.month}\\"
            raw = pd.read_csv(f"{folder_path}DAT_MT_{self.symbol}_M1_2023{self.month}.csv", sep=',', header=None)

3. Run the Backtester: Initialize and run the backtester for each strategy, this can be done by running **strategy_indicators.py**.

       # When running the code, first initialize the following variables
       months = [] # Define here the inputs of the months within the year (initialized as 2023) that you want to use for the backscoring, in format 'XX' (e.g., ['01,'02','03'] for January, February, March).
       currency_pair = '' # Define the currency pair, in format AAABBB (e.g., EURUSD, AUDJPY, etc.)
       time_interval = '' # Define time interval, in format xmin or xh (e.g., 30min, 1h, etc.)
       ptc = 0.0000xx # Define costs of trading per pip (e.g., 0.00035)

## Example
    # Python
    from MACDBacktester import MACDBacktester as MACD
    from RSIBacktester import RSIBacktester as RSI
    from PPBacktester import PPBacktester as PP
    from BBBacktester import BBBacktester as BB
    from FRLBacktester import FRLBacktester as FRL
    from SMABacktester import SMABacktester as SMA
    from EMABacktester import EMABacktester as EMA
    from SOBacktester import SOBacktester as SO

    # Initialize the parameters needed to load the appropriate data
    months = ['01','02','03','04','05','06']
    currency_pair = 'EURUSD'
    time_interval = '30min'
    ptc = 0.000035

    for month in months:
        # Initialize start date and end date
        start_date = f'2023-{month}-01'
        end_date = f'2023-{month}-30'

        # MACD class initialization
        ema_s_macd = 12
        ema_l_macd = 26
        signal_mw = 9
        macd = MACD(currency_pair, ema_s_macd, ema_l_macd, signal_mw, start_date, end_date, month, time_interval, ptc) # Initialize the MACD with set parameters
        # Optimize parameters + save plot of results
        print('Optimal parameters MACD: [ema_s, ema_l, signal_mw]', macd.optimize_parameters((5, 20, 1),(21,50,1),(5,20,1))) # Optimize the MACD with set ranges
        macd.test_strategy() # Gives the returns of the optimal strategy
        macd.plot_results() # Plots and saves the results of the optimisation
        
        # RSI class initialization
        periods_rsi = 20
        rsi_upper = 70
        rsi_lower = 30
        rsi = RSI(currency_pair, periods_rsi, rsi_lower, rsi_upper, start_date, end_date, month, time_interval, ptc) Initialize the RSI with set parameter
        # Optimize parameters + save plot of results
        print('Optimal parameters RSI: [periods_rsi, rsi_lower, rsi_upper]', rsi.optimize_parameters((5, 20, 1),(20,35,1),(65,80,1))) # Optimize the RSI with set ranges, higher votality
        #print('Optimal parameters RSI: [periods_rsi, rsi_lower, rsi_upper]', rsi.optimize_parameters((20, 50, 1),(15,25,1),(75,85,1))) # Optimize the RSI with set ranges, lower votality
        rsi.test_strategy() # Gives the returns of the optimal strategy
        rsi.plot_results() # Plots and saves the results of the optimisation
    
        # SMA class initialization - V
        sma_s = 30
        sma_l = 80
        sma = SMA(currency_pair, sma_s, sma_l, start_date, end_date, month, time_interval, ptc) # Initialize the SMA with set parameters
        print('Optimal parameters SMA: [sma_s, sma_l]', sma.optimize_parameters((20, 50, 1),(51,150,1),(5,20,1))) # Optimize the SMA with set ranges
        sma.test_strategy() # Gives the returns of the optimal strategy
        sma.plot_results() # Plots and saves the results of the optimisation

        # Similarly, initialize and run other backtesters...
        # ...

## Classes and Methods

### 'MACDBacktester'

- **Methods**:
  - '__init__(data)': Initializes with historical data.
  - optimize_parameters(): Optimizes the short EMA, long EMA, and signal line.

### 'RSIBacktester'

- **Methods**:
  - '__init__(data)': Initializes with historical data.
  - 'optimize_parameters()': Optimizes the period, lower RSI, and upper RSI levels.

### 'PPBacktester'

- **Methods**:
  - '__init__(data)': Initializes with historical data.
  - 'optimize_parameters(): Optimizes the pivot point parameters.

### 'BBBacktester'

- **Methods**:
  - '__init__(data)': Initializes with historical data.
  - 'optimize_parameters()': Optimizes the Bollinger Bands parameters.

### 'FRLBacktester'

- **Methods**:
  - '__init__(data)': Initializes with historical data.
  - 'optimize_parameters()': Optimizes the Fibonacci levels.

### SMABacktester

- **Methods**:
  - '__init__(data)': Initializes with historical data.
  - 'optimize_parameters()': Optimizes the SMA short and SMA long.

### EMABacktester

- **Methods**:
  - '__init__(data)': Initializes with historical data.
  - 'optimize_parameters()': Optimizes the EMA short and EMA long.

### SOBacktester

- **Methods**:
  - __init__(data): Initializes with historical data.
  - optimize_parameters(): Optimizes the stochastic parameters.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any bugs or feature requests.

## Contact

For any questions or inquiries, please contact stevenvdh95@gmail.com.
