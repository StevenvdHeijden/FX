# Forex Trading Strategy Tester and Optimizer

## Overview

This Python code is designed to test various trading strategies for the forex market by optimizing their parameters and performing backtesting on user-defined historical data intervals. The user can select which currency pair to use, with the current data sourced from [histdata.com](https://www.histdata.com). It is recommended to use similar data for consistency. The optimization is based on maximizing net returns.

## Features

- **period_converter**: Allows for the user to convert the minute chart loaded from histdata, into whatever period desired
- **MACDBacktester**: Initializes a MACD strategy and optimizes the short EMA, long EMA, and signal line.
- **RSIBacktester**: Initializes an RSI strategy and optimizes the period, lower RSI, and upper RSI levels.
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
            
            # Similarly, initialize and run other backtesters...
            # ...

            # Consolidate the outputs into the final table
            output = pd.DataFrame(table, index=[0])
            all_months_output = pd.concat([all_months_output, output], ignore_index=True)
            print(all_months_output)

        # Save all results to a CSV file
        all_months_output.to_csv('indicator_Backtesting_JuneNovember.csv', sep=';', decimal=',', index=False)

## Classes and Methods

### 'MACDBacktester'

- **Methods**:
  - '__init__(data)': Initializes with historical data.
  - optimize_parameters(): Optimizes the short EMA, long EMA, and signal line.

### 'RSIBacktester'

- **Methods**:
  - '__init__(data)': Initializes with historical data.
  - 'optimize_parameters()': Optimizes the period, lower RSI, and upper RSI levels.

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
