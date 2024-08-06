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

    # Initialize and run the MACD backtester
    macd_tester = MACDBacktester(data)
    macd_tester.optimize_parameters()

    # Initialize and run the RSI backtester
    rsi_tester = RSIBacktester(data)
    rsi_tester.optimize_parameters()

    # Similarly, initialize and run other backtesters...
    # ...

    # Combine strategies and determine optimal weights
    strategy_indicators = StrategyIndicators([macd_tester, rsi_tester, ...])
    optimal_strategy = strategy_indicators.determine_optimal_strategy()

    # Run the combined strategy
    optimal_strategy.run_backtest()

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
  - 'optimize_parameters()': Optimizes the SMA period.

### EMABacktester

- **Methods**:
  - '__init__(data)': Initializes with historical data.
  - 'optimize_parameters()': Optimizes the EMA period.

### SOBacktester

- **Methods**:
  - __init__(data): Initializes with historical data.
  - optimize_parameters(): Optimizes the stochastic parameters.

### StrategyIndicators

- **Methods**:
  - __init__(strategies): Initializes with a list of strategy instances.
  - determine_optimal_strategy(): Combines strategies and determines the optimal weights.
  - run_backtest(): Runs the backtest with the combined strategy.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any bugs or feature requests.

## Contact

For any questions or inquiries, please contact stevenvdh95@gmail.com.
