import pandas as pd

class PeriodConverter:
    def __init__(self, data):
        self.data = data

    def convert_to_timeframe(self, timeframe):
        """
        Convert forex data to the specified timeframe.

        Parameters:
        - timeframe (str): Target timeframe (e.g., '30m', '1h', '1d', '1w').

        Returns:
        - pd.DataFrame: Converted dataframe with the specified timeframe.
        """
        if timeframe not in ['30min', '1h', '1d', '1w']:
            raise ValueError("Invalid timeframe. Supported timeframes: '30min', '1h', '1d', '1w'.")

        # Convert 'Date' and 'Time' columns to datetime and set as index
        datetime_index = pd.to_datetime(self.data['Date'] + ' ' + self.data['Time'])
        self.data.set_index(datetime_index, inplace=True)

        # Resample data based on the specified timeframe
        ohlc_dict = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }
        resampled_data = self.data.resample(timeframe).apply(ohlc_dict)

        return resampled_data