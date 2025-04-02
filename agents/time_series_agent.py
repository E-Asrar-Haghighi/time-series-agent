import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from datetime import datetime
import plotly.graph_objects as go
from typing import Tuple, Optional
import os

class TimeSeriesAgent:
    def __init__(self):
        self.data = None
        self.date_column = None
        self.value_column = None
        self.forecast_periods = None
        self.prophet_model = None
        self.forecast_results = None
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load CSV data and return first 5 rows for preview."""
        try:
            # Read CSV with explicit data types
            self.data = pd.read_csv(file_path, dtype={
                'ds': str,  # Read date as string initially
                'y': str    # Read value as string initially
            })
            return self.data.head()
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def select_columns(self, date_column: str, value_column: str) -> None:
        """Select and validate date and value columns."""
        if date_column not in self.data.columns or value_column not in self.data.columns:
            raise ValueError("One or both columns not found in the dataset")
        
        self.date_column = date_column
        self.value_column = value_column
        
        # Extract selected columns
        self.data = self.data[[date_column, value_column]].copy()
    
    def check_formats(self) -> Tuple[str, str]:
        """Check and return data types of date and value columns."""
        date_type = str(self.data[self.date_column].dtype)
        value_type = str(self.data[self.value_column].dtype)
        return date_type, value_type
    
    def correct_formats(self) -> None:
        """Convert date to datetime and value to numeric if needed."""
        try:
            # Convert date column to datetime
            self.data[self.date_column] = pd.to_datetime(self.data[self.date_column])
            # Ensure date format is YYYY-MM-DD
            self.data[self.date_column] = self.data[self.date_column].dt.strftime('%Y-%m-%d')
            self.data[self.date_column] = pd.to_datetime(self.data[self.date_column])
            
            # Convert value column to numeric
            # First, clean any quotes or whitespace
            self.data[self.value_column] = self.data[self.value_column].str.strip().str.strip('"\'')
            
            # Replace 'Null' with NaN
            self.data[self.value_column] = self.data[self.value_column].replace('Null', np.nan)
            
            # Convert to numeric
            self.data[self.value_column] = pd.to_numeric(self.data[self.value_column], errors='coerce')
            
            # Check if conversion was successful
            if self.data[self.value_column].isnull().all():
                raise ValueError("Unable to convert value column to numeric format. Please check your data.")
                
        except Exception as e:
            raise ValueError(f"It is not possible to proceed with the current data format. Please correct your data format and try again. Error: {str(e)}")
    
    def get_data_info(self) -> dict:
        """Get data shape and null counts."""
        return {
            'shape': self.data.shape,
            'nulls': self.data.isnull().sum().to_dict()
        }
    
    def handle_null_values(self, action: str) -> None:
        """Handle null values based on user choice."""
        if action.lower() == 'correct':
            print("\nPlease choose how to handle null values:")
            print("1. Remove rows with null values")
            print("2. Replace with mean")
            print("3. Replace with median")
            print("4. Replace with mode")
            
            while True:
                try:
                    choice = int(input("\nEnter your choice (1-4): "))
                    if choice not in [1, 2, 3, 4]:
                        print("Please enter a number between 1 and 4")
                        continue
                    break
                except ValueError:
                    print("Please enter a valid number")
            
            if choice == 1:
                # Remove rows with null values
                self.data = self.data.dropna()
                print("Null values have been removed from the dataset.")
            elif choice == 2:
                # Replace with mean
                mean_value = self.data[self.value_column].mean()
                self.data[self.value_column] = self.data[self.value_column].fillna(mean_value)
                print(f"Null values have been replaced with mean value: {mean_value:.2f}")
            elif choice == 3:
                # Replace with median
                median_value = self.data[self.value_column].median()
                self.data[self.value_column] = self.data[self.value_column].fillna(median_value)
                print(f"Null values have been replaced with median value: {median_value:.2f}")
            elif choice == 4:
                # Replace with mode
                mode_value = self.data[self.value_column].mode().iloc[0]
                self.data[self.value_column] = self.data[self.value_column].fillna(mode_value)
                print(f"Null values have been replaced with mode value: {mode_value:.2f}")
            
            # Show updated null value count
            null_count = self.data[self.value_column].isnull().sum()
            print(f"\nCurrent null value count: {null_count}")
            
        elif action.lower() == 'continue':
            print("Continuing with null values in the dataset.")
        else:
            raise ValueError("Invalid action. Please choose 'correct' or 'continue'.")
    
    def get_basic_stats(self) -> dict:
        """Calculate basic statistics."""
        return {
            'max': self.data[self.value_column].max(),
            'min': self.data[self.value_column].min(),
            'mean': self.data[self.value_column].mean(),
            'median': self.data[self.value_column].median(),
            'mode': self.data[self.value_column].mode().iloc[0]
        }
    
    def detect_outliers(self) -> pd.DataFrame:
        """Detect outliers using IQR method."""
        Q1 = self.data[self.value_column].quantile(0.25)
        Q3 = self.data[self.value_column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = self.data[
            (self.data[self.value_column] < lower_bound) | 
            (self.data[self.value_column] > upper_bound)
        ]
        return outliers
    
    def plot_boxplot(self) -> None:
        """Plot boxplot to visualize outliers."""
        plt.figure(figsize=(10, 6))
        sns.boxplot(y=self.data[self.value_column])
        plt.title(f'Boxplot of {self.value_column}')
        plt.ylabel(self.value_column)
        plt.savefig('plots/boxplot.png')
        plt.close()
    
    def plot_distribution(self) -> None:
        """Plot distribution with histogram and KDE."""
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.data, x=self.value_column, kde=True)
        plt.title(f'Distribution of {self.value_column}')
        plt.savefig('plots/distribution.png')
        plt.close()
    
    def plot_raw_data(self) -> None:
        """Plot raw data with summary lines."""
        plt.figure(figsize=(15, 7))
        plt.plot(self.data[self.date_column], self.data[self.value_column], label='Raw Data')
        
        stats = self.get_basic_stats()
        for stat_name, stat_value in stats.items():
            plt.axhline(y=stat_value, color='r', linestyle='--', label=f'{stat_name}: {stat_value:.2f}')
        
        plt.title(f'Raw Data with Summary Lines')
        plt.xlabel('Date')
        plt.ylabel(self.value_column)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('plots/raw_data.png')
        plt.close()
    
    def plot_moving_averages(self) -> None:
        """Plot raw data with moving averages."""
        plt.figure(figsize=(15, 7))
        plt.plot(self.data[self.date_column], self.data[self.value_column], label='Raw Data', alpha=0.5)
        
        # Calculate moving averages
        ma10 = self.data[self.value_column].rolling(window=10).mean()
        ma50 = self.data[self.value_column].rolling(window=50).mean()
        
        plt.plot(self.data[self.date_column], ma10, label='10-day MA', alpha=0.8)
        plt.plot(self.data[self.date_column], ma50, label='50-day MA', alpha=0.8)
        
        plt.title(f'Raw Data with Moving Averages')
        plt.xlabel('Date')
        plt.ylabel(self.value_column)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('plots/moving_averages.png')
        plt.close()
    
    def set_forecast_periods(self, periods: int) -> None:
        """Set the number of periods to forecast."""
        if periods <= 0:
            raise ValueError("Number of periods must be positive")
        self.forecast_periods = periods
    
    def prepare_prophet_data(self) -> pd.DataFrame:
        """Prepare data in the format required by Prophet."""
        prophet_data = self.data.copy()
        prophet_data.columns = ['ds', 'y']  # Prophet requires these column names
        return prophet_data
    
    def train_prophet_model(self) -> None:
        """Train the Prophet model."""
        self.prophet_model = Prophet()
        prophet_data = self.prepare_prophet_data()
        self.prophet_model.fit(prophet_data)
    
    def make_forecast(self) -> pd.DataFrame:
        """Generate forecast using the trained Prophet model."""
        if self.prophet_model is None:
            raise ValueError("Model has not been trained yet")
            
        if self.forecast_periods is None:
            raise ValueError("Number of forecast periods not set. Please set forecast periods first.")
            
        future_dates = self.prophet_model.make_future_dataframe(periods=self.forecast_periods)
        self.forecast_results = self.prophet_model.predict(future_dates)
        return self.forecast_results
    
    def plot_monthly_boxplot(self) -> None:
        """Plot monthly boxplot to visualize seasonal patterns."""
        # Extract month from date
        self.data['month'] = self.data[self.date_column].dt.month
        
        plt.figure(figsize=(15, 7))
        sns.boxplot(data=self.data, x='month', y=self.value_column)
        plt.title('Monthly Distribution of Values')
        plt.xlabel('Month')
        plt.ylabel(self.value_column)
        
        # Add month names to x-axis
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        plt.xticks(range(12), month_names)
        
        plt.tight_layout()
        plt.savefig('plots/monthly_boxplot.png')
        plt.close()
        
        # Remove the month column after plotting
        self.data = self.data.drop('month', axis=1)

    def plot_forecast(self) -> None:
        """Plot the forecast results."""
        if self.forecast_results is None:
            raise ValueError("No forecast results available")
            
        # Plot forecast
        plt.figure(figsize=(15, 7))
        plt.plot(self.data[self.date_column], self.data[self.value_column], label='Actual', color='blue', alpha=0.5)
        plt.plot(self.forecast_results['ds'], self.forecast_results['yhat'], label='Forecast', color='red')
        plt.fill_between(self.forecast_results['ds'], 
                        self.forecast_results['yhat_lower'], 
                        self.forecast_results['yhat_upper'], 
                        color='red', alpha=0.2, label='Confidence Interval')
        plt.title('Time Series with Forecast')
        plt.xlabel('Date')
        plt.ylabel(self.value_column)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('plots/forecast.png')
        plt.close()
        
        # Plot trend
        plt.figure(figsize=(15, 7))
        plt.plot(self.forecast_results['ds'], self.forecast_results['trend'], label='Trend', color='green')
        plt.title('Trend Component')
        plt.xlabel('Date')
        plt.ylabel(self.value_column)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('plots/trend.png')
        plt.close()
        
        # Plot yearly seasonality
        plt.figure(figsize=(15, 7))
        plt.plot(self.forecast_results['ds'], self.forecast_results['yearly'], label='Yearly Seasonality', color='blue')
        plt.title('Yearly Seasonality Component')
        plt.xlabel('Date')
        plt.ylabel(self.value_column)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('plots/seasonality.png')
        plt.close() 