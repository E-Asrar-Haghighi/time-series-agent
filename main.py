import os
from agents.time_series_agent import TimeSeriesAgent

def main():
    try:
        agent = TimeSeriesAgent()
        
        # Get file path from user
        file_path = input("\nPlease provide the path to your CSV file: ")
        
        print("\nLoading data...")
        try:
            data_preview = agent.load_data(file_path)
            print("\nData Preview (First 5 rows):")
            print(data_preview)
        except Exception as e:
            print(f"\nError loading data: {str(e)}")
            return
        
        # Select columns
        print("\nPlease select the columns:")
        date_column = input("Enter the name of the date column: ")
        value_column = input("Enter the name of the value column: ")
        try:
            agent.select_columns(date_column, value_column)
        except ValueError as e:
            print(f"\nError selecting columns: {str(e)}")
            return
        
        # Check and correct data formats
        try:
            date_type, value_type = agent.check_formats()
            print("\nCurrent data formats:")
            print(f"Date column format: {date_type}")
            print(f"Value column format: {value_type}")
        except Exception as e:
            print(f"\nError checking formats: {str(e)}")
            return
        
        # Check if formats are appropriate
        date_is_appropriate = date_type == 'datetime64[ns]'
        value_is_appropriate = value_type in ['float64', 'int64']
        
        if not (date_is_appropriate and value_is_appropriate):
            print("\nThe data formats are not appropriate for time series analysis.")
            print("Date column should be datetime format")
            print("Value column should be numeric format (float or integer)")
            
            while True:
                choice = input("\nWould you like to correct the formats? (yes/no): ")
                if choice.lower() == 'yes':
                    try:
                        agent.correct_formats()
                        # Check and print new formats
                        new_date_type, new_value_type = agent.check_formats()
                        print("\nNew data formats:")
                        print(f"Date column format: {new_date_type}")
                        print(f"Value column format: {new_value_type}")
                        break
                    except ValueError as e:
                        print(f"\nError correcting formats: {str(e)}")
                        return
                elif choice.lower() == 'no':
                    print("\nIt is not possible to continue with inappropriate data formats.")
                    while True:
                        exit_choice = input("Would you like to exit or try changing the data format? (exit/change): ")
                        if exit_choice.lower() == 'exit':
                            print("Exiting the program. Please ensure your data is in the correct format before trying again.")
                            return
                        elif exit_choice.lower() == 'change':
                            try:
                                agent.correct_formats()
                                new_date_type, new_value_type = agent.check_formats()
                                print("\nNew data formats:")
                                print(f"Date column format: {new_date_type}")
                                print(f"Value column format: {new_value_type}")
                                break
                            except ValueError as e:
                                print(f"\nError correcting formats: {str(e)}")
                                return
                        else:
                            print("Please enter either 'exit' or 'change'")
                    if exit_choice.lower() == 'exit':
                        return
                    break
                else:
                    print("Please enter either 'yes' or 'no'")
        else:
            print("\nData formats are appropriate for time series analysis.")
        
        # Get data info
        try:
            info = agent.get_data_info()
            print("\nData Shape:", info['shape'])
            print("Null Values:")
            for col, count in info['nulls'].items():
                print(f"{col}: {count}")
        except Exception as e:
            print(f"\nError getting data info: {str(e)}")
            return
        
        if any(info['nulls'].values()):
            print("\nNull values detected. How would you like to proceed?")
            print("1. Correct null values")
            print("2. Continue with null values")
            choice = input("\nEnter your choice (1 or 2): ")
            
            try:
                if choice == '1':
                    agent.handle_null_values('correct')
                else:
                    agent.handle_null_values('continue')
            except ValueError as e:
                print(f"\nError handling null values: {str(e)}")
                return
        
        # Get basic statistics
        try:
            stats = agent.get_basic_stats()
            print("\nBasic Statistics:")
            for stat_name, value in stats.items():
                print(f"{stat_name}: {value:.2f}")
        except Exception as e:
            print(f"\nError calculating basic statistics: {str(e)}")
            return
        
        # Detect outliers
        try:
            outliers = agent.detect_outliers()
            print(f"\nNumber of outliers detected: {len(outliers)}")
        except Exception as e:
            print(f"\nError detecting outliers: {str(e)}")
            return
        
        # Plot visualizations
        print("\nGenerating plots...")
        try:
            agent.plot_boxplot()
            agent.plot_distribution()
            agent.plot_monthly_boxplot()
            agent.plot_raw_data()
            agent.plot_moving_averages()
        except Exception as e:
            print(f"\nError generating plots: {str(e)}")
            return
        
        # Set forecast periods
        try:
            periods = int(input("\nEnter the number of periods to forecast: "))
            agent.set_forecast_periods(periods)
        except ValueError as e:
            print(f"\nError setting forecast periods: {str(e)}")
            return
        
        # Prepare data for Prophet
        try:
            prophet_data = agent.prepare_prophet_data()
        except Exception as e:
            print(f"\nError preparing Prophet data: {str(e)}")
            return
        
        # Train Prophet model and make predictions
        try:
            agent.train_prophet_model()
            forecast = agent.make_forecast()
        except Exception as e:
            print(f"\nError training model and making predictions: {str(e)}")
            return
        
        # Display forecast table
        print("\nForecast Results:")
        print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
        
        # Plot forecast
        try:
            agent.plot_forecast()
        except Exception as e:
            print(f"\nError plotting forecast: {str(e)}")
            return
        
        print("\nForecasting complete!")
        
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        print("Please check your data format and try again.")

if __name__ == "__main__":
    main()
