from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from datetime import datetime

# Load environment variables
load_dotenv()

def get_data_summary():
    """Get summary of the data and analysis results."""
    # Read the data
    data = pd.read_csv('data/example.csv')
    data['ds'] = pd.to_datetime(data['ds'])
    data['y'] = pd.to_numeric(data['y'], errors='coerce')  # Convert to numeric, handle errors gracefully
    
    # Calculate basic statistics
    stats = {
        'total_observations': len(data),
        'date_range': f"{data['ds'].min().strftime('%Y-%m-%d')} to {data['ds'].max().strftime('%Y-%m-%d')}",
        'max_value': float(data['y'].max()),
        'min_value': float(data['y'].min()),
        'mean_value': float(data['y'].mean()),
        'median_value': float(data['y'].median()),
        'mode_value': float(data['y'].mode().iloc[0]),
        'null_values': int(data['y'].isnull().sum()),
        'std_value': float(data['y'].std()),
        'skew_value': float(data['y'].skew()),
        'kurtosis_value': float(data['y'].kurtosis())
    }
    
    # Calculate outliers
    Q1 = data['y'].quantile(0.25)
    Q3 = data['y'].quantile(0.75)
    IQR = Q3 - Q1
    outliers = data[(data['y'] < (Q1 - 1.5 * IQR)) | (data['y'] > (Q3 + 1.5 * IQR))]
    stats['outlier_count'] = len(outliers)
    stats['outlier_percentage'] = float((len(outliers) / len(data)) * 100)
    
    # Calculate monthly statistics
    data['month'] = data['ds'].dt.month
    monthly_stats = data.groupby('month')['y'].agg(['mean', 'std', 'count']).round(2).to_dict()
    
    # Calculate trend
    data['year'] = data['ds'].dt.year
    yearly_stats = data.groupby('year')['y'].mean().round(2).to_dict()
    
    # Calculate seasonality
    data['month'] = data['ds'].dt.month
    monthly_means = data.groupby('month')['y'].mean().round(2).to_dict()
    
    return stats, monthly_stats, yearly_stats, monthly_means

def analyze_report():
    # Initialize OpenAI with higher max tokens and latest model
    llm = OpenAI(
        temperature=0.7,
        max_tokens=2000,
        model="gpt-3.5-turbo-instruct"
    )
    
    # Get data summary
    stats, monthly_stats, yearly_stats, monthly_means = get_data_summary()
    
    # Create prompt template
    prompt_template = PromptTemplate(
        input_variables=["stats", "monthly_stats", "yearly_stats", "monthly_means"],
        template="""
        You are an expert data analyst. Please analyze the following time series data and provide detailed insights:

        Data Summary:
        - Total Observations: {stats[total_observations]}
        - Date Range: {stats[date_range]}
        - Maximum Value: {stats[max_value]:.2f}
        - Minimum Value: {stats[min_value]:.2f}
        - Mean Value: {stats[mean_value]:.2f}
        - Median Value: {stats[median_value]:.2f}
        - Mode Value: {stats[mode_value]:.2f}
        - Standard Deviation: {stats[std_value]:.2f}
        - Skewness: {stats[skew_value]:.2f}
        - Kurtosis: {stats[kurtosis_value]:.2f}
        - Number of Null Values: {stats[null_values]}
        - Number of Outliers: {stats[outlier_count]} ({stats[outlier_percentage]:.1f}%)

        Monthly Statistics:
        {monthly_stats}

        Yearly Trends:
        {yearly_stats}

        Monthly Patterns:
        {monthly_means}

        Please provide a comprehensive analysis covering:

        1. Overall Data Characteristics
           - Key statistics and their implications
           - Data quality assessment
           - Notable patterns in the data

        2. Time Series Analysis
           - Trend analysis and its significance
           - Seasonality patterns and their strength
           - Monthly and yearly patterns

        3. Distribution and Pattern Analysis
           - Distribution characteristics
           - Monthly patterns and seasonal effects
           - Year-over-year changes

        4. Outlier Analysis
           - Impact of outliers on the analysis
           - Potential causes of outliers
           - Recommendations for handling outliers

        5. Recommendations
           - Suggestions for further analysis
           - Potential improvements to the model
           - Business implications and actionable insights

        Format your response in a clear, professional manner with appropriate sections and bullet points.
        Use specific numbers from the data to support your analysis.
        Be thorough but concise.
        """
    )
    
    # Generate analysis
    prompt = prompt_template.format(stats=stats, monthly_stats=monthly_stats, 
                                  yearly_stats=yearly_stats, monthly_means=monthly_means)
    
    try:
        analysis = llm.invoke(prompt)
        return analysis
    except Exception as e:
        print(f"Error generating analysis: {str(e)}")
        return None

def save_analysis(analysis):
    if analysis is not None:
        with open('time_series_analysis_insights.md', 'w', encoding='utf-8') as f:
            f.write(analysis)
        print("Analysis has been saved to 'time_series_analysis_insights.md'")
    else:
        print("No analysis to save.")

def main():
    # Generate analysis
    analysis = analyze_report()
    
    # Save analysis
    save_analysis(analysis)

if __name__ == "__main__":
    main() 