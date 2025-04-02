# Time Series Analysis System

A comprehensive time series analysis tool that performs data analysis, generates visualizations, and provides forecasting capabilities using Prophet and AI-powered insights.

## Features

- Interactive data loading and preprocessing
- Automatic data format detection and correction
- Null value handling with multiple strategies
- Outlier detection using IQR method
- Multiple visualization types:
  - Box plots
  - Distribution plots
  - Monthly box plots
  - Raw data plots
  - Moving averages
  - Forecast visualizations
- Prophet-based time series forecasting
- AI-powered insights generation

## Project Structure

```
Time Series Agent/
├── main.py                     # Main time series analysis script
├── analyze_report.py           # AI-powered analysis script
├── agents/
│   ├── __init__.py
│   └── time_series_agent.py    # Core time series agent
├── data/
│   └── (your data files)      # Your CSV files
├── plots/
│   └── (generated plots)      # Generated visualizations
├── venv/                      # Virtual environment
├── README.md                  # Project documentation
└── requirements.txt           # Project dependencies
```

## Setup

1. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

1. Run the main application:
```bash
python main.py
```

The application will guide you through:
- Loading your CSV data file
- Selecting date and value columns
- Data format verification and correction
- Null value handling
- Outlier detection
- Visualization generation
- Forecasting configuration and execution

2. Generate AI-powered insights (optional):
```bash
python analyze_report.py
```

## Output

The system generates:
- Multiple visualization plots in the `plots/` directory
- Time series forecasts with confidence intervals
- AI-generated insights in `time_series_analysis_insights.md`

## Requirements

- Python 3.8+
- Core dependencies:
  - pandas>=1.5.0
  - numpy>=1.21.0
  - prophet>=1.1.0
  - matplotlib>=3.5.0
  - seaborn>=0.11.0
  - plotly
- AI analysis dependencies:
  - langchain-openai
  - python-dotenv

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 