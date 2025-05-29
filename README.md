# Medical Article Evaluation Visualizer

This is a Streamlit application that visualizes medical article evaluation data from JSON files. It provides an interactive interface to explore different aspects of medical articles and their evaluations.

## Features

- Display basic article information (title, author, journal, year, PMID)
- Interactive aspect selector for different article components
- Radar charts showing completeness and conciseness evaluations
- Detailed metrics display for each aspect
- Support for various medical article aspects (Objective, Intervention, Comparator, etc.)

## Requirements

- Python 3.7+
- Streamlit
- Pandas
- Plotly

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Make sure your JSON file is in the same directory as the application
2. Run the Streamlit app:
```bash
streamlit run app.py
```

## Using the Application

1. The application will automatically load the article data
2. Use the aspect selector to choose which aspect of the article to analyze
3. View the summary and evaluation metrics for each aspect
4. Explore the radar chart showing completeness and conciseness evaluations

## Supported Aspects

- Objective
- Intervention
- Comparator
- Blinding Method
- Population
- Medicines
- Treatment Duration
- Primary Endpoints
- Follow-up Duration
- Outcomes
- Findings
- Reference

Each aspect includes evaluations for:
- Summary
- Sentences
- Key Phrases

Rated on both Completeness and Conciseness (1-5 scale)