# ATP Tennis Match Prediction

## Project Overview
This project focuses on applying Machine Learning techniques to predict the outcomes of ATP tennis matches. The primary objective is to analyze historical data, identify key performance indicators, and build predictive models that can determine the winner based on pre-match information.

## Dataset Description
The dataset consists of ATP match records from the years 2020 to 2024.
*   **Source:** Consolidated CSV files (`atp_matches_2020.csv` to `atp_matches_2024.csv`).
*   **Volume:** Approximately 13,000 matches.
*   **Key Features:**
    *   **Tournament Info:** Surface (Hard, Clay, Grass), Tournament Level, Date.
    *   **Player Info:** Ranking, Age, Height, Handedness, Nationality.
    *   **Match Statistics:** Aces, Double Faults, Break Points Saved/Faced, Service Percentages.

## Project Structure
*   `data/`: Contains raw CSV data files.
*   `data_preprocessing.py`: Script for initial data cleaning and merging.
*   `eda_exploration.py`: Script for exploratory data analysis and statistical summaries.
*   `pyproject.toml`: Project configuration and dependencies.

## Requirements
*   Python >= 3.12
*   pandas, numpy, scikit-learn, matplotlib, seaborn
