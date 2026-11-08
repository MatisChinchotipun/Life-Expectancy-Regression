# Life-Expectancy-Regression
Least-squares regression analysis using GDP per capita and BMI to predict life expectancy.

# Life Expectancy Prediction Using Least-Squares Regression

This project was developed for a Scientific Computing course.

## Objective

To investigate the relationships between life expectancy, GDP per capita, and BMI using country-level data from 2015.

## Regression Models

Three regression models are implemented for each predictor:

1. Linear regression
2. Quadratic regression
3. Exponential regression

## Implementation

The regression calculations are performed in Python using NumPy, Pandas, and Matplotlib. SciPy is also used if nonlinear least-squares fitting is selected for the exponential model.

## Model Evaluation

The models are evaluated using:

- Sum of Squared Errors (SSE)
- Coefficient of Determination (R²)
- Adjusted R²
- Root Mean Square Error (RMSE)

## Source Code

- `BMI.py` — Regression analysis using BMI.
- `GDP.py` — Regression analysis using GDP per capita.

## Dataset

Country-level life expectancy data from 2015, selected from a larger multi-year dataset.

Original dataset source: [Insert source link]
