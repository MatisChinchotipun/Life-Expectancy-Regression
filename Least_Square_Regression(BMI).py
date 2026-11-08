import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
# ==========================================
# 1. READ DATA
# ==========================================
df = pd.read_csv("life_expectancy_2015_updated.csv")

data = df[["BMI", "Life_expectancy"]].dropna()

x = data["BMI"].to_numpy()
y = data["Life_expectancy"].to_numpy()

n = len(x)

# Total sum of squares
y_mean = np.mean(y)
SStot = np.sum((y - y_mean)**2)


# ==========================================
# 2. LINEAR REGRESSION
# y = a0 + a1*x
# ==========================================

Sx = np.sum(x)
Sy = np.sum(y)
Sxx = np.sum(x**2)
Sxy = np.sum(x*y)

# Normal equation
A = np.array([
    [n,  Sx],
    [Sx, Sxx]
])

B = np.array([
    Sy,
    Sxy
])

# Solve for coefficients
a0_lin, a1_lin = np.linalg.solve(A, B)

# Predicted values using BMI x
y_pred_lin = a0_lin + a1_lin*x

# SSE - Sum of Squared Error
SSE_lin = np.sum((y - y_pred_lin)**2)

# R-squared
R2_lin = 1 - SSE_lin/SStot

# Adjusted R-squared
p = 1
adj_R2_lin = 1 - (1-R2_lin)*(n-1)/(n-p-1)

# RMSE
RMSE_lin = np.sqrt(SSE_lin/n)

print("\n===== LINEAR REGRESSION =====")
print(f"a0 = {a0_lin:.6f}")
print(f"a1 = {a1_lin:.6f}")
print(f"Model: y = {a0_lin:.6f} + ({a1_lin:.6f})x")
print(f"SSE = {SSE_lin:.4f}")
print(f"RMSE = {RMSE_lin:.4f}")
print(f"R^2 = {R2_lin:.4f}")
print(f"Adjusted R^2 = {adj_R2_lin:.4f}")


# ==========================================
# 3. QUADRATIC REGRESSION
# y = a0 + a1*x + a2*x^2
# ==========================================

Sxxx = np.sum(x**3)
Sxxxx = np.sum(x**4)
Sxxy = np.sum((x**2)*y)

# Normal equation
A = np.array([
    [n,   Sx,   Sxx],
    [Sx,  Sxx,  Sxxx],
    [Sxx, Sxxx, Sxxxx]
])

B = np.array([
    Sy,
    Sxy,
    Sxxy
])

# Solve for coefficients
a0_quad, a1_quad, a2_quad = np.linalg.solve(A, B)

# Predicted values
y_pred_quad = a0_quad + a1_quad*x + a2_quad*x**2

# SSE
SSE_quad = np.sum((y - y_pred_quad)**2)

# R-squared
R2_quad = 1 - SSE_quad/SStot

# Adjusted R-squared
p = 2
adj_R2_quad = 1 - (1-R2_quad)*(n-1)/(n-p-1)

# RMSE
RMSE_quad = np.sqrt(SSE_quad/n)

print("\n===== QUADRATIC REGRESSION =====")
print(f"a0 = {a0_quad:.6f}")
print(f"a1 = {a1_quad:.6f}")
print(f"a2 = {a2_quad:.6f}")
print(f"Model: y = {a0_quad:.6f} + ({a1_quad:.6f})x + ({a2_quad:.6f})x^2")
print(f"SSE = {SSE_quad:.4f}")
print(f"RMSE = {RMSE_quad:.4f}")
print(f"R^2 = {R2_quad:.4f}")
print(f"Adjusted R^2 = {adj_R2_quad:.4f}")


# ==========================================
# 4. EXPONENTIAL REGRESSION
# y = a*e^(b*x)
# ==========================================

# Define exponential function
def exponential(x, a, b):
    return a * np.exp(b*x)

# Fit using nonlinear least squares
params, covariance = curve_fit(
    exponential,
    x,
    y,
    p0=[np.mean(y), 0.0],
    maxfev=10000
)

a_exp, b_exp = params

# Predicted values
y_pred_exp = exponential(x, a_exp, b_exp)

# SSE
SSE_exp = np.sum((y - y_pred_exp)**2)

# R-squared
R2_exp = 1 - SSE_exp/SStot

# Adjusted R-squared
p = 2
adj_R2_exp = 1 - (1-R2_exp)*(n-1)/(n-p-1)

# RMSE
RMSE_exp = np.sqrt(SSE_exp/n)

print("\n===== EXPONENTIAL REGRESSION =====")
print(f"a = {a_exp:.6f}")
print(f"b = {b_exp:.6f}")
print(f"Model: y = {a_exp:.6f} * e^({b_exp:.6f}x)")
print(f"SSE = {SSE_exp:.4f}")
print(f"RMSE = {RMSE_exp:.4f}")
print(f"R^2 = {R2_exp:.4f}")
print(f"Adjusted R^2 = {adj_R2_exp:.4f}")


# ==========================================
# 5. COMPARISON OF ALL THREE MODELS
# ==========================================

print("\n===== MODEL COMPARISON =====")
print(f"Linear R^2:      {R2_lin:.4f}")
print(f"Quadratic R^2:   {R2_quad:.4f}")
print(f"Exponential R^2: {R2_exp:.4f}")

results = {
    "Linear": R2_lin,
    "Quadratic": R2_quad,
    "Exponential": R2_exp
}

best_model = max(results, key=results.get)
print(f"\nBest fitting model: {best_model}")


# ==========================================
# 6. COMBINED GRAPH
# ==========================================

x_line = np.linspace(x.min(), x.max(), 500)

# Regression curves
y_line_lin = a0_lin + a1_lin*x_line

y_line_quad = (
    a0_quad + a1_quad*x_line + a2_quad*x_line**2
)

y_line_exp = exponential(x_line, a_exp, b_exp)

plt.figure(figsize=(10, 6))

# Original data
plt.scatter(
    x, y,
    color="gray",
    alpha=0.5,
    label="Actual Data"
)

# Linear
plt.plot(
    x_line, y_line_lin,
    color="red",
    label=f"Linear (R² = {R2_lin:.4f})"
)

# Quadratic
plt.plot(
    x_line, y_line_quad,
    color="blue",
    label=f"Quadratic (R² = {R2_quad:.4f})"
)

# Exponential
plt.plot(
    x_line, y_line_exp,
    color="green",
    label=f"Exponential (R² = {R2_exp:.4f})"
)

plt.xlabel("BMI")
plt.ylabel("Life Expectancy (Years)")
plt.title("BMI vs Life Expectancy - Regression Comparison")

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("BMI_regression_comparison.png", dpi=300)
plt.show()
