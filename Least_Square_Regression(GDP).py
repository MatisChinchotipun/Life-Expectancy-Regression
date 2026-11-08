
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. READ DATA
# ==========================================

df = pd.read_csv("life_expectancy_2015_updated.csv")

# Remove missing and invalid values
data = df[["GDP_per_capita", "Life_expectancy"]].dropna()

x = data["GDP_per_capita"].to_numpy(dtype=float) / 1000
y = data["Life_expectancy"].to_numpy(dtype=float)

valid = np.isfinite(x) & np.isfinite(y) & (y > 0)
x = x[valid]
y = y[valid]

n = len(x)

# Total sum of squares
y_mean = np.mean(y)
SStot = np.sum((y - y_mean)**2)

print(f"Number of observations: {n}")


# ==========================================
# 2. LINEAR REGRESSION
# y = a0 + a1*x
# ==========================================

# Compute sums
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

# Predicted values
y_pred_lin = a0_lin + a1_lin*x

# Residuals and SSE
residual_lin = y - y_pred_lin
SSE_lin = np.sum(residual_lin**2)

# R-squared
R2_lin = 1 - SSE_lin/SStot

# Adjusted R-squared
p = 1
adj_R2_lin = 1 - (1-R2_lin)*(n-1)/(n-p-1)

# RMSE
RMSE_lin = np.sqrt(SSE_lin/n)

# Print results
print("\n===== LINEAR REGRESSION =====")
print(f"a0 = {a0_lin:.6f}")
print(f"a1 = {a1_lin:.6f}")
print(f"Model: y = {a0_lin:.6f} + ({a1_lin:.6f})x")

print("\n===== NUMERICAL RESULTS =====")
print(f"SSE = {SSE_lin:.4f}")
print(f"RMSE = {RMSE_lin:.4f}")
print(f"R^2 = {R2_lin:.4f}")
print(f"Adjusted R^2 = {adj_R2_lin:.4f}")


# ==========================================
# 3. QUADRATIC REGRESSION
# y = a0 + a1*x + a2*x^2
# ==========================================

# Compute sums
Sxxx = np.sum(x**3)
Sxxxx = np.sum(x**4)
Sxxy = np.sum((x**2)*y)

# Normal equation
A = np.array([
    [n,    Sx,   Sxx],
    [Sx,   Sxx,  Sxxx],
    [Sxx,  Sxxx, Sxxxx]
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

# Residuals and SSE
residual_quad = y - y_pred_quad
SSE_quad = np.sum(residual_quad**2)

# R-squared
R2_quad = 1 - SSE_quad/SStot

# Adjusted R-squared
p = 2
adj_R2_quad = 1 - (1-R2_quad)*(n-1)/(n-p-1)

# RMSE
RMSE_quad = np.sqrt(SSE_quad/n)

# Print results
print("\n===== QUADRATIC REGRESSION =====")
print(f"a0 = {a0_quad:.6f}")
print(f"a1 = {a1_quad:.6f}")
print(f"a2 = {a2_quad:.10f}")

print(
    f"Model: y = {a0_quad:.6f} "
    f"+ ({a1_quad:.6f})x "
    f"+ ({a2_quad:.10f})x^2"
)

print("\n===== NUMERICAL RESULTS =====")
print(f"SSE = {SSE_quad:.4f}")
print(f"RMSE = {RMSE_quad:.4f}")
print(f"R^2 = {R2_quad:.4f}")
print(f"Adjusted R^2 = {adj_R2_quad:.4f}")


# ==========================================
# 4. EXPONENTIAL REGRESSION
# y = b * e^(m*x)
# ==========================================

# 1. Transform data
#
# y = b*e^(m*x)
# ln(y) = ln(b) + m*x
#
# Let Y = ln(y)
# Then Y = a0 + a1*x

X = x
Y = np.log(y)

# 2. Compute sums
SX = np.sum(X)
SY = np.sum(Y)
SXX = np.sum(X**2)
SXY = np.sum(X*Y)

# 3. Normal equation
A = np.array([
    [n,  SX],
    [SX, SXX]
])

B = np.array([
    SY,
    SXY
])

# 4. Solve coefficients
a0_exp, a1_exp = np.linalg.solve(A, B)

# 5. Convert to exponential parameters
b_exp = np.exp(a0_exp)
m_exp = a1_exp

# 6. Predicted values on original scale
y_pred_exp = b_exp * np.exp(m_exp*x)

# 7. Residuals and SSE
residual_exp = y - y_pred_exp
SSE_exp = np.sum(residual_exp**2)

# 8. R-squared
R2_exp = 1 - SSE_exp/SStot

# 9. Adjusted R-squared
p = 1
adj_R2_exp = 1 - (1-R2_exp)*(n-1)/(n-p-1)

# 10. RMSE
RMSE_exp = np.sqrt(SSE_exp/n)

# Print results
print("\n===== EXPONENTIAL REGRESSION =====")
print(f"b = {b_exp:.6f}")
print(f"m = {m_exp:.10f}")

print(
    f"Model: y = {b_exp:.6f} "
    f"* e^({m_exp:.10f}x)"
)

print("\n===== NUMERICAL RESULTS =====")
print(f"SSE = {SSE_exp:.4f}")
print(f"RMSE = {RMSE_exp:.4f}")
print(f"R^2 = {R2_exp:.4f}")
print(f"Adjusted R^2 = {adj_R2_exp:.4f}")


# ==========================================
# 5. MODEL COMPARISON
# ==========================================

print("\n===== MODEL COMPARISON =====")

print(f"{'Model':<15} {'R^2':<12} {'Adjusted R^2':<16} {'RMSE':<12}")

print(
    f"{'Linear':<15} "
    f"{R2_lin:<12.4f} "
    f"{adj_R2_lin:<16.4f} "
    f"{RMSE_lin:<12.4f}"
)

print(
    f"{'Quadratic':<15} "
    f"{R2_quad:<12.4f} "
    f"{adj_R2_quad:<16.4f} "
    f"{RMSE_quad:<12.4f}"
)

print(
    f"{'Exponential':<15} "
    f"{R2_exp:<12.4f} "
    f"{adj_R2_exp:<16.4f} "
    f"{RMSE_exp:<12.4f}"
)

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

# Linear regression
y_line_lin = a0_lin + a1_lin*x_line

# Quadratic regression
y_line_quad = (
    a0_quad + a1_quad*x_line + a2_quad*x_line**2
)

# Exponential regression
y_line_exp = b_exp * np.exp(m_exp*x_line)

plt.figure(figsize=(10, 6))

# Actual data points
plt.scatter(
    x,
    y,
    color="gray",
    alpha=0.5,
    label="Actual data"
)

# Linear regression curve
plt.plot(
    x_line,
    y_line_lin,
    color="red",
    linewidth=2,
    label=f"Linear (R² = {R2_lin:.4f})"
)

# Quadratic regression curve
plt.plot(
    x_line,
    y_line_quad,
    color="blue",
    linewidth=2,
    label=f"Quadratic (R² = {R2_quad:.4f})"
)

# Exponential regression curve
plt.plot(
    x_line,
    y_line_exp,
    color="green",
    linewidth=2,
    label=f"Exponential (R² = {R2_exp:.4f})"
)

plt.xlabel("GDP per Capita (Thousand USD)")
plt.ylabel("Life Expectancy (Years)")

plt.title(
    "GDP per Capita vs Life Expectancy - Regression Comparison"
)

plt.legend()
plt.grid(True)

plt.tight_layout()

# Save graph for report
plt.savefig("GDP_regression_comparison.png", dpi=300)

plt.show()
