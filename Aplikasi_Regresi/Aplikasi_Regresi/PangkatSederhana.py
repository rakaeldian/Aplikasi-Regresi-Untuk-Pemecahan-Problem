import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

data_url = r"C:\Users\Public\Download\student_performance.csv"
data = pd.read_csv(data_url)

data = data[['Hours Studied', 'Sample Question Papers Practiced', 'Performance Index']]
X = data['Hours Studied'].values
y = data['Performance Index'].values

def simple_power_model(x, a, b):
    return a * (x ** b)

# Fitting model to data
params, _ = curve_fit(simple_power_model, X, y)
a, b = params
y_pred_power = simple_power_model(X, a, b)

# Plotting the exponential regression
plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(X, y_pred_power, color='green', label='Power Regression')
plt.xlabel('Hours Studied')
plt.ylabel('Performance Index')
plt.title('Power Regression')
plt.legend()
plt.show()

# Calculating RMS Error for the power model
rms_power = np.sqrt(mean_squared_error(y, y_pred_power))
print(f'RMS Error for Power Model: {rms_power}')
