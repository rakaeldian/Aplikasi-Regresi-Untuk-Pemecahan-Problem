import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error

#Raka Eldiansyah Putra
#21120122140150

#Mengambil data
data_url = r"C:\Users\Public\donwload\student_performance.csv"
data = pd.read_csv(data_url)

# Memilih kolom
data = data[['Hours Studied', 'Sample Question Papers Practiced', 'Performance Index']]

# Mengambil data yang dibutuhkan untuk analisis
X = data['Hours Studied'].values
y = data['Performance Index'].values

# Model Eksponensial
def exponential_model(x, a, b):
    return a * np.exp(b * x)

# Memasang model eksponensial ke data
params, covariance = curve_fit(exponential_model, X, y)
a, b = params
y_pred_exponential = exponential_model(X, a, b)

# Plot data dan hasil regresi eksponensial
plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(X, y_pred_exponential, color='green', label='Exponential Regression')
plt.xlabel('Hours Studied')
plt.ylabel('Performance Index')
plt.title('Exponential Regression')
plt.legend()
plt.show()

# Menghitung RMS error untuk model eksponensial
rms_exponential = np.sqrt(mean_squared_error(y, y_pred_exponential))
print(f'RMS Error for Exponential Model: {rms_exponential}')
