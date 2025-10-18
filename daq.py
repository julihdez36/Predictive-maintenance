
'''
Naturaleza de lo datos:
    1. Cross-sectional [corte transversal]:
        Registros muestrales sobre la cual se 
        estudian diferentes características en
        un corte específico del tiempo.
        IID: Independientes e identicamente distribuidos 
    2. Time-series [series temporales]:
        El criterio de indexación es temporal y 
        por tanto, los registros exhiben una
        correlación serial (no hay independencia)
        
    3. Longitudinales

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

df1 = sns.load_dataset('penguins')
df1.columns

df = sns.load_dataset('dowjones')

plt.plot(df.Date, df.Price)
plt.grid(linestyle = '--')
plt.title('Dow-Jones Industrial Stock Price Index for US')
plt.ylabel('Prices (USD)')
plt.xlabel('Date')

'''
Las series temporales requieren un análisis de
componentes:
    1. Tendencia (-ciclo)
    2. El ciclo
    3. La estacionalidad
    4. Aleatoriedad
'''

air = sm.datasets.get_rdataset("AirPassengers").data
air.columns
plt.plot(air.time, air.value) 
plt.grid(linestyle = '--')
plt.title('Vuelos aerolinea')
plt.xlabel('Fecha')
plt.ylabel('Pasajeros')


co2 = sm.datasets.co2.load_pandas().data
co2.columns

co2['date']= pd.to_datetime(co2['Index'])

plt.plot(co2)


df

# Introducción a las series temporales

# https://filippomb.github.io/python-time-series-handbook/notebooks/01/introduction_to_time_series.html


'''

Componentes de series de tiempo 

A menudo se supone que una serie temporal se compone de tres componentes:

    - Tendencia : la dirección a largo plazo.

    - Estacionalidad : el comportamiento periódico.

    - Residuos : las fluctuaciones irregulares.
'''

# Tendencia

time = np.arange(144) #Secuencia del 0 al 143
trend = time * 2.65 +100 # combinación lineal

fig, ax = plt.subplots(1, 1, figsize=(8, 3))
ax.plot(time, trend, color='tab:red')
ax.set_xlabel("Months")
ax.set_ylabel("Passengers")
plt.grid()
plt.title("Trend vs Time")

# Estacionalidad 

seasonal = 20 + np.sin( time * 0.5) * 20

fig, ax = plt.subplots(1, 1, figsize=(10, 3))
ax.plot(time, seasonal, color='tab:orange')
ax.set_xlabel("Months")
ax.set_ylabel("Passengers")
plt.grid()
plt.title("Seasonality vs Time");

# Residuos

residuals = np.random.normal(loc=0.0, scale=3, size=len(time))

fig, ax = plt.subplots(1, 1, figsize=(8, 3))
ax.plot(time, residuals, color='tab:green')
ax.set_xlabel("Months")
ax.set_ylabel("Passengers")
plt.grid()
plt.title("Residuals vs Time");


# Modelos de descomposición 

'''
Los componentes de series temporales se pueden descomponer con los siguientes modelos:

    1. Descomposición aditiva

    2. Descomposición multiplicativa

    3. Descomposición pseudoaditiva
    
'''


fig, ax = plt.subplots(nrows = 3, ncols = 1)
ax[0].plot(time, trend)
ax[0].set_title('Componente tendencial')
ax[1].plot(time, seasonal)
ax[1].set_title('Componente estacional')
ax[2].plot(time, residuals)
ax[2].set_title('Componente residual')
plt.tight_layout()


# Modelo aditivo

trend_residuals = trend + residuals
plt.plot(trend_residuals)

trend_seasonal = trend + seasonal
plt.plot(trend_seasonal)


additive = trend + seasonal + residuals

fig, ax = plt.subplots(1, 1, figsize=(8, 3))
ax.plot(time, additive, 'tab:blue')
ax.set_xlabel("Months")
ax.set_ylabel("Passengers")
plt.title("Additive Time Series")
plt.grid();


# Modelo multiplicativo

multiplicative = trend * seasonal # we do not include residuals to make the pattern more clear

fig, ax = plt.subplots(1, 1, figsize=(8, 3))
ax.plot(time, multiplicative, 'tab:blue')
ax.set_xlabel("Months")
ax.set_ylabel("Passengers")
plt.title("Multiplicative Time Series")
plt.grid();

# Descomposición de series de tiempo 

slope, intercept = np.polyfit(np.arange(len(additive)), additive, 1) # estimate line coefficient
trend = np.arange(len(additive)) * slope + intercept # linear trend
detrended = additive - trend # remove the trend

plt.figure(figsize=(8, 3))
plt.plot(additive, label='Original')
plt.plot(trend, label='Trend')
plt.plot(detrended, label='Detrended')
plt.grid()
plt.legend();

# Descomposición aditiva 

from statsmodels.tsa.seasonal import seasonal_decompose

additive_decomposition = seasonal_decompose(x=additive, model='additive', period=12)

# Utility function to make the plots
def seas_decomp_plots(original, decomposition):
    _, axes = plt.subplots(4, 1, sharex=True, sharey=False, figsize=(7, 5))
    axes[0].plot(original, label='Original')
    axes[0].legend(loc='upper left')
    axes[1].plot(decomposition.trend, label='Trend')
    axes[1].legend(loc='upper left')
    axes[2].plot(decomposition.seasonal, label='Seasonality')
    axes[2].legend(loc='upper left')
    axes[3].plot(decomposition.resid, label='Residuals')
    axes[3].legend(loc='upper left')
    plt.show()    
    
seas_decomp_plots(additive, additive_decomposition)


## Descomposición multiplicativa

multiplicative_decomposition = seasonal_decompose(x=multiplicative, model='multiplicative', 
                                                  period=12)
seas_decomp_plots(multiplicative, multiplicative_decomposition)


# Locally estimated scatterplot smoothing (LOESS)

from statsmodels.tsa.seasonal import STL

stl_decomposition = STL(endog=additive, period=12, robust=True).fit()
seas_decomp_plots(additive, stl_decomposition)



seasonal = 20 + np.sin( time * 0.5) * 20

from scipy.fft import fft

def fft_analysis(signal):
    
    # Linear detrending
    slope, intercept = np.polyfit(np.arange(len(signal)), signal, 1)
    trend = np.arange(len(signal)) * slope + intercept 
    detrended = signal - trend 
    
    fft_values = fft(detrended)
    frequencies = np.fft.fftfreq(len(fft_values))

    # Remove negative frequencies and sort
    positive_frequencies = frequencies[frequencies > 0]
    magnitudes = np.abs(fft_values)[frequencies > 0]

    # Identify dominant frequency
    dominant_frequency = positive_frequencies[np.argmax(magnitudes)]
    print(f"Dominant Frequency: {dominant_frequency:.3f}")

    # Convert frequency to period (e.g., days, weeks, months, etc.)
    dominant_period = 1 / dominant_frequency
    print(f"Dominant Period: {dominant_period:.2f} time units")
    
    return dominant_period, positive_frequencies, magnitudes

period, freqs, magnitudes = fft_analysis(seasonal)

# Plotting the spectrum
plt.figure(figsize=(10, 3))
plt.stem(freqs, magnitudes)
plt.title('Frequency Spectrum')
plt.xlabel('Frequency')
plt.ylabel('Magnitude')
plt.show()