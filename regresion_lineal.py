import numpy as np
from sklearn.linear_model import LinearRegression

# Datos de entrenamiento
tamaños = np.array([50, 100, 150, 200, 250, 300]).reshape(-1, 1)
precios = np.array([50000, 100000, 150000, 200000, 250000, 300000])

# Crear el modelo de regresión lineal
modelo = LinearRegression()

# Entrenar el modelo
modelo.fit(tamaños, precios)

# Datos de prueba
tamaño_prueba = np.array([120]).reshape(-1, 1)

# Realizar la predicción
precio_predicho = modelo.predict(tamaño_prueba)

# Imprimir el resultado
print("El precio predicho para una casa de 120 metros cuadrados es: ", precio_predicho)
