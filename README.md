# Mini Neural Network Framework

## Descripción

Este proyecto consiste en la implementación de un framework ligero de redes neuronales desarrollado desde cero en Python utilizando únicamente NumPy. No se emplean librerías de machine learning como TensorFlow o PyTorch, con el objetivo de comprender a detalle el funcionamiento interno de estos modelos.

El sistema incluye la implementación manual de propagación hacia adelante (forward pass), retropropagación (backward pass) y distintos métodos de optimización, incluyendo tanto enfoques de primer orden como de segundo orden.

---

## Objetivo

El propósito principal de este proyecto es profundizar en los fundamentos de aprendizaje automático mediante la construcción de un modelo completamente funcional desde cero. En particular:

* Comprender el flujo completo de entrenamiento de una red neuronal
* Implementar y analizar algoritmos de optimización
* Evaluar el comportamiento y eficiencia de distintos métodos de entrenamiento
* Desarrollar una arquitectura modular y reutilizable

---

## Características

* Arquitectura modular orientada a extensibilidad
* Implementación de capas densas (fully connected)
* Funciones de activación: ReLU y Sigmoid
* Funciones de pérdida:

  * Binary Cross Entropy
  * Mean Squared Error
* Optimizadores:

  * Gradient Descent
  * Newton-Raphson basado en segunda derivada (Hessiana)
* Interfaz de línea de comandos (CLI)
* API para integración en otros scripts
* Soporte para datasets en formato CSV y datos sintéticos
* Herramientas de visualización de convergencia

---

## Comparación de métodos de optimización

| Método           | Ventajas                        | Desventajas              |
| ---------------- | ------------------------------- | ------------------------ |
| Gradient Descent | Implementación simple y estable | Convergencia más lenta   |
| Newton-Raphson   | Convergencia rápida             | Alto costo computacional |

El método de Newton aprovecha información de segunda derivada para ajustar los parámetros de forma más eficiente, aunque su costo computacional lo hace menos viable en modelos de gran escala.


---

## Estructura del proyecto

```
mini_nn/
├── core/
├── optimizers/
├── utils/
├── interfaces/
├── experiments/
├── main.py
└── README.md
```


---

## Notas finales

Este proyecto fue desarrollado con fines educativos como parte del proceso de formación en machine learning. El enfoque principal fue entender los mecanismos internos de entrenamiento y optimización, priorizando la claridad de implementación sobre la eficiencia computacional.

La implementación busca servir como base para experimentación y extensión hacia modelos más complejos.
