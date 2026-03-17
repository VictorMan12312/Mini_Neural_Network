`minineural` es un framework ligero de redes neuronales implementado desde cero en Python utilizando NumPy.
Incluye soporte para múltiples capas, funciones de pérdida y optimizadores, incluyendo un método de segundo orden basado en Newton-Raphson.

---

## Requisitos

* Python 3.8+
* numpy
* matplotlib (opcional, para visualización)
* pandas (para carga de datasets)

Instalación de dependencias:

```bash
pip install numpy matplotlib pandas
```

---

## Estructura del proyecto

```
mini_nn/
├── core/           # Definición del modelo y capas
├── optimizers/     # Algoritmos de optimización
├── utils/          # Utilidades (datos, métricas, gráficas)
├── interfaces/     # CLI y API
├── experiments/    # Scripts de prueba
├── main.py         # Punto de entrada
```

---

## Componentes principales

### core/

* `layers.py`
  Contiene la implementación de capas densas y funciones de activación.
  Cada capa implementa:

  * `forward()`
  * `backward()`

* `model.py`
  Maneja:

  * propagación hacia adelante
  * retropropagación
  * ciclo de entrenamiento

* `loss.py`
  Funciones de pérdida:

  * Binary Cross Entropy
  * Mean Squared Error

---

### optimizers/

* `gradient_descent.py`
  Implementación estándar de descenso por gradiente.

* `newton.py`
  Implementación del método de Newton-Raphson usando gradiente y Hessiana.
  Utiliza pseudo-inversa para mayor estabilidad.

* `base_optimizer.py`
  Clase base para todos los optimizadores.

---

### utils/

* `data_loader.py`
  Carga datasets desde CSV o genera datos sintéticos.

* `metrics.py`
  Cálculo de métricas como accuracy.

* `visualization.py`
  Gráficas de entrenamiento (loss vs iteraciones).

---

### interfaces/

* `cli.py`
  Permite ejecutar el entrenamiento desde terminal.

* `api.py`
  Permite usar el modelo como librería dentro de otros scripts.

---

### experiments/

Scripts para pruebas:

* `test_synthetic.py`
  Dataset generado artificialmente.

* `test_titanic.py`
  Dataset real para clasificación.

---

## Uso

### Ejecutar desde CLI

```bash
python main.py --dataset synthetic --optimizer gd --epochs 100
```

```bash
python main.py --dataset synthetic --optimizer newton --epochs 10
```

---

## Flujo de entrenamiento

1. Se cargan los datos
2. Se construye el modelo (capas)
3. Forward pass
4. Cálculo de loss
5. Backward pass (gradientes)
6. Actualización de parámetros (optimizer)

---

## Notas importantes

* La implementación de Newton-Raphson puede ser costosa computacionalmente
* Puede haber inestabilidad numérica si la Hessiana no es invertible
* Se usa pseudo-inversa (`np.linalg.pinv`) para mitigar este problema

---

## Propósito

Este proyecto fue desarrollado con fines educativos para comprender en detalle:

* Backpropagation
* Optimización de primer y segundo orden
* Estructura interna de redes neuronales

---
