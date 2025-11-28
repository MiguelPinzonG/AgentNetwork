# Agente de Encaminamiento Racional con Utilidad Multiatributo

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**Proyecto académico basado en el Capítulo 15: "Making Simple Decisions" del libro "Artificial Intelligence: A Modern Approach" (4ª edición) de Russell & Norvig.**

Este proyecto implementa un agente racional que toma decisiones de encaminamiento en redes estocásticas, maximizando la utilidad esperada y calculando el Valor de la Información Perfecta (VPI) para determinar cuándo vale la pena sondear el estado de la red antes de tomar una decisión.

---

## 📋 Tabla de Contenidos

- [Características Principales](#-características-principales)
- [Conceptos Teóricos](#-conceptos-teóricos)
- [Instalación](#-instalación)
- [Uso Rápido](#-uso-rápido)
- [Arquitectura del Código](#-arquitectura-del-código)
- [Escenarios de Prueba](#-escenarios-de-prueba)
- [Guía de API](#-guía-de-api)
- [Personalización](#-personalización)
- [Resultados y Visualizaciones](#-resultados-y-visualizaciones)
- [Referencias](#-referencias)

---

## ✨ Características Principales

- **Modelo de Red Estocástica**: Simulación de redes con enlaces que pueden estar congestionados con probabilidades configurables
- **Utilidad Multiatributo**: Función de utilidad que considera latencia, pérdida de paquetes y costo
- **Cálculo de VPI**: Implementación del algoritmo de Valor de la Información Perfecta
- **Tres Políticas de Decisión**:
  - No sondear nunca (decisión basada en utilidad esperada)
  - Sondear siempre (información perfecta con costo)
  - Sondear solo si VPI > costo (decisión racional óptima)
- **Simulación Monte Carlo**: Evaluación empírica de políticas con miles de episodios
- **Visualizaciones Detalladas**: Gráficos comparativos de utilidad, latencia, pérdida y costos
- **Análisis Estadístico Completo**: Medias, desviaciones estándar, percentiles y distribuciones

---

## 📚 Conceptos Teóricos

### Utilidad Multiatributo

La función de utilidad evalúa la calidad de una ruta considerando múltiples atributos:

```
U(ruta) = -(α × latencia + β × pérdida + γ × costo)
```

Donde:
- **latencia**: Tiempo de transmisión en milisegundos
- **pérdida**: Probabilidad de pérdida de paquetes (0-1)
- **costo**: Costo monetario o de recursos
- **α, β, γ**: Pesos que reflejan las preferencias del usuario

Esta es una función de utilidad **aditiva** que asume independencia entre atributos. El signo negativo convierte la minimización de costos en maximización de utilidad.

### Valor de la Información Perfecta (VPI)

El VPI cuantifica cuánto vale conocer el estado real de la red antes de tomar una decisión:

```
VPI = E_s[max_a U(a,s)] - max_a E_s[U(a,s)]
```

Donde:
- **E_s[max_a U(a,s)]**: Utilidad esperada si conocemos el estado s y elegimos la mejor acción
- **max_a E_s[U(a,s)]**: Utilidad esperada de la mejor acción sin conocer el estado

**Interpretación**:
- Si **VPI > costo_sondeo**: Vale la pena sondear
- Si **VPI ≤ costo_sondeo**: Es mejor decidir sin sondear

El VPI es siempre **no negativo** (la información nunca puede empeorar las decisiones) y depende de:
1. **Incertidumbre**: Mayor incertidumbre → mayor VPI
2. **Diferencia entre alternativas**: Si todas las rutas son similares → VPI bajo
3. **Flexibilidad**: Más opciones disponibles → mayor potencial de VPI

### Teoría de Decisión bajo Incertidumbre

El agente opera bajo el principio de **Máxima Utilidad Esperada (MEU)**:

1. **Sin información**: Elige la acción que maximiza E[U(a)] sobre todos los estados posibles
2. **Con información perfecta**: Observa el estado s y elige la acción que maximiza U(a,s)
3. **Decisión de sondeo**: Sondea si y solo si VPI > costo_sondeo

Este enfoque es **racionalmente óptimo** bajo los axiomas de utilidad de von Neumann-Morgenstern.

---

## 🔧 Instalación

### Requisitos del Sistema

- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Instalación de Dependencias

1. **Clonar o descargar el proyecto**:
   ```bash
   cd proyecto
   ```

2. **Crear un entorno virtual (recomendado)**:
   ```bash
   python -m venv .venv
   
   # Windows
   .venv\Scripts\activate
   
   # Linux/Mac
   source .venv/bin/activate
   ```

3. **Instalar dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

### Verificación de Instalación

```bash
python -c "import numpy, matplotlib, seaborn; print('✓ Todas las dependencias instaladas correctamente')"
```

---

## 🚀 Uso Rápido

### Ejecutar el Análisis Completo

```bash
python cap15.py
```

Este comando ejecuta los tres escenarios de prueba, genera visualizaciones y muestra tablas comparativas.

### Salida Esperada

El programa generará:

1. **Análisis teórico** para cada escenario:
   - Utilidad esperada de cada ruta sin sondeo
   - Mejor ruta y su utilidad esperada
   - VPI (Valor de la Información Perfecta)
   - Decisión racional (sondear o no sondear)

2. **Simulación** (5000 episodios por política):
   - Estadísticas detalladas (media, desviación estándar, percentiles)
   - Comparación de las tres políticas
   - Frecuencia de uso de cada ruta

3. **Visualizaciones** (archivos PNG):
   - `resultados_escenario_1_probabilidades_moderadas.png`
   - `resultados_escenario_2_alta_incertidumbre.png`
   - `resultados_escenario_3_red_casi_determinista.png`
   - `comparacion_escenarios.png`

### Ejemplo de Salida

```
####################################################################################################
# ESCENARIO 1: PROBABILIDADES MODERADAS
####################################################################################################

┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                      ANÁLISIS TEÓRICO (Utilidad Esperada y VPI)                                  │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘

Rutas disponibles:
  Ruta 1: A -> B -> D         → EU(sin sondeo) =  -38.450
  Ruta 2: A -> C -> D         → EU(sin sondeo) =  -42.300

✓ Mejor ruta sin sondeo: A -> B -> D
  Utilidad esperada: -38.450

✓ EU(decisión con sondeo perfecto, incluyendo costo): -30.125

✓ VPI (Valor de la Información Perfecta): 8.325
✓ Costo de sondeo: 15.000

→ DECISIÓN RACIONAL: NO SONDEAR (VPI ≤ costo)
  Pérdida si se sondea: 6.675
```

---

## 🏗️ Arquitectura del Código

El proyecto está organizado en un único archivo `cap15.py` con las siguientes componentes:

```
cap15.py
├── 1. Modelo de Red
│   ├── Enlace (dataclass)
│   └── Red (class)
├── 2. Función de Utilidad
│   └── FuncionUtilidad (class)
├── 3. Agente Racional
│   └── Agente (class)
├── 4. Simulador
│   └── Simulador (class)
├── 5. Escenarios
│   ├── construir_escenario_base()
│   ├── escenario_1()
│   ├── escenario_2()
│   └── escenario_3()
├── 6. Visualización y Análisis
│   ├── crear_visualizaciones()
│   ├── imprimir_tabla_comparativa()
│   ├── analizar_escenario()
│   └── comparar_escenarios()
└── 7. Programa Principal
    └── main()
```

### Componentes Principales

#### 1. **Enlace** (dataclass)

Representa un enlace dirigido en la red con comportamiento estocástico.

**Atributos**:
- `origen`, `destino`: Nodos conectados
- `p_cong`: Probabilidad de congestión (0-1)
- `lat_no_cong`, `lat_cong`: Latencia sin/con congestión (ms)
- `loss_no_cong`, `loss_cong`: Pérdida sin/con congestión (0-1)
- `costo`: Costo fijo por usar el enlace

**Método clave**:
- `atributos_dado_estado(congestionado)`: Devuelve (latencia, pérdida, costo) según el estado

#### 2. **Red** (class)

Modelo de la topología de red y sus estados posibles.

**Métodos principales**:
- `agregar_enlace(enlace)`: Añade un enlace a la red
- `definir_rutas(origen, destino, rutas)`: Define rutas predefinidas
- `enumerar_estados()`: Genera todos los estados posibles (2^n combinaciones)
- `muestrear_estado(rng)`: Genera un estado aleatorio según probabilidades
- `atributos_ruta_dado_estado(ruta, estado)`: Calcula atributos agregados de una ruta

#### 3. **FuncionUtilidad** (class)

Función de utilidad multiatributo aditiva.

**Parámetros**:
- `alpha`: Peso de la latencia
- `beta`: Peso de la pérdida
- `gamma`: Peso del costo

**Método**:
- `evaluar(latencia, perdida, costo)`: Calcula U = -(α×lat + β×loss + γ×cost)

#### 4. **Agente** (class)

Agente racional que toma decisiones de encaminamiento.

**Métodos clave**:
- `utilidad_esperada_ruta_sin_sondeo(ruta)`: Calcula E[U(ruta)]
- `mejor_ruta_sin_sondeo()`: Devuelve la ruta con mayor utilidad esperada
- `utilidad_esperada_decision_con_sondeo_perfecto()`: Calcula E[max U(ruta|estado)] - costo
- `valor_esperado_informacion()`: Calcula VPI
- `mejor_ruta_dado_estado(estado)`: Decisión óptima conociendo el estado

#### 5. **Simulador** (class)

Simula episodios de envío de tráfico bajo diferentes políticas.

**Políticas**:
- `"no_probe"`: No sondear nunca
- `"probe_always"`: Sondear siempre
- `"vpi"`: Sondear solo si VPI > costo

**Método principal**:
- `ejecutar_politica(politica, n_episodios)`: Ejecuta n episodios y devuelve estadísticas completas

---

## 🧪 Escenarios de Prueba

El proyecto incluye tres escenarios que ilustran diferentes comportamientos del VPI:

### Escenario 1: Probabilidades Moderadas

**Configuración**:
- Probabilidades de congestión: 0.3, 0.2, 0.5, 0.4
- Incertidumbre moderada

**Resultado esperado**:
- VPI ≈ 7-10 (moderado)
- Decisión: **NO SONDEAR** (VPI < costo_sondeo = 15)

**Interpretación**: La incertidumbre no es suficiente para justificar el costo de sondeo.

### Escenario 2: Alta Incertidumbre

**Configuración**:
- Probabilidades de congestión: 0.5, 0.5, 0.5, 0.5
- Máxima incertidumbre (entropía máxima)

**Resultado esperado**:
- VPI ≈ 15-20 (alto)
- Decisión: **SONDEAR** (VPI > costo_sondeo)

**Interpretación**: Con alta incertidumbre, la información es valiosa y justifica el costo.

### Escenario 3: Red Casi Determinista

**Configuración**:
- Probabilidades de congestión: 0.05, 0.05, 0.95, 0.95
- Incertidumbre muy baja

**Resultado esperado**:
- VPI ≈ 0-2 (muy bajo)
- Decisión: **NO SONDEAR** (VPI << costo_sondeo)

**Interpretación**: Cuando el estado es casi predecible, la información adicional aporta poco valor.

---

## 📖 Guía de API

### Crear una Red Personalizada

```python
from cap15 import Red, Enlace, FuncionUtilidad, Agente

# Crear red
red = Red()

# Añadir enlaces
red.agregar_enlace(Enlace(
    origen="A",
    destino="B",
    p_cong=0.3,
    lat_no_cong=10.0,
    lat_cong=40.0,
    loss_no_cong=0.01,
    loss_cong=0.08,
    costo=1.0
))

# Definir rutas
ruta1 = [("A", "B"), ("B", "C")]
ruta2 = [("A", "D"), ("D", "C")]
red.definir_rutas("A", "C", [ruta1, ruta2])
```

### Crear un Agente y Calcular VPI

```python
# Definir función de utilidad
funcion_utilidad = FuncionUtilidad(
    alpha=1.0,    # Penalización por ms de latencia
    beta=200.0,   # Penalización por unidad de pérdida
    gamma=5.0     # Penalización por unidad de costo
)

# Crear agente
agente = Agente(
    red=red,
    funcion_utilidad=funcion_utilidad,
    rutas=[ruta1, ruta2],
    costo_sondeo=15.0
)

# Calcular VPI
vpi = agente.valor_esperado_informacion()
print(f"VPI = {vpi:.3f}")

# Decisión racional
if vpi > agente.costo_sondeo:
    print("Decisión: SONDEAR")
else:
    print("Decisión: NO SONDEAR")
```

### Ejecutar Simulación

```python
from cap15 import Simulador

# Crear simulador
simulador = Simulador(agente, origen="A", destino="C", rng_seed=42)

# Ejecutar política
resultados = simulador.ejecutar_politica("vpi", n_episodios=1000)

# Acceder a estadísticas
print(f"Utilidad media: {resultados['utilidad_media']:.3f}")
print(f"Latencia media: {resultados['latencia_media']:.2f} ms")
print(f"Pérdida media: {resultados['perdida_media']*100:.2f}%")
```

---

## 🎨 Personalización

### Modificar Parámetros de Utilidad

Para cambiar las preferencias del agente, ajusta los pesos α, β, γ:

```python
# Priorizar latencia baja
funcion_utilidad = FuncionUtilidad(alpha=5.0, beta=100.0, gamma=1.0)

# Priorizar baja pérdida
funcion_utilidad = FuncionUtilidad(alpha=1.0, beta=500.0, gamma=1.0)

# Priorizar bajo costo
funcion_utilidad = FuncionUtilidad(alpha=1.0, beta=100.0, gamma=20.0)
```

### Crear un Escenario Personalizado

```python
def mi_escenario():
    red, origen, destino, rutas = construir_escenario_base(
        p_cong_A_B=0.4,
        p_cong_B_D=0.3,
        p_cong_A_C=0.6,
        p_cong_C_D=0.5
    )
    
    funcion_utilidad = FuncionUtilidad(alpha=2.0, beta=150.0, gamma=3.0)
    costo_sondeo = 10.0
    
    return red, origen, destino, rutas, funcion_utilidad, costo_sondeo

# Analizar
from cap15 import analizar_escenario
resultados = analizar_escenario(
    "Mi Escenario Personalizado",
    mi_escenario,
    n_episodios=5000
)
```

### Ajustar Visualizaciones

Modifica las constantes al inicio de las funciones de visualización:

```python
# En crear_visualizaciones()
colores = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # Paleta personalizada
plt.rcParams['figure.figsize'] = (20, 12)   # Tamaño de figura
```

---

## 📊 Resultados y Visualizaciones

### Gráficos Generados

Cada escenario genera 6 gráficos comparativos:

1. **Distribución de Utilidad**: Histogramas superpuestos de las tres políticas
2. **Boxplot de Utilidad**: Comparación de medianas, cuartiles y outliers
3. **Latencia Media**: Barras con desviación estándar
4. **Pérdida Media**: Comparación de pérdida de paquetes
5. **Costo Medio**: Incluye el costo de sondeo cuando aplica
6. **Frecuencia de Uso de Rutas**: Qué rutas elige cada política

### Interpretación de Resultados

**Escenario 1 (Moderada Incertidumbre)**:
- "No sondeo" y "Sondeo según VPI" tienen rendimiento similar (ambas no sondean)
- "Sondeo siempre" tiene menor utilidad debido al costo innecesario

**Escenario 2 (Alta Incertidumbre)**:
- "Sondeo siempre" y "Sondeo según VPI" tienen mejor rendimiento (ambas sondean)
- "No sondeo" sufre por tomar decisiones subóptimas frecuentemente

**Escenario 3 (Baja Incertidumbre)**:
- Todas las políticas tienen rendimiento similar
- La red es predecible, por lo que sondear no aporta valor

### Métricas Clave

- **Utilidad Media**: Métrica principal de rendimiento
- **Desviación Estándar**: Indica variabilidad/riesgo
- **Percentiles**: P25, P50 (mediana), P75 para análisis de distribución
- **Frecuencia de Rutas**: Revela qué rutas son preferidas en la práctica

---

## 📚 Referencias

### Bibliografía Principal

- **Russell, S., & Norvig, P. (2020)**. *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson.
  - **Capítulo 15**: Making Simple Decisions
  - **Sección 15.3**: The Value of Information

### Conceptos Relacionados

- **Teoría de Utilidad**: von Neumann-Morgenstern utility theory
- **Teoría de Decisión**: Decision theory under uncertainty
- **Valor de la Información**: Value of Perfect Information (VPI)
- **Utilidad Multiatributo**: Multi-attribute utility theory (MAUT)

### Recursos Adicionales

- [AIMA Code Repository](https://github.com/aimacode/aima-python) - Implementaciones de referencia
- [Decision Theory (Stanford Encyclopedia)](https://plato.stanford.edu/entries/decision-theory/)

---

## 📄 Licencia

Este proyecto es material académico desarrollado con fines educativos.

---

## 👥 Créditos

Proyecto desarrollado como parte del curso de **Modelos Estocásticos**, basado en los conceptos del libro "Artificial Intelligence: A Modern Approach" de Stuart Russell y Peter Norvig.

---

**¿Preguntas o sugerencias?** Este es un proyecto académico abierto a mejoras y extensiones.
