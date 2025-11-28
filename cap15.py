"""
Proyecto: Agente de encaminamiento racional con utilidad multiatributo y valor de la información
Basado en Capítulo 15: "Making Simple Decisions" (Russell & Norvig, AIMA 4e)

Este archivo contiene una implementación completa y autocontenida con:
- Modelo de red estocástica (clases Red y Enlace).
- Función de utilidad multiatributo (FuncionUtilidad).
- Agente racional que maximiza utilidad esperada y calcula VPI (Agente).
- Simulador de políticas de sondeo (Simulador).
- Tres escenarios de prueba (escenario_1, escenario_2, escenario_3).
- Un main simple por consola para ejecutar y ver resultados numéricos.

Autor: Proyecto académico - Modelos Estocásticos
Referencia: Russell, S., & Norvig, P. (2020). Artificial Intelligence: A Modern Approach (4th ed.)
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional
import itertools
import random
import statistics
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns


# ==============================================================================
# CONFIGURACIÓN Y CONSTANTES
# ==============================================================================

# Configuración de simulación
DEFAULT_EPISODES = 5000
DEFAULT_RANDOM_SEED = 42

# Configuración de visualización
PLOT_DPI = 300
PLOT_FIGURE_SIZE = (18, 10)
PLOT_COLORS = ['#3498db', '#e74c3c', '#2ecc71']  # Azul, Rojo, Verde
PLOT_STYLE = 'whitegrid'

# Nombres de políticas
POLICY_NO_PROBE = "no_probe"
POLICY_PROBE_ALWAYS = "probe_always"
POLICY_VPI = "vpi"

# Type aliases para mejor legibilidad
NetworkState = Dict[Tuple[str, str], bool]  # Estado de red: {(origen, destino): congestionado}
Route = List[Tuple[str, str]]  # Ruta: lista de enlaces (origen, destino)
RouteAttributes = Tuple[float, float, float]  # (latencia, pérdida, costo)


# ----------------------------------------------------------------------
# 1. Modelo de red: Enlace y Red
# ----------------------------------------------------------------------

@dataclass
class Enlace:
    """
    Representa un enlace dirigido en la red con comportamiento estocástico de congestión.
    
    Un enlace puede estar congestionado o no, afectando su latencia y pérdida de paquetes.
    La probabilidad de congestión determina la distribución de estados del enlace.
    
    Attributes:
        origen (str): Nodo de origen del enlace
        destino (str): Nodo de destino del enlace
        p_cong (float): Probabilidad de congestión (rango: 0.0-1.0)
        lat_no_cong (float): Latencia en ms cuando NO está congestionado
        lat_cong (float): Latencia en ms cuando SÍ está congestionado
        loss_no_cong (float): Probabilidad de pérdida sin congestión (rango: 0.0-1.0)
        loss_cong (float): Probabilidad de pérdida con congestión (rango: 0.0-1.0)
        costo (float): Costo fijo por usar este enlace
    
    Example:
        >>> enlace = Enlace(
        ...     origen="A", destino="B",
        ...     p_cong=0.3,
        ...     lat_no_cong=10.0, lat_cong=40.0,
        ...     loss_no_cong=0.01, loss_cong=0.08,
        ...     costo=1.0
        ... )
        >>> lat, loss, cost = enlace.atributos_dado_estado(congestionado=True)
        >>> print(f"Latencia: {lat} ms, Pérdida: {loss*100}%, Costo: {cost}")
    """
    origen: str
    destino: str
    p_cong: float              # Probabilidad de estar congestionado
    lat_no_cong: float         # Latencia cuando NO está congestionado
    lat_cong: float            # Latencia cuando SÍ está congestionado
    loss_no_cong: float        # Pérdida cuando NO está congestionado (0–1)
    loss_cong: float           # Pérdida cuando SÍ está congestionado (0–1)
    costo: float               # Costo fijo por usar este enlace

    def __post_init__(self) -> None:
        """Valida los parámetros del enlace."""
        if not 0.0 <= self.p_cong <= 1.0:
            raise ValueError(f"p_cong debe estar en [0, 1], recibido: {self.p_cong}")
        if not 0.0 <= self.loss_no_cong <= 1.0:
            raise ValueError(f"loss_no_cong debe estar en [0, 1], recibido: {self.loss_no_cong}")
        if not 0.0 <= self.loss_cong <= 1.0:
            raise ValueError(f"loss_cong debe estar en [0, 1], recibido: {self.loss_cong}")
        if self.lat_no_cong < 0 or self.lat_cong < 0:
            raise ValueError("Las latencias no pueden ser negativas")
        if self.costo < 0:
            raise ValueError("El costo no puede ser negativo")

    def atributos_dado_estado(self, congestionado: bool) -> RouteAttributes:
        """
        Devuelve (latencia, pérdida, costo) para este enlace dado si está
        congestionado o no.
        """
        if congestionado:
            return self.lat_cong, self.loss_cong, self.costo
        else:
            return self.lat_no_cong, self.loss_no_cong, self.costo


class Red:
    """
    Modelo de la red como un conjunto de nodos y enlaces dirigidos.
    
    La red mantiene la topología (nodos y enlaces) y permite:
    - Enumerar todos los estados posibles de congestión
    - Muestrear estados aleatorios según probabilidades
    - Calcular atributos de rutas dado un estado
    
    Los estados de la red se representan como:
        estado[(u, v)] = True/False
    donde True = congestionado, False = no congestionado.
    
    Attributes:
        nodos (List[str]): Lista de nodos en la red
        enlaces (Dict[Tuple[str, str], Enlace]): Enlaces indexados por (origen, destino)
        rutas_predefinidas (Dict): Rutas predefinidas entre pares de nodos
    
    Example:
        >>> red = Red()
        >>> red.agregar_enlace(Enlace("A", "B", 0.3, 10, 40, 0.01, 0.08, 1.0))
        >>> red.definir_rutas("A", "D", [[('A', 'B'), ('B', 'D')]])
        >>> for estado, prob in red.enumerar_estados():
        ...     print(f"Estado: {estado}, Probabilidad: {prob}")
    """

    def __init__(self) -> None:
        self.nodos: List[str] = []
        self.enlaces: Dict[Tuple[str, str], Enlace] = {}
        # Opcionalmente, se pueden predefinir rutas:
        self.rutas_predefinidas: Dict[Tuple[str, str], List[List[Tuple[str, str]]]] = {}

    # -------------------------------
    # Gestión de topología
    # -------------------------------

    def agregar_enlace(self, enlace: Enlace) -> None:
        """
        Agrega un enlace a la red y registra sus nodos.
        """
        key = (enlace.origen, enlace.destino)
        self.enlaces[key] = enlace
        if enlace.origen not in self.nodos:
            self.nodos.append(enlace.origen)
        if enlace.destino not in self.nodos:
            self.nodos.append(enlace.destino)

    # -------------------------------
    # Rutas
    # -------------------------------

    def definir_rutas(self, origen: str, destino: str, rutas: List[List[Tuple[str, str]]]) -> None:
        """
        Permite guardar explícitamente un conjunto de rutas entre origen y destino.

        Cada ruta es una lista de tuplas (origen, destino) que corresponden a enlaces.
        """
        self.rutas_predefinidas[(origen, destino)] = rutas

    def obtener_rutas(self, origen: str, destino: str) -> List[List[Tuple[str, str]]]:
        """
        Devuelve las rutas predefinidas entre origen y destino.

        En este proyecto trabajamos con pocas rutas y las fijamos manualmente.
        Si no existen en rutas_predefinidas, se podría implementar un DFS/BFS,
        pero aquí no es necesario.
        """
        key = (origen, destino)
        if key not in self.rutas_predefinidas:
            raise ValueError(f"No hay rutas definidas entre {origen} y {destino}")
        return self.rutas_predefinidas[key]

    # -------------------------------
    # Estados de la red
    # -------------------------------

    def enlaces_keys(self) -> List[Tuple[str, str]]:
        """
        Devuelve la lista de claves de enlaces (u, v) en un orden fijo.
        """
        return list(self.enlaces.keys())

    def enumerar_estados(self) -> Iterable[Tuple[Dict[Tuple[str, str], bool], float]]:
        """
        Genera todos los estados posibles de congestión en la red junto con su
        probabilidad.

        Para m enlaces, genera 2^m estados. Esto es viable para redes pequeñas.
        """
        keys = self.enlaces_keys()
        m = len(keys)
        # Cada estado es una combinación de True/False para cada enlace
        for bits in itertools.product([False, True], repeat=m):
            estado = {}
            prob = 1.0
            for key, congestado in zip(keys, bits):
                enlace = self.enlaces[key]
                if congestado:
                    prob *= enlace.p_cong
                else:
                    prob *= (1.0 - enlace.p_cong)
                estado[key] = congestado
            yield estado, prob

    def muestrear_estado(self, rng: random.Random) -> Dict[Tuple[str, str], bool]:
        """
        Genera un estado aleatorio de la red de acuerdo con las probabilidades
        de congestión de cada enlace.
        """
        estado: Dict[Tuple[str, str], bool] = {}
        for key, enlace in self.enlaces.items():
            r = rng.random()
            congestado = (r < enlace.p_cong)
            estado[key] = congestado
        return estado

    # -------------------------------
    # Atributos de una ruta dado un estado
    # -------------------------------

    def atributos_ruta_dado_estado(
        self,
        ruta: List[Tuple[str, str]],
        estado: Dict[Tuple[str, str], bool]
    ) -> Tuple[float, float, float]:
        """
        Dada una ruta (lista de enlaces (u,v)) y un estado de red, calcula:

        - latencia total (suma de latencias de los enlaces),
        - pérdida total de la ruta (combinando pérdidas como 1 - product(1 - loss_enlace)),
        - costo total (suma de costos de los enlaces).
        """
        lat_total = 0.0
        costo_total = 0.0
        prob_no_perdida = 1.0  # para combinar pérdidas

        for key in ruta:
            enlace = self.enlaces[key]
            congestado = estado[key]
            lat, loss, costo = enlace.atributos_dado_estado(congestado)
            lat_total += lat
            costo_total += costo
            prob_no_perdida *= (1.0 - loss)

        loss_total = 1.0 - prob_no_perdida
        return lat_total, loss_total, costo_total


# ----------------------------------------------------------------------
# 2. Función de Utilidad multiatributo
# ----------------------------------------------------------------------

class FuncionUtilidad:
    """
    Función de utilidad multiatributo aditiva.
    
    Implementa la función:
        U = - (alpha * latencia + beta * pérdida + gamma * costo)
    
    El signo negativo convierte la minimización de costos en maximización de utilidad.
    Los pesos (alpha, beta, gamma) reflejan las preferencias del usuario sobre cada atributo.
    
    Attributes:
        alpha (float): Peso de la latencia (penalización por ms)
        beta (float): Peso de la pérdida (penalización por unidad de probabilidad)
        gamma (float): Peso del costo (penalización por unidad monetaria)
    
    Example:
        >>> # Priorizar baja latencia
        >>> util = FuncionUtilidad(alpha=5.0, beta=100.0, gamma=1.0)
        >>> u = util.evaluar(latencia=25.0, perdida=0.03, costo=2.0)
        >>> print(f"Utilidad: {u}")
    
    Note:
        - Valores típicos: alpha=1.0, beta=200.0, gamma=5.0
        - Mayor peso = mayor importancia de ese atributo
        - La utilidad resultante es siempre negativa o cero
    """

    def __init__(self, alpha: float, beta: float, gamma: float) -> None:
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def evaluar(self, latencia: float, perdida: float, costo: float) -> float:
        """
        Devuelve la utilidad escalar correspondiente a los atributos de la ruta.
        """
        return -(self.alpha * latencia + self.beta * perdida + self.gamma * costo)


# ----------------------------------------------------------------------
# 3. Agente racional (decisión, utilidad esperada, VPI)
# ----------------------------------------------------------------------

class Agente:
    """
    Agente de encaminamiento racional basado en teoría de decisión bajo incertidumbre.
    
    El agente implementa el principio de Máxima Utilidad Esperada (MEU) y calcula
    el Valor de la Información Perfecta (VPI) para determinar cuándo sondear.
    
    Capacidades:
    - Evaluar rutas con función de utilidad multiatributo
    - Calcular utilidad esperada de cada ruta sin sondeo
    - Calcular utilidad esperada con sondeo perfecto
    - Calcular VPI: VPI = E[max_a U(a,s)] - max_a E[U(a,s)]
    - Tomar decisiones óptimas dado el estado conocido
    
    Attributes:
        red (Red): Modelo de la red con topología y probabilidades
        funcion_utilidad (FuncionUtilidad): Función de evaluación de rutas
        rutas (List[Route]): Conjunto de rutas candidatas
        costo_sondeo (float): Costo de obtener información perfecta
    
    Example:
        >>> agente = Agente(red, funcion_utilidad, rutas, costo_sondeo=15.0)
        >>> vpi = agente.valor_esperado_informacion()
        >>> if vpi > agente.costo_sondeo:
        ...     print("Decisión: SONDEAR")
        ... else:
        ...     print("Decisión: NO SONDEAR")
    
    Reference:
        Russell & Norvig (2020), AIMA 4e, Chapter 15.3: The Value of Information
    """

    def __init__(
        self,
        red: Red,
        funcion_utilidad: FuncionUtilidad,
        rutas: List[List[Tuple[str, str]]],
        costo_sondeo: float
    ) -> None:
        self.red = red
        self.funcion_utilidad = funcion_utilidad
        self.rutas = rutas
        self.costo_sondeo = costo_sondeo

    # -------------------------------
    # Utilidades de rutas
    # -------------------------------

    def utilidad_ruta_dado_estado(
        self,
        ruta: List[Tuple[str, str]],
        estado: Dict[Tuple[str, str], bool]
    ) -> float:
        """
        Utilidad de una ruta en un estado concreto de la red.
        """
        lat, loss, cost = self.red.atributos_ruta_dado_estado(ruta, estado)
        return self.funcion_utilidad.evaluar(lat, loss, cost)

    def utilidad_y_atributos_ruta_dado_estado(
        self,
        ruta: List[Tuple[str, str]],
        estado: Dict[Tuple[str, str], bool]
    ) -> Tuple[float, float, float, float]:
        """
        Devuelve (U, lat, loss, cost) de una ruta en un estado concreto.
        Útil para simulación de episodios.
        """
        lat, loss, cost = self.red.atributos_ruta_dado_estado(ruta, estado)
        u = self.funcion_utilidad.evaluar(lat, loss, cost)
        return u, lat, loss, cost

    # -------------------------------
    # Utilidad esperada (sin sondeo)
    # -------------------------------

    def utilidad_esperada_ruta_sin_sondeo(self, ruta: List[Tuple[str, str]]) -> float:
        """
        Calcula la utilidad esperada de una ruta sin sondeo previo,
        enumerando todos los estados posibles de la red.
        """
        eu = 0.0
        for estado, prob in self.red.enumerar_estados():
            u = self.utilidad_ruta_dado_estado(ruta, estado)
            eu += prob * u
        return eu

    def mejor_ruta_sin_sondeo(self) -> Tuple[List[Tuple[str, str]], float]:
        """
        Devuelve (mejor_ruta, utilidad_esperada) bajo la política sin sondeo,
        es decir, la ruta con mayor utilidad esperada.
        """
        mejor_ruta = None
        mejor_eu = float("-inf")
        for ruta in self.rutas:
            eu = self.utilidad_esperada_ruta_sin_sondeo(ruta)
            if eu > mejor_eu:
                mejor_eu = eu
                mejor_ruta = ruta
        assert mejor_ruta is not None
        return mejor_ruta, mejor_eu

    def utilidad_esperada_decision_sin_sondeo(self) -> float:
        """
        Utilidad esperada total de la decisión óptima sin sondeo.
        """
        _, mejor_eu = self.mejor_ruta_sin_sondeo()
        return mejor_eu

    # -------------------------------
    # Utilidad esperada con sondeo perfecto
    # -------------------------------

    def utilidad_esperada_decision_con_sondeo_perfecto(self) -> float:
        """
        Caso idealizado: el sondeo revela el estado real de todos los enlaces
        sin error (información perfecta).

        EU(probe) = sum_s P(s) * max_r U(r, s) - costo_sondeo
        """
        eu = 0.0
        for estado, prob in self.red.enumerar_estados():
            mejor_u = float("-inf")
            for ruta in self.rutas:
                u = self.utilidad_ruta_dado_estado(ruta, estado)
                if u > mejor_u:
                    mejor_u = u
            eu += prob * mejor_u
        eu -= self.costo_sondeo
        return eu

    # -------------------------------
    # Valor esperado de la información (VPI)
    # -------------------------------

    def valor_esperado_informacion(self) -> float:
        """
        VPI = [sum_s P(s) max_r U(r, s)] - [max_r sum_s P(s) U(r, s)]

        Es el valor esperado de la información perfecta SIN descontar el costo
        del sondeo. Se compara contra costo_sondeo para decidir si vale la pena.
        """
        # Parte 1: sum_s P(s) max_r U(r, s)
        esperado_max_condicional = 0.0
        for estado, prob in self.red.enumerar_estados():
            mejor_u = float("-inf")
            for ruta in self.rutas:
                u = self.utilidad_ruta_dado_estado(ruta, estado)
                if u > mejor_u:
                    mejor_u = u
            esperado_max_condicional += prob * mejor_u

        # Parte 2: max_r sum_s P(s) U(r, s)
        _, eu_mejor_sin_sondeo = self.mejor_ruta_sin_sondeo()

        vpi = esperado_max_condicional - eu_mejor_sin_sondeo
        return vpi

    # -------------------------------
    # Decisión óptima dado estado conocido (para política con sondeo)
    # -------------------------------

    def mejor_ruta_dado_estado(
        self,
        estado: Dict[Tuple[str, str], bool]
    ) -> Tuple[List[Tuple[str, str]], float]:
        """
        Devuelve (mejor_ruta, utilidad) cuando el agente CONOCE el estado actual
        de la red (información perfecta).
        """
        mejor_ruta = None
        mejor_u = float("-inf")
        for ruta in self.rutas:
            u = self.utilidad_ruta_dado_estado(ruta, estado)
            if u > mejor_u:
                mejor_u = u
                mejor_ruta = ruta
        assert mejor_ruta is not None
        return mejor_ruta, mejor_u


# ----------------------------------------------------------------------
# 4. Simulador de políticas de decisión
# ----------------------------------------------------------------------

class Simulador:
    """
    Simula episodios de envío de tráfico bajo distintas políticas:

    - Política 1: no sondear nunca.
    - Política 2: sondear siempre (información perfecta) con costo de sondeo.
    - Política 3: sondear solo si VPI > costo_sondeo.
    """

    def __init__(
        self,
        agente: Agente,
        origen: str,
        destino: str,
        rng_seed: int = 42
    ) -> None:
        self.agente = agente
        self.origen = origen
        self.destino = destino
        self.rng = random.Random(rng_seed)

    # -------------------------------
    # Episodios individuales por política
    # -------------------------------

    def episodio_no_sondeo(self) -> Tuple[float, float, float, float, List[Tuple[str, str]]]:
        """
        Un episodio bajo política "no sondear nunca".

        - Se genera un estado real de la red.
        - El agente elige una ruta usando solo utilidad esperada (sin ver el estado).
        - Se evalúa la utilidad real en ese estado.

        Devuelve (U_real, lat, loss, cost, ruta_elegida).
        """
        # Estado real
        estado_real = self.agente.red.muestrear_estado(self.rng)
        # Decisión óptima sin sondeo
        ruta_elegida, _ = self.agente.mejor_ruta_sin_sondeo()
        # Evaluar atributos reales y utilidad real
        u, lat, loss, cost = self.agente.utilidad_y_atributos_ruta_dado_estado(ruta_elegida, estado_real)
        return u, lat, loss, cost, ruta_elegida

    def episodio_sondeo_siempre(self) -> Tuple[float, float, float, float, List[Tuple[str, str]]]:
        """
        Un episodio bajo política "sondear siempre" (información perfecta).

        - Se genera un estado real de la red.
        - El agente SONDEA (conoce el estado real).
        - Elige la ruta óptima para ese estado.
        - La utilidad real se descuenta por el costo de sondeo.

        Devuelve (U_real_con_costo_sondeo, lat, loss, cost, ruta_elegida).
        """
        estado_real = self.agente.red.muestrear_estado(self.rng)
        ruta_elegida, _ = self.agente.mejor_ruta_dado_estado(estado_real)
        u, lat, loss, cost = self.agente.utilidad_y_atributos_ruta_dado_estado(ruta_elegida, estado_real)
        u -= self.agente.costo_sondeo
        return u, lat, loss, cost, ruta_elegida

    def episodio_sondeo_si_VPI_positivo(self, vpi: float) -> Tuple[float, float, float, float, List[Tuple[str, str]]]:
        """
        Episodio bajo política "sondear solo si VPI > costo_sondeo".

        La decisión de sondear o no se toma fuera (una sola vez),
        y aquí simplemente se ejecuta el episodio con la política resultante.
        """
        if vpi > self.agente.costo_sondeo:
            # Equivalente a política "sondear siempre"
            return self.episodio_sondeo_siempre()
        else:
            # Equivalente a política "no sondear nunca"
            return self.episodio_no_sondeo()

    # -------------------------------
    # Ejecución de múltiples episodios
    # -------------------------------

    def ejecutar_politica(
        self,
        politica: str,
        n_episodios: int
    ) -> Dict:
        """
        Ejecuta n_episodios bajo la política dada y devuelve estadísticas detalladas:

        - utilidad_media, std, percentiles
        - latencia_media, std, percentiles
        - perdida_media, std, percentiles
        - costo_medio, std, percentiles
        - freq_ruta_i (para cada ruta)
        - vpi_usado (solo para política "vpi")
        - datos_completos: listas con todos los valores para análisis posterior
        """
        utilidades = []
        latencias = []
        perdidas = []
        costos = []
        # Conteo de qué rutas se eligieron
        ruta_a_indice: Dict[str, int] = {}
        uso_por_ruta: Dict[int, int] = {}

        # Preparamos un identificador textual para cada ruta
        for idx, ruta in enumerate(self.agente.rutas):
            ruta_str = " -> ".join([ruta[0][0]] + [dst for (_, dst) in ruta])
            ruta_a_indice[ruta_str] = idx
            uso_por_ruta[idx] = 0

        # Para política basada en VPI, lo calculamos una vez
        vpi = None
        if politica == "vpi":
            vpi = self.agente.valor_esperado_informacion()

        for _ in range(n_episodios):
            if politica == "no_probe":
                u, lat, loss, cost, ruta_elegida = self.episodio_no_sondeo()
            elif politica == "probe_always":
                u, lat, loss, cost, ruta_elegida = self.episodio_sondeo_siempre()
            elif politica == "vpi":
                assert vpi is not None
                u, lat, loss, cost, ruta_elegida = self.episodio_sondeo_si_VPI_positivo(vpi)
            else:
                raise ValueError("Política desconocida: use 'no_probe', 'probe_always' o 'vpi'.")

            utilidades.append(u)
            latencias.append(lat)
            perdidas.append(loss)
            costos.append(cost)

            # Ruta elegida → marcamos en el conteo
            ruta_str = " -> ".join([ruta_elegida[0][0]] + [dst for (_, dst) in ruta_elegida])
            idx_ruta = ruta_a_indice[ruta_str]
            uso_por_ruta[idx_ruta] += 1

        # Estadísticas detalladas
        resultados: Dict = {}
        
        # Utilidad
        resultados["utilidad_media"] = statistics.mean(utilidades)
        resultados["utilidad_std"] = statistics.stdev(utilidades) if len(utilidades) > 1 else 0.0
        resultados["utilidad_p25"] = np.percentile(utilidades, 25)
        resultados["utilidad_p50"] = np.percentile(utilidades, 50)
        resultados["utilidad_p75"] = np.percentile(utilidades, 75)
        resultados["utilidad_min"] = min(utilidades)
        resultados["utilidad_max"] = max(utilidades)
        
        # Latencia
        resultados["latencia_media"] = statistics.mean(latencias)
        resultados["latencia_std"] = statistics.stdev(latencias) if len(latencias) > 1 else 0.0
        resultados["latencia_p25"] = np.percentile(latencias, 25)
        resultados["latencia_p50"] = np.percentile(latencias, 50)
        resultados["latencia_p75"] = np.percentile(latencias, 75)
        
        # Pérdida
        resultados["perdida_media"] = statistics.mean(perdidas)
        resultados["perdida_std"] = statistics.stdev(perdidas) if len(perdidas) > 1 else 0.0
        resultados["perdida_p25"] = np.percentile(perdidas, 25)
        resultados["perdida_p50"] = np.percentile(perdidas, 50)
        resultados["perdida_p75"] = np.percentile(perdidas, 75)
        
        # Costo
        resultados["costo_medio"] = statistics.mean(costos)
        resultados["costo_std"] = statistics.stdev(costos) if len(costos) > 1 else 0.0
        resultados["costo_p25"] = np.percentile(costos, 25)
        resultados["costo_p50"] = np.percentile(costos, 50)
        resultados["costo_p75"] = np.percentile(costos, 75)

        # Frecuencias de elección de rutas
        total = float(n_episodios)
        for ruta_str, idx in ruta_a_indice.items():
            resultados[f"freq_{ruta_str}"] = uso_por_ruta[idx] / total

        if vpi is not None:
            resultados["vpi_usado"] = vpi
        
        # Datos completos para visualización
        resultados["datos_completos"] = {
            "utilidades": utilidades,
            "latencias": latencias,
            "perdidas": perdidas,
            "costos": costos
        }

        return resultados


# ----------------------------------------------------------------------
# 5. Definición de escenarios de red
# ----------------------------------------------------------------------

def construir_escenario_base(
    p_cong_A_B: float,
    p_cong_B_D: float,
    p_cong_A_C: float,
    p_cong_C_D: float
) -> Tuple[Red, str, str, List[List[Tuple[str, str]]]]:
    """
    Construye una red pequeña con los nodos A, B, C, D y dos rutas:
        R1: A -> B -> D
        R2: A -> C -> D

    Las probabilidades de congestión se parametrizan para reutilizar
    la misma estructura en diferentes escenarios.
    """
    red = Red()

    # Enlace A-B
    red.agregar_enlace(
        Enlace(
            origen="A",
            destino="B",
            p_cong=p_cong_A_B,
            lat_no_cong=10.0,
            lat_cong=40.0,
            loss_no_cong=0.01,
            loss_cong=0.08,
            costo=1.0,
        )
    )

    # Enlace B-D
    red.agregar_enlace(
        Enlace(
            origen="B",
            destino="D",
            p_cong=p_cong_B_D,
            lat_no_cong=15.0,
            lat_cong=35.0,
            loss_no_cong=0.01,
            loss_cong=0.05,
            costo=1.0,
        )
    )

    # Enlace A-C
    red.agregar_enlace(
        Enlace(
            origen="A",
            destino="C",
            p_cong=p_cong_A_C,
            lat_no_cong=8.0,
            lat_cong=50.0,
            loss_no_cong=0.02,
            loss_cong=0.10,
            costo=2.0,  # por ejemplo, enlace "premium"
        )
    )

    # Enlace C-D
    red.agregar_enlace(
        Enlace(
            origen="C",
            destino="D",
            p_cong=p_cong_C_D,
            lat_no_cong=10.0,
            lat_cong=45.0,
            loss_no_cong=0.02,
            loss_cong=0.08,
            costo=2.0,
        )
    )

    # Definimos las rutas:
    ruta1 = [("A", "B"), ("B", "D")]  # R1: A -> B -> D
    ruta2 = [("A", "C"), ("C", "D")]  # R2: A -> C -> D
    rutas = [ruta1, ruta2]

    # Registramos estas rutas en la red
    red.definir_rutas("A", "D", rutas)

    origen = "A"
    destino = "D"
    return red, origen, destino, rutas


def escenario_1() -> Tuple[Red, str, str, List[List[Tuple[str, str]]], FuncionUtilidad, float]:
    """
    Escenario 1:
    - Red pequeña (A-B-D y A-C-D).
    - Probabilidades de congestión moderadas.
    - Parámetros de utilidad fijos.
    """
    # Probabilidades moderadas
    red, origen, destino, rutas = construir_escenario_base(
        p_cong_A_B=0.3,
        p_cong_B_D=0.2,
        p_cong_A_C=0.5,
        p_cong_C_D=0.4,
    )

    # Pesos de la función de utilidad (ejemplo razonable)
    alpha = 1.0    # penalización por ms de latencia
    beta = 200.0   # penalización por unidad de pérdida
    gamma = 5.0    # penalización por unidad de costo

    funcion_utilidad = FuncionUtilidad(alpha, beta, gamma)

    # Costo de sondeo (por ejemplo, equivalente a 15 unidades de utilidad)
    costo_sondeo = 15.0

    return red, origen, destino, rutas, funcion_utilidad, costo_sondeo


def escenario_2() -> Tuple[Red, str, str, List[List[Tuple[str, str]]], FuncionUtilidad, float]:
    """
    Escenario 2:
    - Misma topología, mayor incertidumbre (p_cong ~ 0.5).
    - El VPI debería tender a ser más alto.
    """
    red, origen, destino, rutas = construir_escenario_base(
        p_cong_A_B=0.5,
        p_cong_B_D=0.5,
        p_cong_A_C=0.5,
        p_cong_C_D=0.5,
    )

    alpha = 1.0
    beta = 200.0
    gamma = 5.0
    funcion_utilidad = FuncionUtilidad(alpha, beta, gamma)
    costo_sondeo = 15.0

    return red, origen, destino, rutas, funcion_utilidad, costo_sondeo


def escenario_3() -> Tuple[Red, str, str, List[List[Tuple[str, str]]], FuncionUtilidad, float]:
    """
    Escenario 3:
    - Misma topología, red casi determinista.
    - Probabilidades de congestión muy bajas o muy altas.
    - El VPI debería ser cercano a cero.
    """
    red, origen, destino, rutas = construir_escenario_base(
        p_cong_A_B=0.05,
        p_cong_B_D=0.05,
        p_cong_A_C=0.95,
        p_cong_C_D=0.95,
    )

    alpha = 1.0
    beta = 200.0
    gamma = 5.0
    funcion_utilidad = FuncionUtilidad(alpha, beta, gamma)
    costo_sondeo = 15.0

    return red, origen, destino, rutas, funcion_utilidad, costo_sondeo


# ----------------------------------------------------------------------
# 6. Programa principal de prueba
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# 6. Funciones de visualización y análisis
# ----------------------------------------------------------------------

def crear_visualizaciones(
    resultados_dict: Dict[str, Dict],
    nombre_escenario: str,
    guardar: bool = True
) -> None:
    """
    Crea visualizaciones comparativas de las políticas.
    
    Args:
        resultados_dict: Diccionario con resultados de cada política
        nombre_escenario: Nombre del escenario para el título
        guardar: Si True, guarda las figuras como archivos PNG
    """
    # Configurar estilo usando constantes
    sns.set_style(PLOT_STYLE)
    plt.rcParams['figure.figsize'] = PLOT_FIGURE_SIZE
    
    fig, axes = plt.subplots(2, 3, figsize=PLOT_FIGURE_SIZE)
    fig.suptitle(f'Análisis Comparativo de Políticas - {nombre_escenario}', 
                 fontsize=16, fontweight='bold')
    
    politicas = list(resultados_dict.keys())
    
    # 1. Distribución de Utilidad
    ax = axes[0, 0]
    for idx, (pol, color) in enumerate(zip(politicas, PLOT_COLORS)):
        datos = resultados_dict[pol]["datos_completos"]["utilidades"]
        ax.hist(datos, bins=30, alpha=0.6, label=pol, color=color, edgecolor='black')
    ax.set_xlabel('Utilidad', fontweight='bold')
    ax.set_ylabel('Frecuencia', fontweight='bold')
    ax.set_title('Distribución de Utilidad', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Boxplot de Utilidad
    ax = axes[0, 1]
    datos_utilidad = [resultados_dict[pol]["datos_completos"]["utilidades"] 
                      for pol in politicas]
    bp = ax.boxplot(datos_utilidad, labels=politicas, patch_artist=True,
                    notch=True, showmeans=True)
    for patch, color in zip(bp['boxes'], PLOT_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel('Utilidad', fontweight='bold')
    ax.set_title('Comparación de Utilidad (Boxplot)', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Latencia promedio
    ax = axes[0, 2]
    latencias_media = [resultados_dict[pol]["latencia_media"] for pol in politicas]
    latencias_std = [resultados_dict[pol]["latencia_std"] for pol in politicas]
    x_pos = np.arange(len(politicas))
    bars = ax.bar(x_pos, latencias_media, yerr=latencias_std, 
                  color=PLOT_COLORS, alpha=0.7, capsize=5, edgecolor='black')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(politicas, rotation=15, ha='right')
    ax.set_ylabel('Latencia (ms)', fontweight='bold')
    ax.set_title('Latencia Media ± Desv. Estándar', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Agregar valores sobre las barras
    for i, (bar, val) in enumerate(zip(bars, latencias_media)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + latencias_std[i],
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Pérdida promedio
    ax = axes[1, 0]
    perdidas_media = [resultados_dict[pol]["perdida_media"] * 100 for pol in politicas]
    perdidas_std = [resultados_dict[pol]["perdida_std"] * 100 for pol in politicas]
    bars = ax.bar(x_pos, perdidas_media, yerr=perdidas_std,
                  color=PLOT_COLORS, alpha=0.7, capsize=5, edgecolor='black')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(politicas, rotation=15, ha='right')
    ax.set_ylabel('Pérdida (%)', fontweight='bold')
    ax.set_title('Pérdida Media ± Desv. Estándar', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, (bar, val) in enumerate(zip(bars, perdidas_media)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + perdidas_std[i],
                f'{val:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # 5. Costo promedio
    ax = axes[1, 1]
    costos_media = [resultados_dict[pol]["costo_medio"] for pol in politicas]
    costos_std = [resultados_dict[pol]["costo_std"] for pol in politicas]
    bars = ax.bar(x_pos, costos_media, yerr=costos_std,
                  color=PLOT_COLORS, alpha=0.7, capsize=5, edgecolor='black')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(politicas, rotation=15, ha='right')
    ax.set_ylabel('Costo', fontweight='bold')
    ax.set_title('Costo Medio ± Desv. Estándar', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, (bar, val) in enumerate(zip(bars, costos_media)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + costos_std[i],
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 6. Frecuencia de uso de rutas
    ax = axes[1, 2]
    # Obtener nombres de rutas
    rutas_nombres = [k.replace("freq_", "") for k in resultados_dict[politicas[0]].keys() 
                     if k.startswith("freq_")]
    
    if rutas_nombres:
        width = 0.25
        x = np.arange(len(rutas_nombres))
        
        for idx, (pol, color) in enumerate(zip(politicas, PLOT_COLORS)):
            freqs = [resultados_dict[pol][f"freq_{ruta}"] * 100 for ruta in rutas_nombres]
            offset = width * (idx - 1)
            ax.bar(x + offset, freqs, width, label=pol, color=color, 
                   alpha=0.7, edgecolor='black')
        
        ax.set_xlabel('Ruta', fontweight='bold')
        ax.set_ylabel('Frecuencia de uso (%)', fontweight='bold')
        ax.set_title('Frecuencia de Uso de Rutas', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(rutas_nombres, rotation=15, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if guardar:
        filename = f"resultados_{nombre_escenario.lower().replace(' ', '_')}.png"
        plt.savefig(filename, dpi=PLOT_DPI, bbox_inches='tight')
        print(f"\n✓ Gráficos guardados en: {filename}")
    
    plt.show()



def imprimir_tabla_comparativa(resultados_dict: Dict[str, Dict], nombre_escenario: str) -> None:
    """
    Imprime una tabla comparativa detallada de todas las políticas.
    """
    print(f"\n{'='*100}")
    print(f"TABLA COMPARATIVA - {nombre_escenario}")
    print(f"{'='*100}\n")
    
    politicas = list(resultados_dict.keys())
    
    # Encabezado
    print(f"{'Métrica':<30} | ", end="")
    for pol in politicas:
        print(f"{pol:^20} | ", end="")
    print()
    print("-" * 100)
    
    # Utilidad
    print(f"{'UTILIDAD':<30} | ", end="")
    for pol in politicas:
        val = resultados_dict[pol]["utilidad_media"]
        print(f"{val:^20.3f} | ", end="")
    print()
    
    print(f"{'  ± Desv. Estándar':<30} | ", end="")
    for pol in politicas:
        val = resultados_dict[pol]["utilidad_std"]
        print(f"{val:^20.3f} | ", end="")
    print()
    
    print(f"{'  Percentil 25':<30} | ", end="")
    for pol in politicas:
        val = resultados_dict[pol]["utilidad_p25"]
        print(f"{val:^20.3f} | ", end="")
    print()
    
    print(f"{'  Mediana (P50)':<30} | ", end="")
    for pol in politicas:
        val = resultados_dict[pol]["utilidad_p50"]
        print(f"{val:^20.3f} | ", end="")
    print()
    
    print(f"{'  Percentil 75':<30} | ", end="")
    for pol in politicas:
        val = resultados_dict[pol]["utilidad_p75"]
        print(f"{val:^20.3f} | ", end="")
    print()
    
    print(f"{'  Rango [min, max]':<30} | ", end="")
    for pol in politicas:
        vmin = resultados_dict[pol]["utilidad_min"]
        vmax = resultados_dict[pol]["utilidad_max"]
        print(f"[{vmin:.1f}, {vmax:.1f}]".center(20) + " | ", end="")
    print()
    
    print("-" * 100)
    
    # Latencia
    print(f"{'LATENCIA (ms)':<30} | ", end="")
    for pol in politicas:
        val = resultados_dict[pol]["latencia_media"]
        print(f"{val:^20.2f} | ", end="")
    print()
    
    print(f"{'  ± Desv. Estándar':<30} | ", end="")
    for pol in politicas:
        val = resultados_dict[pol]["latencia_std"]
        print(f"{val:^20.2f} | ", end="")
    print()
    
    print(f"{'  Mediana':<30} | ", end="")
    for pol in politicas:
        val = resultados_dict[pol]["latencia_p50"]
        print(f"{val:^20.2f} | ", end="")
    print()
    
    print("-" * 100)
    
    # Pérdida
    print(f"{'PÉRDIDA (%)':<30} | ", end="")
    for pol in politicas:
        val = resultados_dict[pol]["perdida_media"] * 100
        print(f"{val:^20.3f} | ", end="")
    print()
    
    print(f"{'  ± Desv. Estándar':<30} | ", end="")
    for pol in politicas:
        val = resultados_dict[pol]["perdida_std"] * 100
        print(f"{val:^20.3f} | ", end="")
    print()
    
    print(f"{'  Mediana':<30} | ", end="")
    for pol in politicas:
        val = resultados_dict[pol]["perdida_p50"] * 100
        print(f"{val:^20.3f} | ", end="")
    print()
    
    print("-" * 100)
    
    # Costo
    print(f"{'COSTO':<30} | ", end="")
    for pol in politicas:
        val = resultados_dict[pol]["costo_medio"]
        print(f"{val:^20.2f} | ", end="")
    print()
    
    print(f"{'  ± Desv. Estándar':<30} | ", end="")
    for pol in politicas:
        val = resultados_dict[pol]["costo_std"]
        print(f"{val:^20.2f} | ", end="")
    print()
    
    print("-" * 100)
    
    # Frecuencia de rutas
    rutas_nombres = [k.replace("freq_", "") for k in resultados_dict[politicas[0]].keys() 
                     if k.startswith("freq_")]
    
    if rutas_nombres:
        print(f"{'FRECUENCIA DE RUTAS (%)':<30} | ", end="")
        print()
        for ruta in rutas_nombres:
            print(f"{'  ' + ruta:<30} | ", end="")
            for pol in politicas:
                val = resultados_dict[pol][f"freq_{ruta}"] * 100
                print(f"{val:^20.2f} | ", end="")
            print()
    
    print("=" * 100)


def analizar_escenario(
    nombre_escenario: str,
    escenario_func,
    n_episodios: int = 5000,
    mostrar_graficos: bool = True,
    guardar_graficos: bool = True
) -> Dict[str, Dict]:
    """
    Analiza un escenario completo: cálculos teóricos, simulación y visualización.
    
    Returns:
        Diccionario con resultados de todas las políticas
    """
    print(f"\n{'#'*100}")
    print(f"# {nombre_escenario.upper()}")
    print(f"{'#'*100}\n")
    
    # Construir escenario
    red, origen, destino, rutas, funcion_utilidad, costo_sondeo = escenario_func()
    
    # Construir agente
    agente = Agente(
        red=red,
        funcion_utilidad=funcion_utilidad,
        rutas=rutas,
        costo_sondeo=costo_sondeo,
    )
    
    # ===== ANÁLISIS TEÓRICO =====
    print("┌" + "─"*98 + "┐")
    print("│" + " ANÁLISIS TEÓRICO (Utilidad Esperada y VPI)".center(98) + "│")
    print("└" + "─"*98 + "┘\n")
    
    print("Rutas disponibles:")
    for idx, ruta in enumerate(rutas):
        ruta_str = " -> ".join([ruta[0][0]] + [dst for (_, dst) in ruta])
        eu = agente.utilidad_esperada_ruta_sin_sondeo(ruta)
        print(f"  Ruta {idx+1}: {ruta_str:<20} → EU(sin sondeo) = {eu:>8.3f}")
    
    mejor_ruta, eu_mejor = agente.mejor_ruta_sin_sondeo()
    mejor_ruta_str = " -> ".join([mejor_ruta[0][0]] + [dst for (_, dst) in mejor_ruta])
    print(f"\n✓ Mejor ruta sin sondeo: {mejor_ruta_str}")
    print(f"  Utilidad esperada: {eu_mejor:.3f}")
    
    eu_con_probe = agente.utilidad_esperada_decision_con_sondeo_perfecto()
    print(f"\n✓ EU(decisión con sondeo perfecto, incluyendo costo): {eu_con_probe:.3f}")
    
    vpi = agente.valor_esperado_informacion()
    print(f"\n✓ VPI (Valor de la Información Perfecta): {vpi:.3f}")
    print(f"✓ Costo de sondeo: {agente.costo_sondeo:.3f}")
    
    if vpi > agente.costo_sondeo:
        print(f"\n→ DECISIÓN RACIONAL: SONDEAR (VPI > costo)")
        print(f"  Ganancia neta esperada: {vpi - agente.costo_sondeo:.3f}")
    else:
        print(f"\n→ DECISIÓN RACIONAL: NO SONDEAR (VPI ≤ costo)")
        print(f"  Pérdida si se sondea: {agente.costo_sondeo - vpi:.3f}")
    
    # ===== SIMULACIÓN =====
    print(f"\n┌" + "─"*98 + "┐")
    print("│" + f" SIMULACIÓN ({n_episodios:,} episodios)".center(98) + "│")
    print("└" + "─"*98 + "┘\n")
    
    simulador = Simulador(agente, origen, destino, rng_seed=42)
    
    print("Ejecutando políticas...")
    res_no_probe = simulador.ejecutar_politica("no_probe", n_episodios)
    print("  ✓ Política 'No sondeo' completada")
    
    res_probe_always = simulador.ejecutar_politica("probe_always", n_episodios)
    print("  ✓ Política 'Sondeo siempre' completada")
    
    res_vpi = simulador.ejecutar_politica("vpi", n_episodios)
    print("  ✓ Política 'Sondeo según VPI' completada")
    
    resultados_dict = {
        "No sondeo": res_no_probe,
        "Sondeo siempre": res_probe_always,
        "Sondeo según VPI": res_vpi
    }
    
    # Imprimir tabla comparativa
    imprimir_tabla_comparativa(resultados_dict, nombre_escenario)
    
    # Crear visualizaciones
    if mostrar_graficos:
        crear_visualizaciones(resultados_dict, nombre_escenario, guardar=guardar_graficos)
    
    return resultados_dict


def comparar_escenarios(
    resultados_escenarios: Dict[str, Dict[str, Dict]]
) -> None:
    """
    Compara resultados entre diferentes escenarios.
    """
    print(f"\n{'='*100}")
    print("COMPARACIÓN ENTRE ESCENARIOS")
    print(f"{'='*100}\n")
    
    escenarios = list(resultados_escenarios.keys())
    
    # Comparar VPI
    print("VPI (Valor de la Información) por escenario:")
    print("-" * 60)
    for esc in escenarios:
        vpi = resultados_escenarios[esc]["Sondeo según VPI"].get("vpi_usado", 0)
        print(f"  {esc:<30}: {vpi:>10.3f}")
    
    print("\n" + "="*100)
    
    # Gráfico comparativo de utilidad entre escenarios
    fig, ax = plt.subplots(figsize=(12, 6))
    
    politicas = ["No sondeo", "Sondeo siempre", "Sondeo según VPI"]
    x = np.arange(len(escenarios))
    width = 0.25
    
    for idx, (pol, color) in enumerate(zip(politicas, PLOT_COLORS)):
        utilidades = [resultados_escenarios[esc][pol]["utilidad_media"] 
                     for esc in escenarios]
        offset = width * (idx - 1)
        ax.bar(x + offset, utilidades, width, label=pol, color=color, 
               alpha=0.7, edgecolor='black')
    
    ax.set_xlabel('Escenario', fontweight='bold', fontsize=12)
    ax.set_ylabel('Utilidad Media', fontweight='bold', fontsize=12)
    ax.set_title('Comparación de Utilidad Media entre Escenarios', 
                 fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(escenarios)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig("comparacion_escenarios.png", dpi=PLOT_DPI, bbox_inches='tight')
    print("\n✓ Gráfico comparativo guardado en: comparacion_escenarios.png")
    plt.show()



# ----------------------------------------------------------------------
# 7. Programa principal de prueba
# ----------------------------------------------------------------------

def main() -> None:
    """
    Ejecuta un análisis completo de los tres escenarios con visualizaciones
    y estadísticas detalladas.
    """
    print("\n" + "="*100)
    print(" PROYECTO: Agente de Encaminamiento Racional con Utilidad Multiatributo ".center(100, "="))
    print(" Basado en Capítulo 15: Making Simple Decisions (Russell & Norvig, AIMA 4e) ".center(100, "="))
    print("="*100)
    
    # Configurar matplotlib para mejor visualización
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Número de episodios para simulación
    n_episodios = 5000
    
    # Analizar cada escenario
    resultados_todos = {}
    
    print("\n" + "▶"*50)
    print("Analizando Escenario 1: Probabilidades Moderadas")
    print("▶"*50)
    resultados_todos["Escenario 1"] = analizar_escenario(
        "Escenario 1: Probabilidades Moderadas",
        escenario_1,
        n_episodios=n_episodios,
        mostrar_graficos=True,
        guardar_graficos=True
    )
    
    print("\n" + "▶"*50)
    print("Analizando Escenario 2: Alta Incertidumbre")
    print("▶"*50)
    resultados_todos["Escenario 2"] = analizar_escenario(
        "Escenario 2: Alta Incertidumbre",
        escenario_2,
        n_episodios=n_episodios,
        mostrar_graficos=True,
        guardar_graficos=True
    )
    
    print("\n" + "▶"*50)
    print("Analizando Escenario 3: Red Casi Determinista")
    print("▶"*50)
    resultados_todos["Escenario 3"] = analizar_escenario(
        "Escenario 3: Red Casi Determinista",
        escenario_3,
        n_episodios=n_episodios,
        mostrar_graficos=True,
        guardar_graficos=True
    )
    
    # Comparación final entre escenarios
    comparar_escenarios(resultados_todos)
    
    print("\n" + "="*100)
    print(" ANÁLISIS COMPLETADO ".center(100, "="))
    print("="*100)
    print("\n✓ Todos los gráficos han sido guardados en el directorio actual.")
    print("✓ Revisa los archivos PNG generados para visualizaciones detalladas.\n")


if __name__ == "__main__":
    main()

