# Simulación de flota de bins retornables — Avance 2

Código y datos de la simulación de Monte Carlo del informe **"Logística inversa de bins retornables en la fruticultura del Maule"** (Avance 2, Módulo Integrador de Ciencias Básicas, Universidad Católica del Maule).

## Cómo reproducir los resultados

```bash
python3 -m pip install -r sim/requirements.txt
python3 sim/simulacion_flota.py
```

Tarda unos 2 segundos. Regenera las tablas de `datos/` y la carpeta `sim/reproducibilidad/`. Con Python 3.13 y NumPy 2.5.3 las tablas salen idénticas byte a byte a las publicadas en el informe.

## Contenido

| Ruta | Qué es |
|---|---|
| `sim/simulacion_flota.py` | Script único de la simulación (ver partes abajo) |
| `sim/requirements.txt` | Dependencias (NumPy) |
| `datos/*.dat` | Tablas que lee el informe LaTeX (pgfplots) |
| `datos/resumen_simulacion.txt` | Resumen numérico del modelo base |
| `sim/reproducibilidad/csv/` | Las mismas tablas en formato CSV |
| `sim/reproducibilidad/registro_ejecucion.txt` | Entorno, resultados, huellas SHA-256 y pruebas de identidad |

## Partes del script

1. Caso base sin recuperación y verificación contra la solución analítica (N0·p0 = 148).
2. Heterogeneidad entre 30 productores, con q_i ~ Beta(2, 2(1−q0)/q0).
3. Escenarios de días sin registro τ (proceso de Poisson).
4. Tamaño de muestra para estimar q.
5. Sensibilidad a la ausencia acumulada p0 (4 a 20 %).
6. Escenarios con recuperación asistida α.
7. Balance de tres cuentas N (disponibles), B (pendientes) y D (bajas), con invariante N + B + D = N0.
8. Verificación de reproducibilidad: CSV, pruebas de identidad y registro.

Parámetros principales: N0 = 1.850 bins, K = 12 ciclos, p0 = 8 %, τ0 = 4 días, C_r = $115.000, S = 10.000 temporadas, semilla 2026 (`numpy.random.default_rng`, PCG64).

**Importante:** los parámetros son supuestos de escenario del equipo, no mediciones de un packing real. Los resultados verifican la coherencia del modelo, no su ajuste a la operación.

## Equipo

Rosario Jesús Torres Villalón, Bastián Agustín Nuñez Urria, Carlos Eduardo Granda Arias, Alejandro Alfonso Inzulza Duarte y Mariana Catalina Morales Muñoz.
Profesor: Nicolas Reyes.
