"""Simulación Monte Carlo del modelo de flota de bins (Avance 2), en un solo archivo.

Partes:
  1-5) Modelo base sin recuperación (código original del equipo, sin cambios):
       N(c+1) = N(c) - L(c), con L(c) ~ Binomial(N(c), q0), q0 = 1 - (1 - p0)^(1/K).
       Verificación vs. solución analítica, banda P5-P95, histograma, heterogeneidad
       Beta entre 30 productores, escenarios de días sin registro tau, tamaño de
       muestra y sensibilidad a p0.
  6)   Escenarios con recuperación asistida: R_rec(c) ~ Binomial(L(c-1), alpha).
  7)   Balance de tres cuentas N/B/D con rescate, cierre de extravíos y bajas por daño.
  8)   Verificación de reproducibilidad: entorno, CSV, pruebas de identidad
       E = N0*p0 y DE = sqrt(N0*p0*(1-p0)) y huellas SHA-256 de las salidas.

Cada parte usa un generador numpy.random.default_rng(2026) (PCG64) propio, de modo
que sus números no dependen de las demás.
Salidas: ../datos/*.dat y resumen_simulacion.txt (las lee el informe LaTeX) y
         ./reproducibilidad/ (CSV y registro de ejecución).
Uso: python3 sim/simulacion_flota.py     (requiere numpy, ver requirements.txt)
"""
import math
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "datos"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(2026)

N0, K, CR = 1850, 12, 115_000      # bins, ciclos/temporada, $/bin
P_BASE = 0.08                      # pérdida anual (supuesto)
TAU0 = 4.0                         # días por ciclo sin registro hoy (supuesto; ciclo ≈ 6,5 días)
S = 10_000                         # temporadas simuladas


def q_de_p(p, k=K):
    return 1 - (1 - p) ** (1 / k)


def simular(q, n0=N0, k=K, s=S, q_por_bin=None):
    """Devuelve matriz (s, k+1) de bins perdidos acumulados."""
    n = np.full(s, n0)
    acum = np.zeros((s, k + 1), dtype=int)
    for c in range(1, k + 1):
        if q_por_bin is None:
            l = rng.binomial(n, q)
        else:  # heterogeneidad: q distinta por grupo de productores
            l = rng.binomial(n, q_por_bin)
        n = n - l
        acum[:, c] = acum[:, c - 1] + l
    return acum


lineas = []
def log(t=""):
    print(t); lineas.append(t)

# 1) Verificación: Monte Carlo vs. solución analítica
q0 = q_de_p(P_BASE)
acum = simular(q0)
final = acum[:, -1]
analitico = N0 * P_BASE
log(f"q0 por ciclo = {q0:.5f}")
log(f"Analitico perdidos = {analitico:.1f}")
log(f"MC media = {final.mean():.1f}  de = {final.std(ddof=1):.1f}  "
    f"P5 = {np.percentile(final,5):.0f}  P95 = {np.percentile(final,95):.0f}")
log(f"Error relativo media vs analitico = {abs(final.mean()-analitico)/analitico*100:.2f} %")
log(f"Costo medio = {final.mean()*CR/1e6:.2f} M  P5 = {np.percentile(final,5)*CR/1e6:.2f} M  P95 = {np.percentile(final,95)*CR/1e6:.2f} M")

# Banda por ciclo (pgfplots)
with open(OUT / "banda_ciclos.dat", "w") as f:
    f.write("c media p5 p95 analitico\n")
    for c in range(K + 1):
        col = acum[:, c]
        f.write(f"{c} {col.mean():.2f} {np.percentile(col,5):.0f} {np.percentile(col,95):.0f} "
                f"{N0*(1-(1-P_BASE)**(c/K)):.2f}\n")

# Histograma final
bins = np.arange(100, 205, 5)
h, e = np.histogram(final, bins=bins)
with open(OUT / "histograma.dat", "w") as f:
    f.write("x frec\n")
    for x, y in zip(e[:-1], h):
        f.write(f"{x} {y/S*100:.3f}\n")

# 2) Heterogeneidad entre productores (supuesto de independencia/igual q)
G = 30  # grupos de productores
a = 2.0
b = a * (1 - q0) / q0  # Beta con media q0
qg = rng.beta(a, b, size=(S, G))
n_g = np.full((S, G), N0 // G); n_g[:, : N0 % G] += 1
acum_h = np.zeros(S, dtype=int)
for c in range(K):
    l = rng.binomial(n_g, qg)
    n_g -= l; acum_h += l.sum(axis=1)
log(f"Heterogeneo (Beta a=2): media = {acum_h.mean():.1f}  de = {acum_h.std(ddof=1):.1f}  "
    f"P5 = {np.percentile(acum_h,5):.0f}  P95 = {np.percentile(acum_h,95):.0f}")

# 3) Escenarios con registro: reducir tau
lam = -math.log(1 - q0) / TAU0
log(f"lambda calibrado = {lam:.6f} por bin-dia (tau0 = {TAU0} d)")
with open(OUT / "escenarios_tau.dat", "w") as f:
    f.write("tau q p media p5 p95 costo ahorro\n")
    for tau in [4, 3, 2, 1]:
        q = 1 - math.exp(-lam * tau)
        p = 1 - (1 - q) ** K
        fin = simular(q)[:, -1]
        costo = fin.mean() * CR / 1e6
        ahorro = (final.mean() - fin.mean()) * CR / 1e6
        f.write(f"{tau} {q*100:.3f} {p*100:.2f} {fin.mean():.1f} {np.percentile(fin,5):.0f} "
                f"{np.percentile(fin,95):.0f} {costo:.2f} {ahorro:.2f}\n")
        log(f"tau={tau}: q={q*100:.3f}% p={p*100:.2f}% perdidos={fin.mean():.1f} "
            f"[{np.percentile(fin,5):.0f},{np.percentile(fin,95):.0f}] costo={costo:.2f}M ahorro={ahorro:.2f}M")

# 4) Tamaño de muestra para estimar q (piloto)
for E in [0.003, 0.002, 0.001]:
    n = 1.96**2 * q0 * (1 - q0) / E**2
    log(f"Error +-{E*100:.1f} pp: n = {n:.0f} bin-ciclos = {n/N0:.1f} ciclos de flota completa")

# 5) Sensibilidad Monte Carlo a p
with open(OUT / "sensibilidad.dat", "w") as f:
    f.write("p q media p5 p95 costo\n")
    for p in [0.04, 0.08, 0.086, 0.12, 0.20]:
        q = q_de_p(p); fin = simular(q)[:, -1]
        f.write(f"{p*100:.1f} {q*100:.2f} {fin.mean():.1f} {np.percentile(fin,5):.0f} "
                f"{np.percentile(fin,95):.0f} {fin.mean()*CR/1e6:.2f}\n")

(OUT / "resumen_simulacion.txt").write_text("\n".join(lineas) + "\n", encoding="utf-8")


# 6) Escenarios con recuperación asistida (generador propio, semilla 2026)
rng_rec = np.random.default_rng(2026)
P_REC = 0.08


def simular_recuperacion(q, alpha):
    """Ausentes netos N0 - N(K) con R_rec(c) ~ Binomial(L(c-1), alpha), sin compras."""
    n = np.full(S, N0)
    l_prev = np.zeros(S, dtype=int)
    for _ in range(K):
        rec = rng_rec.binomial(l_prev, alpha) if alpha > 0 else 0
        l = rng_rec.binomial(n, q)
        n = n - l + rec
        l_prev = l
    return N0 - n


q0_rec = 1 - (1 - P_REC) ** (1 / K)
lam_rec = -math.log(1 - q0_rec) / TAU0
base_rec = None
lineas_rec = []
with open(OUT / "escenarios_recuperacion.dat", "w") as f:
    f.write("tau alpha q media p5 p95 ahorro\n")
    for tau, alpha in [(4, 0.0), (4, 0.4), (3, 0.0), (2, 0.0), (2, 0.4), (2, 0.6)]:
        q = 1 - math.exp(-lam_rec * tau)
        neta = simular_recuperacion(q, alpha)
        if base_rec is None:
            base_rec = neta.mean()
        ahorro = (base_rec - neta.mean()) * CR / 1e6
        f.write(f"{tau} {alpha*100:.0f} {q*100:.3f} {neta.mean():.1f} "
                f"{np.percentile(neta,5):.0f} {np.percentile(neta,95):.0f} {ahorro:.2f}\n")
        t = (f"tau={tau} alpha={alpha*100:.0f}%: ausentes netos={neta.mean():.1f} "
             f"[{np.percentile(neta,5):.0f},{np.percentile(neta,95):.0f}] valor evitado={ahorro:.2f}M")
        print(t); lineas_rec.append(t)


# 7) Balance de tres cuentas N (disponibles), B (pendientes) y D (bajas)
#    N_{c+1} = N_c - L_c + R_rec_c - H_c ;  B_{c+1} = B_c + L_c - R_rec_c - F_c ;
#    D_{c+1} = D_c + F_c + H_c  (sin compras).  Reglas del escenario:
#    - L_c ~ Bin(N_c - H_c, q0): vencen su plazo y pasan de N a B.
#    - R_rec ~ Bin(cohorte de un ciclo en B, alpha): rescate con evidencia, vuelve a N.
#    - F_c: lo que no se rescató tras un ciclo de búsqueda adicional se cierra como
#      extravío definitivo (plazo de cierre = 2 ciclos desde que venció).
#    - H_c ~ Bin(N_c, h): baja por daño confirmado (h = 0,1 % por ciclo, supuesto).
#    - Al cierre de la temporada lo que sigue abierto queda en B (pendiente).
rng_nbd = np.random.default_rng(2026)
H_DANO = 0.001


def simular_tres_cuentas(q, alpha, h=H_DANO):
    n = np.full(S, N0)
    b_nuevo = np.zeros(S, dtype=int)   # vencieron el ciclo anterior (en búsqueda)
    b_viejo = np.zeros(S, dtype=int)   # no rescatados tras la búsqueda (por cerrar)
    f = np.zeros(S, dtype=int)
    hd = np.zeros(S, dtype=int)
    for _ in range(K):
        rec = rng_nbd.binomial(b_nuevo, alpha) if alpha > 0 else np.zeros(S, dtype=int)
        f += b_viejo                   # cierre validado tras el plazo
        b_viejo = b_nuevo - rec
        dano = rng_nbd.binomial(n, h) if h > 0 else np.zeros(S, dtype=int)
        hd += dano
        l = rng_nbd.binomial(n - dano, q)
        n = n - dano - l + rec
        b_nuevo = l
        assert np.all(n + b_nuevo + b_viejo + f + hd == N0)   # invariante del balance
    return n, b_nuevo + b_viejo, f, hd


lineas_nbd = []
with open(OUT / "balance_tres_cuentas.dat", "w") as fo:
    fo.write("alpha h N B F H D F_p5 F_p95\n")
    for alpha, h in [(0.0, 0.0), (0.0, H_DANO), (0.4, H_DANO), (0.6, H_DANO)]:
        n_k, b_k, f_k, h_k = simular_tres_cuentas(q0, alpha, h)
        d_k = f_k + h_k
        fo.write(f"{alpha*100:.0f} {h*100:.1f} {n_k.mean():.1f} {b_k.mean():.1f} {f_k.mean():.1f} "
                 f"{h_k.mean():.1f} {d_k.mean():.1f} {np.percentile(f_k,5):.0f} {np.percentile(f_k,95):.0f}\n")
        t = (f"alpha={alpha*100:.0f}% h={h*100:.1f}%: N={n_k.mean():.1f} B={b_k.mean():.1f} "
             f"F={f_k.mean():.1f} H={h_k.mean():.1f} (N+B+D = {N0} en las {S} temporadas)")
        print(t); lineas_nbd.append(t)
        if alpha == 0 and h == 0:
            ausentes_sin_rescate = (b_k + f_k)


# 8) Verificación de reproducibilidad
import csv, hashlib, platform
from datetime import datetime, timezone

REPRO = Path(__file__).resolve().parent / "reproducibilidad"
(REPRO / "csv").mkdir(parents=True, exist_ok=True)
TABLAS = ["banda_ciclos.dat", "histograma.dat", "escenarios_tau.dat",
          "sensibilidad.dat", "escenarios_recuperacion.dat", "balance_tres_cuentas.dat",
          "resumen_simulacion.txt"]
reg = [f"Ejecución: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
       f"Python {platform.python_version()}, NumPy {np.__version__}, {platform.system()} {platform.machine()}",
       "Generador: numpy.random.default_rng (PCG64), semilla 2026; un generador para las "
       "partes 1-5, otro para la parte 6 y otro para la parte 7; 10.000 temporadas por escenario.",
       f"SHA-256 simulacion_flota.py: {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}",
       "", "Resumen del modelo base:"] + lineas + ["", "Escenarios con recuperación:"] + lineas_rec + ["", "Balance de tres cuentas:"] + lineas_nbd
reg += ["", "Salidas (SHA-256):"]
for t in TABLAS:
    p = OUT / t
    reg.append(f"  {t:30s} {hashlib.sha256(p.read_bytes()).hexdigest()}")
    if t.endswith(".dat"):
        filas = [ln.split() for ln in p.read_text().splitlines() if ln.strip()]
        with open(REPRO / "csv" / t.replace(".dat", ".csv"), "w", newline="") as fcsv:
            csv.writer(fcsv).writerows(filas)

e_teo, de_teo = N0 * P_BASE, math.sqrt(N0 * P_BASE * (1 - P_BASE))
media_r, de_r = round(final.mean(), 1), round(final.std(ddof=1), 1)
pruebas = [
    ("Media MC vs N0*p0", abs(media_r - e_teo) <= 3 * de_teo / math.sqrt(S) + 0.05,
     f"{media_r} vs {e_teo:.2f} (tolerancia 3 errores estándar = {3*de_teo/math.sqrt(S):.2f})"),
    ("DE MC vs sqrt(N0*p0*(1-p0))", abs(de_r - de_teo) / de_teo <= 0.03,
     f"{de_r} vs {de_teo:.2f} (tolerancia 3 %)"),
    ("Tres cuentas sin rescate ni daño: B+F vs N0*p0",
     abs(ausentes_sin_rescate.mean() - e_teo) <= 3 * de_teo / math.sqrt(S),
     f"{ausentes_sin_rescate.mean():.1f} vs {e_teo:.2f}"),
    ("Invariante N+B+D = N0 en todas las temporadas", True, "verificado con assert en cada ciclo"),
]
reg += ["", "Pruebas de identidad:"]
for nombre, ok, det in pruebas:
    reg.append(f"  [{'OK' if ok else 'FALLA'}] {nombre}: {det}")
(REPRO / "registro_ejecucion.txt").write_text("\n".join(reg) + "\n", encoding="utf-8")
print("\n".join(reg[-5:]))
