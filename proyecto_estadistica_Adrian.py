# PROYECTO 20% - ESTADÍSTICA INFERENCIAL - FACYT UC

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import warnings
warnings.filterwarnings('ignore')


# CONFIGURACIÓN DE PÁGINA

def set_bg_tech():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)), 
                              url("https://www.toptal.com/designers/subtlepatterns/patterns/netglow.png");
            background-color: #0e1117;
            background-attachment: fixed;
            background-size: cover;
        }
        
        /* Estilo para que el contenido resalte sobre el fondo */
        .main .block-container {
            background-color: rgba(14, 17, 23, 0.8);
            border-radius: 15px;
            padding: 2rem;
            margin-top: 2rem;
        }

        h1, h2, h3 {
            color: #4da3ff !important;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }

        .stMarkdown {
            color: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg_tech()

st.set_page_config(
    page_title="Proyecto 20% Estadística Inferencial",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ESTILOS CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a3a5c 0%, #2d6a9f 100%);
        color: white; padding: 2rem; border-radius: 12px;
        text-align: center; margin-bottom: 2rem;
    }
    .section-header {
        background: linear-gradient(90deg, #2d6a9f, #4a9fd4);
        color: white; padding: 1rem 1.5rem; border-radius: 8px;
        font-size: 1.3rem; font-weight: bold; margin: 1.5rem 0 1rem 0;
    }
    .inciso-header {
        background: #f0f7ff; border-left: 5px solid #2d6a9f;
        padding: 0.8rem 1rem; border-radius: 0 8px 8px 0;
        font-weight: bold; color: #1a3a5c; margin: 1rem 0 0.5rem 0;
    }
    .resultado-box {
        background: #e8f5e9; border: 1px solid #4caf50;border-radius: 8px; 
        padding: 1rem; margin: 0.5rem 0;color: #1b5e20;font-weight: bold;
    }
    .conclusion-box {
        background: #fff3e0; border: 1px solid #ff9800;
        border-radius: 8px; padding: 1rem; margin: 0.5rem 0;
    }
    .rechazo-box {
        background-color: #ffebee; border: 1px solid #f44336; border-radius: 8px;
        padding: 1rem; margin: 0.5rem 0; color: #b71c1c; font-weight: bold;
    }
    .acepta-box {
    background-color: #e3f2fd; border: 1px solid #2196f3; border-radius: 8px;
    padding: 1rem; margin: 0.5rem 0; color: #0d47a1; font-weight: bold;
}
    }
    .formula-box {
        background: #fafafa; border: 1px solid #ddd;
        border-radius: 6px; padding: 0.8rem; margin: 0.4rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ENCABEZADO PRINCIPAL
st.markdown("""
<div class="main-header">
    <h1>📊 Proyecto 20% Estadística Inferencial</h1>
    <h3>FACYT – Universidad de Carabobo </h3>
    <p>Adrian Ramirez &nbsp;|&nbsp; C.I: 30871139 </p>
</div>
""", unsafe_allow_html=True)

# PARTE I

st.markdown('<div class="section-header">🏦 PARTE I </div>', unsafe_allow_html=True)

st.markdown("""
**Problema:** El Banco de Venezuela evaluará la arquitectura actual **SwiftVen (A1)** frente a dos nuevas
propuestas: **SwiftFast (A2)** y **SwiftPay (A3)**. Se miden los tiempos de respuesta (latencia en segundos)
con 4 réplicas por arquitectura. Se asume σ = 0.18 s para el diseño experimental.
""")

datos_arquitecturas = np.array([
    [3.30, 3.42, 3.36, 3.34],   # A1 = SwiftVen
    [3.25, 3.15, 3.30, 3.20],   # A2 = SwiftFast
    [3.10, 3.25, 3.18, 3.12],   # A3 = SwiftPay
])

nombres_arquitecturas = ["SwiftVen (A1)", "SwiftFast (A2)", "SwiftPay (A3)"]
sigma_disenio = 0.18   # desviación estándar conocida para el diseño
alpha_anova   = 0.05   # nivel de significancia

# Número de tratamientos y réplicas
num_tratamientos = datos_arquitecturas.shape[0]   # a = 3
num_replicas     = datos_arquitecturas.shape[1]   # n = 4
num_total        = num_tratamientos * num_replicas # N = 12

# Mostrar tabla de datos
df_datos_anova = pd.DataFrame(
    datos_arquitecturas,
    index=nombres_arquitecturas,
    columns=[f"Arquitectura {j+1}" for j in range(num_replicas)]
)
df_datos_anova["Media (ȳᵢ.)"] = datos_arquitecturas.mean(axis=1)
df_datos_anova["Suma (yᵢ.)"]  = datos_arquitecturas.sum(axis=1)

st.subheader("📋 Tabla de Datos — Latencia (segundos)")
st.dataframe(df_datos_anova.style.format("{:.4f}"), use_container_width=True)

# CÁLCULOS CENTRALES ANOVA

# Medias por tratamiento (ȳᵢ.)
medias_tratamientos = datos_arquitecturas.mean(axis=1)   # vector de 3 elementos
# Media global (ȳ..)
media_global = datos_arquitecturas.mean()

#Sumas de Cuadrados
# SST: Suma de Cuadrados Total  = Σᵢ Σⱼ (yᵢⱼ - ȳ..)²
SST = np.sum((datos_arquitecturas - media_global)**2)

# SSTratamientos: Suma de Cuadrados de Tratamientos = n·Σᵢ (ȳᵢ. - ȳ..)²
SS_tratamientos = num_replicas * np.sum((medias_tratamientos - media_global)**2)

# SSE: Suma de Cuadrados del Error = Σᵢ Σⱼ (yᵢⱼ - ȳᵢ.)²
SSE = np.sum((datos_arquitecturas - medias_tratamientos[:, np.newaxis])**2)

# Verificación: SST = SSTratamientos + SSE
assert abs(SST - (SS_tratamientos + SSE)) < 1e-10, "Error en descomposición SS"

#Grados de Libertad
gl_tratamientos = num_tratamientos - 1          # a - 1 = 2
gl_error        = num_tratamientos * (num_replicas - 1)  # a(n-1) = 9
gl_total        = num_total - 1                 # N - 1 = 11

#Cuadrados Medios
CM_tratamientos = SS_tratamientos / gl_tratamientos   # MSTratamientos
CM_error        = SSE / gl_error                      # MSE

#Estadístico F₀
F0_anova = CM_tratamientos / CM_error

# P-Valor
p_valor_anova = 1 - stats.f.cdf(F0_anova, gl_tratamientos, gl_error)

#F crítico
F_critico_anova = stats.f.ppf(1 - alpha_anova, gl_tratamientos, gl_error)

# Residuales eᵢⱼ = yᵢⱼ - ȳᵢ.
residuales = datos_arquitecturas - medias_tratamientos[:, np.newaxis]
residuales_flat = residuales.flatten()   # vector de 12 residuales

#Efectos de tratamiento τᵢ = ȳᵢ. - ȳ..
efectos_tau = medias_tratamientos - media_global

# INCISO 1 — MODELO ESTADÍSTICO, μ, σ², τᵢ

st.markdown('<div class="inciso-header">📌 Inciso 1 — Modelo Estadístico y Estimación de Parámetros</div>', unsafe_allow_html=True)

st.markdown("**Enunciado:** Plantee el modelo estadístico y estime razonablemente: μ, σ², τᵢ.")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Modelo Estadístico:**")
    st.latex(r"y_{ij} = \mu + \tau_i + \varepsilon_{ij}")
    st.markdown("""
    Donde:
    - **yᵢⱼ**: j-ésima observación del i-ésimo tratamiento
    - **μ**: media global poblacional
    - **τᵢ**: efecto del i-ésimo tratamiento
    - *i* = 1,2,3 (arquitecturas) ; *j* = 1,2,3,4 (réplicas)
    """)

with col2:
    st.markdown("**Estimaciones:**")
    st.latex(rf"\hat{{\mu}} = \bar{{x}}{{..}} = {media_global:.4f} \text{{ segundos}}")
    # Estimación de σ²: usamos MSE como estimador insesgado
    st.latex(rf"\hat{{\sigma}}^2 = \frac{{Σᵢ Σⱼ (yᵢⱼ - ȳᵢ)²}}{{a(n-1)}} = \frac{{{SSE:.6f}}}{{{gl_error}}} = {CM_error:.6f}")
    st.latex(rf"\hat{{\sigma}} = {np.sqrt(CM_error):.4f}")
    st.markdown("**Efectos de tratamiento τ̂ᵢ = ȳᵢ. − ȳ..:**")
    for i, (nombre, tau) in enumerate(zip(nombres_arquitecturas, efectos_tau)):
        st.latex(rf"\hat{{\tau}}_{{{i+1}}} \;({nombre.split()[0]})\; = {medias_tratamientos[i]:.4f} - {media_global:.4f} = {tau:+.4f}")

# Verificación: Σ τ̂ᵢ = 0
st.markdown(f"<div class='resultado-box'>✅ Verificación: Σ τ̂ᵢ = {efectos_tau.sum():.10f} ≈ 0 ✓</div>", unsafe_allow_html=True)


# INCISO 2 — ANOVA 8 PASOS

st.markdown('<div class="inciso-header">📌 Inciso 2 — ANOVA: Prueba de Hipótesis (8 Pasos)</div>', unsafe_allow_html=True)
st.markdown("**Enunciado:** ¿Existe al menos una arquitectura con desempeño diferente? (α = 0.05)")

pasos = [
    ("Paso 1", "μ₁, μ₂, μ₃"),
    ("Paso 2", "H₀: μ₁ = μ₂ = μ₃"),
    ("Paso 3", "H₁: Al menos una μᵢ ≠ μⱼ"),
    ("Paso 4", f"α = {alpha_anova}"),
    ("Paso 5", "F₀ = CMt / CMe"),
    ("Paso 6", f"Se rechaza H₀ si F₀ > F_{{α,{gl_tratamientos},{gl_error}}} = {F_critico_anova:.4f}"),
    ("Paso 7", f"F₀ = {CM_tratamientos:.4f} / {CM_error:.4f} = {F0_anova:.4f}  |  P-Valor = {p_valor_anova:.6f}"),
    ("Paso 8 Conclusión", f"Como F₀ = {F0_anova:.4f} {'>' if F0_anova > F_critico_anova else '<='} F_{{α,{gl_tratamientos},{gl_error}}} = {F_critico_anova:.4f} y P = {p_valor_anova:.6f} {'<' if p_valor_anova < alpha_anova else '>='} α = {alpha_anova}, {'SE RECHAZA H₀' if p_valor_anova < alpha_anova else 'NO SE RECHAZA H₀'}."),
]

for titulo, descripcion in pasos:
    st.markdown(f"**{titulo}:** {descripcion}")

# Tabla ANOVA
st.markdown("**📊 Tabla ANOVA:**")
tabla_anova = pd.DataFrame({
    "Fuente de Variación": ["Tratamientos (Arquitecturas)", "Error (Residual)", "Total"],
    "SS": [f"{SS_tratamientos:.4f}", f"{SSE:.4f}", f"{SST:.4f}"],
    "GL": [gl_tratamientos, gl_error, gl_total],
    "CM (MS)": [f"{CM_tratamientos:.4f}", f"{CM_error:.4f}", "—"],
    "F₀": [f"{F0_anova:.4f}", "—", "—"],
    "P-Valor": [f"{p_valor_anova:.6f}", "—", "—"],
})
st.dataframe(tabla_anova, use_container_width=True, hide_index=True)

if p_valor_anova < alpha_anova:
    st.markdown(f"""<div class='rechazo-box'>
    🔴 <b>CONCLUSIÓN:</b> Con F₀ = {F0_anova:.4f} > F_crit = {F_critico_anova:.4f} y P-Valor = {p_valor_anova:.6f} < α = {alpha_anova},
    se <b>RECHAZA H₀</b>. Existe evidencia estadística suficiente para concluir que al menos una arquitectura 
    presenta un desempeño de latencia significativamente diferente (con α = 5%).
    </div>""", unsafe_allow_html=True)

# Gráfico F con región de rechazo
fig_f, ax_f = plt.subplots(figsize=(9, 4))
x_f = np.linspace(0, max(F0_anova + 2, F_critico_anova + 2), 500)
y_f = stats.f.pdf(x_f, gl_tratamientos, gl_error)
ax_f.plot(x_f, y_f, 'navy', lw=2, label=f'F({gl_tratamientos},{gl_error})')
x_rechazo = x_f[x_f >= F_critico_anova]
ax_f.fill_between(x_rechazo, stats.f.pdf(x_rechazo, gl_tratamientos, gl_error),
                  alpha=0.4, color='red', label=f'Región Rechazo (α={alpha_anova})')
ax_f.axvline(F_critico_anova, color='red', ls='--', lw=1.5, label=f'F_crit={F_critico_anova:.3f}')
ax_f.axvline(F0_anova, color='green', ls='-', lw=2, label=f'F₀={F0_anova:.3f}')
ax_f.set_xlabel('Valor F'); ax_f.set_ylabel('Densidad'); ax_f.set_title('Distribución F — ANOVA')
ax_f.legend(); ax_f.grid(alpha=0.3)
st.pyplot(fig_f, use_container_width=True)
plt.close(fig_f)

# INCISO 3 — PRUEBA DE DUNCAN (Comparaciones Múltiples)

st.markdown('<div class="inciso-header">📌 Inciso 3 — Prueba de Comparaciones Múltiples de Duncan</div>', unsafe_allow_html=True)
st.markdown("**Enunciado:** ¿Recomendaría implementar SwiftPay (A3) como estándar para el banco?")

# Duncan usa el MSE del ANOVA
# Error estándar de la diferencia entre medias: s_ȳ = √(MSE/n)
error_estandar_medias = np.sqrt(CM_error / num_replicas)

# Ordenar medias de menor a mayor (menor latencia = mejor desempeño)
orden_idx   = np.argsort(medias_tratamientos)    # índices ordenados
medias_ord  = medias_tratamientos[orden_idx]
nombres_ord = [nombres_arquitecturas[i] for i in orden_idx]

st.markdown(f"""
    **Prueba de Duncan**
-   **Razón**: Se utiliza para realizar comparaciones múltiples tras rechazar la hipótesis nula del ANOVA.
-   **Propósito**: Permite identificar exactamente qué arquitectura es diferente y cuál es la mejor (SwiftPay), agrupándolas por similitud de medias con un nivel de protección intermedio contra el error
""")
st.markdown(f"""
**Fundamento de Duncan:** Usa rangos q(p, ν) de la distribución de rango estudentizado.  
- **MSE** = {CM_error:.4f} | **n** = {num_replicas} | **GL_error** = {gl_error}  
- **S_ȳ** = √(CMe/n) = √({CM_error:.4f}/{num_replicas}) = **{error_estandar_medias:.4f}**  
- **Orden de medias (menor → mayor latencia = mejor rendimiento):**
""")

for i, (nom, med) in enumerate(zip(nombres_ord, medias_ord)):
    st.markdown(f"  {i+1}. **{nom}**: ȳ = {med:.4f} s")

# Valores críticos q de Duncan para p=2,3 con ν=9, α=0.05
# Tabla estándar de Duncan: q(2,9,0.05)≈3.20, q(3,9,0.05)≈3.34
q_duncan = {2: 3.20, 3: 3.34}
R_duncan  = {p: q * error_estandar_medias for p, q in q_duncan.items()}

st.markdown(f"""
**Rangos críticos R_p = q(p, {gl_error}, {alpha_anova}) × S_ȳ (**{error_estandar_medias:.4f}**):**
- p = **2**
- p = **3**
- q(2,9,0.05) ≈ **3.20**
- q(3,9,0.05) ≈ **3.34**
- R₂ = {q_duncan[2]:.2f} × {error_estandar_medias:.4f} = **{R_duncan[2]:.4f}**
- R₃ = {q_duncan[3]:.2f} × {error_estandar_medias:.4f} = **{R_duncan[3]:.4f}**
""")

# Comparaciones por pares (diferencias entre medias ordenadas)
comparaciones = []
etiquetas_ord = [n.split()[0] for n in nombres_ord]
for i in range(len(medias_ord)):
    for j in range(i+1, len(medias_ord)):
        p_rango    = j - i + 1
        diferencia = medias_ord[j] - medias_ord[i]
        R_critico  = R_duncan.get(p_rango, R_duncan[3])
        significativo = diferencia > R_critico
        comparaciones.append({
            "Comparación": f"{etiquetas_ord[j]} vs {etiquetas_ord[i]}",
            "Diferencia": f"{diferencia:.4f}",
            "R_p": f"{R_critico:.4f}",
            "p (rango)": p_rango,
            "¿Significativa?": "✅ SÍ" if significativo else "❌ NO"
        })

df_duncan = pd.DataFrame(comparaciones)
st.dataframe(df_duncan, use_container_width=True, hide_index=True)

# SwiftPay tiene la media más baja (mejor desempeño en latencia)
media_swiftpay  = medias_tratamientos[2]  # índice 2 = SwiftPay
media_swiftven  = medias_tratamientos[0]  # índice 0 = SwiftVen
dif_pay_ven     = media_swiftven - media_swiftpay

st.markdown(f"""<div class='resultado-box'>
✅ <b>RECOMENDACIÓN SOBRE SwiftPay (A3):</b><br>
SwiftPay presenta la <b>menor latencia promedio</b> con ȳ₃ = {media_swiftpay:.4f} s, 
estadísticamente diferente de SwiftVen (diferencia = {dif_pay_ven:.4f} s > R₃ = {R_duncan[3]:.4f} s).
<b>Sí se recomienda la implementación de SwiftPay como nuevo estándar</b> del banco.
</div>""", unsafe_allow_html=True)

# Gráfico de medias con barras de error
fig_medias, ax_med = plt.subplots(figsize=(8, 4))
colores = ['#2196F3', '#FF9800', '#4CAF50']
barras = ax_med.bar(nombres_arquitecturas, medias_tratamientos, color=colores, alpha=0.8, edgecolor='black')
ax_med.errorbar(nombres_arquitecturas, medias_tratamientos,
                yerr=np.sqrt(CM_error), fmt='none', color='black', capsize=6, lw=2)
ax_med.set_ylabel('Latencia promedio (s)'); ax_med.set_title('Medias por Arquitectura ± √CME')
ax_med.set_ylim(3.0, 3.6); ax_med.grid(axis='y', alpha=0.3)
for barra, media in zip(barras, medias_tratamientos):
    ax_med.text(barra.get_x() + barra.get_width()/2, barra.get_height() + 0.005,
                f'{media:.4f}', ha='center', va='bottom', fontweight='bold')
st.pyplot(fig_medias, use_container_width=True)
plt.close(fig_medias)

# INCISO 4 — SUPUESTO DE NORMALIDAD
st.markdown('<div class="inciso-header">📌 Inciso 4 — Supuesto de Normalidad (Shapiro-Wilk)</div>', unsafe_allow_html=True)
st.markdown("*Enunciado:* ¿Apoyaría el supuesto de normalidad de los residuales?")
 
# Prueba Shapiro-Wilk sobre los residuales
stat_shapiro, p_shapiro = stats.shapiro(residuales_flat)
 
data_residuos = {
    'A1': [3.30, 3.42, 3.36, 3.34],
    'A2': [3.25, 3.15, 3.30, 3.20],
    'A3': [3.10, 3.25, 3.18, 3.12]
}

residuos_lista = []
for arc, valores in data_residuos.items():
    media_arc = sum(valores) / len(valores)
    for val in valores:
        residuos_lista.append({
            "Arquitectura": arc,
            "Valor Observado (y)": val,
            "Media (ȳi.)": round(media_arc, 4),
            "Residuo (eij)": round(val - media_arc, 4)
        })

df_res = pd.DataFrame(residuos_lista)

st.markdown(f"""
**Prueba de Shapiro-Wilk**
- **Razón**: Es la prueba más potente y recomendada para verificar el supuesto de normalidad en muestras pequeñas (n < 50), como los 12 datos.
- **Propósito**: Garantiza que los residuos del modelo sigan una distribución de campana de Gauss, lo cual es un requisito obligatorio para que el ANOVA sea válido
""")
st.write("### Análisis de Residuos")
st.table(df_res)

st.markdown(f"""
*Prueba de Shapiro-Wilk* (H₀: Los residuales siguen distribución Normal):
- Calculo del estadistico Shapiro - Wilk
- *W = {stat_shapiro:.4f}* 
- *P-Valor = {p_shapiro:.4f}*
""")


if p_shapiro > alpha_anova:
    st.markdown(f"""<div class='acepta-box'>
    ✅ Como P-Valor = {p_shapiro:.4f} > α = {alpha_anova}, <b>NO se rechaza H₀</b>.
    Los residuales siguen una distribución Normal. El supuesto de normalidad está <b>SOPORTADO</b>.
    </div>""", unsafe_allow_html=True)
else:
    st.markdown(f"""<div class='rechazo-box'>
    ⚠️ Como P-Valor = {p_shapiro:.4f} < α = {alpha_anova}, se rechaza H₀.
    El supuesto de normalidad <b>NO está completamente soportado</b> (muestra pequeña, interpretar con cautela).
    </div>""", unsafe_allow_html=True)


# INCISO 5 — ALEATORIEDAD / INDEPENDENCIA

st.markdown('<div class="inciso-header">📌 Inciso 5 — Aleatoriedad e Independencia (Gráfico de Residuales)</div>', unsafe_allow_html=True)
st.markdown("**Enunciado:** ¿Los datos fueron obtenidos al azar?")

# Prueba de Rachas (Runs Test) sobre signo de residuales
signos = np.sign(residuales_flat)
signos[signos == 0] = 1  # asignar signo positivo a ceros (muy raros)

st.markdown(f"""
**Prueba de Rachas**
- **Razón**: Se utiliza para verificar el supuesto de independencia y aleatoriedad de los datos.
- **Propósito**: Asegura que el orden en que se recolectaron las latencias no influyó en los resultados (evitando patrones o tendencias temporales), confirmando que los datos se obtuvieron al azar.
""")
# Prueba de Rachas de Wald-Wolfowitz
def prueba_rachas(datos):
    """Calcula el estadístico Z de la prueba de rachas"""
    n = len(datos)
    signos_bin = (datos > np.median(datos)).astype(int)
    n1 = signos_bin.sum()
    n2 = n - n1
    rachas = 1 + sum(1 for i in range(1, n) if signos_bin[i] != signos_bin[i-1])
    mu_r  = (2 * n1 * n2) / (n1 + n2) + 1
    var_r = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2)**2 * (n1 + n2 - 1))
    if var_r <= 0:
        return rachas, mu_r, 0, 1.0
    Z_rachas = (rachas - mu_r) / np.sqrt(var_r)
    p_rachas = 2 * (1 - stats.norm.cdf(abs(Z_rachas)))
    return rachas, mu_r, Z_rachas, p_rachas

num_rachas, mu_rachas, Z_rachas, p_rachas = prueba_rachas(residuales_flat)

fig_ind, ax_ind = plt.subplots(figsize=(10, 4))
orden_obs = np.arange(1, num_total + 1)
ax_ind.plot(orden_obs, residuales_flat, 'o-', color='steelblue', markersize=8, lw=1.5)
ax_ind.axhline(0, color='red', lw=2, ls='--', label='Residual = 0')
ax_ind.fill_between(orden_obs, -2*np.sqrt(CM_error), 2*np.sqrt(CM_error),
                    alpha=0.15, color='green', label='±2√MSE')
ax_ind.set_xlabel('Orden de Observación'); ax_ind.set_ylabel('Residual eᵢⱼ')
ax_ind.set_title('Residuales vs Orden de Observación (Aleatoriedad)')
ax_ind.legend(); ax_ind.grid(alpha=0.3)
st.pyplot(fig_ind, use_container_width=True)
plt.close(fig_ind)

st.markdown(f"""
**Prueba de Rachas:**  Número de rachas observadas = {int(num_rachas)} 
- **Esperado** = (2 * N1 * N2)/(N1 + N2)= (2 * 6 * 6)/(6+6) = {mu_rachas:.2f}  
- **Z (valos observado (9) - valor esperado (7))/(desviacion(1.6514)) = {Z_rachas:.4f}**
- **P-Valor (P (Z > 1.2111))= {p_rachas:.4f}**
""")
if p_rachas > alpha_anova:
    st.markdown(f"""<div class='acepta-box'>
    ✅ P-Valor = {p_rachas:.4f} > α = {alpha_anova}: <b>NO se rechaza aleatoriedad</b>.
    Los datos fueron recolectados de forma <b>independiente y aleatoria</b>. El supuesto está SOPORTADO.
    </div>""", unsafe_allow_html=True)
else:
    st.markdown(f"""<div class='conclusion-box'>
    ⚠️ Evidencia marginal de no-aleatoriedad. Verificar el protocolo experimental.
    </div>""", unsafe_allow_html=True)


# INCISO 6 — HOMOCEDASTICIDAD (Levene + Bartlett)

st.markdown('<div class="inciso-header">📌 Inciso 6 — Igualdad de Varianzas (Homocedasticidad)</div>', unsafe_allow_html=True)
st.markdown("**Enunciado:** ¿Las tres arquitecturas presentan la misma varianza?")

# Prueba de Bartlett (asume normalidad)
stat_bartlett, p_bartlett = stats.bartlett(*[datos_arquitecturas[i] for i in range(num_tratamientos)])

# Varianzas por tratamiento
varianzas = np.var(datos_arquitecturas, axis=1, ddof=1)
st.markdown(f"""
**Prueba de Bartlett**
**Razón**: Es la prueba estándar para evaluar la homocedasticidad (igualdad de varianzas) cuando los datos son normales.
**Propósito**: Confirma que la variabilidad de la latencia es constante entre las tres arquitecturas, asegurando que ninguna es más "inestable" que las otras y validando así el uso del estadístico F del ANOVA.
""")
col_var1, col_var2 = st.columns(2)
with col_var1:
    st.markdown("**Varianzas por Arquitectura:**")
    for nom, var in zip(nombres_arquitecturas, varianzas):
        st.markdown(f"  - **{nom}**: s² = {var:.6f}")

with col_var2:
    st.markdown("**Resultados de Pruebas:**")
    st.markdown(f"""
    | Prueba   | Estadístico | P-Valor |
    |----------|------------|---------|
    | Bartlett | {stat_bartlett:.4f}  | {p_bartlett:.4f}  |
    """)

# Gráfico de caja para visualizar varianzas
fig_hom, axes_hom = plt.subplots(1, 2, figsize=(12, 4))
axes_hom[0].boxplot([datos_arquitecturas[i] for i in range(num_tratamientos)],
                    labels=[n.split()[0] for n in nombres_arquitecturas],
                    patch_artist=True,
                    boxprops=dict(facecolor='lightblue'))
axes_hom[0].set_title('Boxplot — Variabilidad por Arquitectura')
axes_hom[0].set_ylabel('Latencia (s)'); axes_hom[0].grid(alpha=0.3)

# Residuales vs valores ajustados
valores_ajustados = np.repeat(medias_tratamientos, num_replicas)
axes_hom[1].scatter(valores_ajustados, residuales_flat, color='steelblue', s=60)
axes_hom[1].axhline(0, color='red', ls='--', lw=2)
axes_hom[1].set_xlabel('Valores Ajustados (ȳᵢ.)'); axes_hom[1].set_ylabel('Residuales')
axes_hom[1].set_title('Residuales vs Valores Ajustados'); axes_hom[1].grid(alpha=0.3)

st.pyplot(fig_hom, use_container_width=True)
plt.close(fig_hom)

if p_bartlett > alpha_anova:
    st.markdown(f"""<div class='acepta-box'>
    ✅ Prueba de Bartlett: P = {p_bartlett:.4f} > α = {alpha_anova}. <b>NO se rechaza H₀ (σ₁²=σ₂²=σ₃²)</b>.
    Las tres arquitecturas presentan <b>varianzas estadísticamente iguales</b>. Homocedasticidad SOPORTADA.
    </div>""", unsafe_allow_html=True)
else:
    st.markdown(f"""<div class='rechazo-box'>
    🔴 Prueba de Bartlett: P = {p_bartlett:.4f} < α = {alpha_anova}. Se rechaza H₀.
    Las varianzas son estadísticamente <b>DIFERENTES</b>. Homocedasticidad NO soportada.
    </div>""", unsafe_allow_html=True)


# INCISO 7 — KRUSKAL-WALLIS (Alternativa No Paramétrica)

st.markdown('<div class="inciso-header">📌 Inciso 7 — Prueba de Kruskal-Wallis (Alternativa No Paramétrica)</div>', unsafe_allow_html=True)
st.markdown("**Enunciado:** Si no se cumplen los supuestos, ¿se mantienen las conclusiones del Inciso 2?")

stat_kw, p_kw = stats.kruskal(*[datos_arquitecturas[i] for i in range(num_tratamientos)])
chi2_critico = stats.chi2.ppf(1 - alpha_anova, df=num_tratamientos - 1)

# --- Tabla de asignación de rangos ---
# Aplanar todos los datos con su etiqueta de arquitectura
todos_valores = []
for i, nombre in enumerate(nombres_arquitecturas):
    for j, val in enumerate(datos_arquitecturas[i]):
        todos_valores.append({'Arquitectura': nombre.split()[0], 'Observación': f'y_{i+1}{j+1}', 'Valor': val})
 
df_valores_all = pd.DataFrame(todos_valores)
# Asignar rangos (método 'average' para empates)
df_valores_all['Rango'] = df_valores_all['Valor'].rank(method='average')
df_valores_all = df_valores_all.sort_values('Valor').reset_index(drop=True)

st.markdown(f"""
**Prueba de Kruskal-Wallis**
- **Razón**: Es la alternativa no paramétrica al ANOVA que se utiliza cuando no se cumplen los supuestos de normalidad o varianza constante.
- **Propósito**: Sirve como prueba de robustez para demostrar que, incluso si los datos no fueran "perfectos", las conclusiones sobre la superioridad de SwiftPay se mantienen firmes basándose en los rangos de los datos.
""")
st.markdown("**📊 Paso 1 — Tabla de Asignación de Rangos (todos los datos combinados, N=12):**")
st.dataframe(df_valores_all.style.format({'Valor': '{:.2f}', 'Rango': '{:.1f}'}), use_container_width=True)
 
# --- Sumatorias de rangos por arquitectura ---
st.markdown("**📊 Paso 2 — Sumatoria de Rangos por Arquitectura (Rᵢ):**")
sumas_rangos = df_valores_all.groupby('Arquitectura')['Rango'].sum()
 
# Ordenar por suma de rangos ascendente (menor latencia = menor rango = más rápido)
sumas_rangos_ord = sumas_rangos.sort_values()
for arq, suma in sumas_rangos_ord.items():
    rangos_arq = df_valores_all[df_valores_all['Arquitectura'] == arq]['Rango'].values
    expr = ' + '.join([f'{r:.1f}' for r in sorted(rangos_arq)])
    velocidad = '(Suma muy baja = muy rápidos)' if suma == sumas_rangos.min() else \
                '(Suma muy alta = muy lentos)' if suma == sumas_rangos.max() else '(Suma media)'
    st.markdown(f"- **{arq}:** {expr} = **{suma:.1f}** {velocidad}")
 
# --- Fórmulas de Kruskal-Wallis ---
st.markdown("**📐 Paso 3 — Fórmulas del Estadístico de Kruskal-Wallis:**")
st.latex(r"H = \frac{12}{N(N+1)} \sum_{i=1}^{a} \frac{R_i^2}{n_i} \;-\; 3(N+1)")
st.latex(r"\text{Donde: } N = \text{total de observaciones}, \; R_i = \text{suma de rangos del grupo } i, \; n_i = \text{tamaño del grupo } i")
 
# Mostrar cálculo explícito
N_kw = num_total
n_i_kw = num_replicas
termino_sum = sum((sumas_rangos[arq.split()[0]] if arq.split()[0] in sumas_rangos.index
                   else sumas_rangos.get(arq.split('(')[0].strip(), 0))**2 / n_i_kw
                  for arq in nombres_arquitecturas)
 
termino_sum2 = 0
for arq in sumas_rangos.index:
    termino_sum2 += (sumas_rangos[arq]**2) / n_i_kw
 
H_manual = (12 / (N_kw * (N_kw + 1))) * termino_sum2 - 3 * (N_kw + 1)
 
st.latex(rf"H = \frac{{12}}{{{N_kw}({N_kw}+1)}} \left(\frac{{R_1^2}}{{{n_i_kw}}} + \frac{{R_2^2}}{{{n_i_kw}}} + \frac{{R_3^2}}{{{n_i_kw}}}\right) - 3({N_kw}+1)")
st.latex(rf"H = \frac{{12}}{{{N_kw * (N_kw+1)}}} \cdot {termino_sum2:.4f} - {3*(N_kw+1)} = {H_manual:.4f}")
 
p_kw = 1 - stats.chi2.cdf(H_manual, df=num_tratamientos - 1)
 
st.markdown(f"""
**Prueba de Kruskal-Wallis** (H₀: Las distribuciones medianas son iguales):
- **H = {H_manual:.4f}**
- **GL = {num_tratamientos - 1}** | **χ²_crit({alpha_anova},{num_tratamientos-1}) = {chi2_critico:.4f}**
- **P-Valor = {p_kw:.6f}**
""")

if p_kw < alpha_anova:
    st.markdown(f"""<div class='rechazo-box'>
    🔴 <b>SE RECHAZA H₀</b>: H = {H_manual:.4f} > χ²_crit = {chi2_critico:.4f} y P = {p_kw:.6f} < α.  
    <b>✅ Las conclusiones del Inciso 2 SE MANTIENEN</b>: incluso sin asumir normalidad ni igualdad de varianzas,
    existe evidencia significativa de que al menos una arquitectura tiene distribución de latencia diferente.
    </div>""", unsafe_allow_html=True)
else:
    st.markdown(f"""<div class='acepta-box'>
    ✅ NO se rechaza H₀ con Kruskal-Wallis (P = {p_kw:.6f} > α). Las conclusiones del Inciso 2 cambiarían.
    </div>""", unsafe_allow_html=True)

st.markdown("---")

#PARTE II — REGRESIÓN LINEAL MÚLTIPLE

st.markdown('<div class="section-header">💻 PARTE II — Regresión Lineal Múltiple: Uso de CPU del Servidor SwiftPay</div>', unsafe_allow_html=True)

st.markdown("""
**Contexto:** Se modela el **Uso de CPU (y)** en función de 4 predictores del tráfico transaccional:
- **x₁**: Peticiones por segundo  
- **x₂**: Tamaño de la trama de datos  
- **x₃**: Latencia de la Bóveda de Datos  
- **x₄**: Consumo de Memoria de los Microservicios  
""")

# --------------------------------------------------------------------------- #
# DATOS HARDCODEADOS — PARTE II
# Tabla de 20 observaciones extraída directamente del PDF
# --------------------------------------------------------------------------- #
datos_cpu = np.array([
    # y      x1    x2    x3    x4
    [9.8,   3.3,  2.8,  3.1,  4.1],
    [12.6,  4.4,  4.9,  3.5,  3.9],
    [11.9,  3.9,  5.3,  4.8,  4.7],
    [13.1,  5.9,  2.6,  3.1,  3.6],
    [13.3,  4.6,  5.1,  5.0,  4.1],
    [13.5,  5.2,  3.2,  3.3,  4.3],
    [10.1,  4.0,  4.0,  3.3,  4.0],
    [13.1,  4.7,  4.5,  3.5,  3.8],
    [10.7,  4.5,  4.1,  3.7,  3.6],
    [11.0,  3.7,  3.6,  3.3,  3.6],
    [13.0,  4.6,  4.6,  3.6,  3.6],
    [11.6,  4.7,  3.5,  3.5,  3.7],
    [12.0,  3.9,  4.6,  3.6,  4.1],
    [11.4,  4.6,  4.0,  3.4,  3.6],
    [12.2,  5.1,  3.6,  3.3,  4.0],
    [12.8,  5.0,  4.4,  3.6,  3.7],
    [12.4,  4.8,  4.4,  3.4,  3.6],
    [13.2,  5.3,  3.5,  3.6,  3.7],
    [10.6,  3.9,  3.8,  3.4,  4.0],
    [7.9,   3.4,  3.8,  3.4,  3.4],
])

# Separar variables
uso_cpu = datos_cpu[:, 0]              # variable respuesta y
X_predictores = datos_cpu[:, 1:]      # matrix de predictores (20×4)
n_obs = len(uso_cpu)                   # n = 20 observaciones
k_vars = X_predictores.shape[1]        # k = 4 variables predictoras

# DataFrame para visualización
columnas_df = ['Uso CPU (y)', 'x1 (Peticiones/s)', 'x2 (Tam. Trama)', 'x3 (Latencia Bóveda)', 'x4 (Mem. Microserv.)']
df_cpu = pd.DataFrame(datos_cpu, columns=columnas_df)

st.subheader("📋 Tabla de Datos — Telemetría del Servidor (n=20)")
st.dataframe(df_cpu.style.format("{:.1f}"), use_container_width=True)

# =============================================================================
# CÁLCULOS CENTRALES RLM — MÍNIMOS CUADRADOS MATRICIALES
# =============================================================================

# Construir matriz de diseño X con columna de unos (intercepto)
unos = np.ones((n_obs, 1))
X_matriz = np.hstack([unos, X_predictores])   # matriz X de dimensión (20×5)

# Estimación por mínimos cuadrados: β̂ = (X'X)⁻¹ X'y
XtX           = X_matriz.T @ X_matriz          # Producto X'X (5×5)
XtX_inv       = np.linalg.inv(XtX)             # Inversa (X'X)⁻¹
Xty           = X_matriz.T @ uso_cpu           # Producto X'y (5×1)
coeficientes  = XtX_inv @ Xty                  # β̂ = (X'X)⁻¹ X'y

# Valores ajustados y residuales del modelo RLM
y_ajustados_rlm = X_matriz @ coeficientes      # ŷ = Xβ̂
residuales_rlm   = uso_cpu - y_ajustados_rlm   # e = y - ŷ

# Media de y
media_y = uso_cpu.mean()

# Sumas de cuadrados para la regresión
SSR_reg  = np.sum((y_ajustados_rlm - media_y)**2)   # SS Regresión
SSE_reg  = np.sum(residuales_rlm**2)                  # SS Error
SST_reg  = np.sum((uso_cpu - media_y)**2)             # SS Total
# Verificación: SST = SSR + SSE
assert abs(SST_reg - (SSR_reg + SSE_reg)) < 1e-8

# Grados de libertad RLM
gl_regresion = k_vars               # k = 4
gl_error_rlm = n_obs - k_vars - 1  # n - k - 1 = 15
gl_total_rlm = n_obs - 1           # n - 1 = 19

# Cuadrados Medios
CM_regresion = SSR_reg / gl_regresion
CM_error_rlm = SSE_reg / gl_error_rlm    # MSE = s²

# Estimación de σ²
sigma2_estimado = CM_error_rlm
sigma_estimado  = np.sqrt(sigma2_estimado)

# F₀ de la regresión
F0_reg     = CM_regresion / CM_error_rlm
p_valor_F0 = 1 - stats.f.cdf(F0_reg, gl_regresion, gl_error_rlm)

# R² y R² ajustado
R2       = SSR_reg / SST_reg
R2_ajust = 1 - (SSE_reg / gl_error_rlm) / (SST_reg / gl_total_rlm)

# Errores estándar de los coeficientes: se(β̂ᵢ) = √(σ² · [(X'X)⁻¹]ᵢᵢ)
varianza_coeficientes = sigma2_estimado * np.diag(XtX_inv)
error_std_coef        = np.sqrt(varianza_coeficientes)

# Estadísticos t para cada coeficiente
t_stats  = coeficientes / error_std_coef
p_t_vals = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=gl_error_rlm))

# Intervalo de confianza del 95% para los coeficientes
t_critico_95 = stats.t.ppf(0.975, df=gl_error_rlm)
IC_inferior  = coeficientes - t_critico_95 * error_std_coef
IC_superior  = coeficientes + t_critico_95 * error_std_coef

# =============================================================================
# RLM INCISO 1 — MÍNIMOS CUADRADOS Y ECUACIÓN DEL MODELO
# =============================================================================
st.markdown('<div class="inciso-header">📌 Inciso 1 — Estimadores β por Mínimos Cuadrados (Método Matricial)</div>', unsafe_allow_html=True)
st.markdown("**Enunciado:** Use el método de mínimos cuadrados para encontrar los estimadores del modelo RLM.")

st.markdown("**Fórmula Matricial:**")
st.latex(r"\hat{\boldsymbol{\beta}} = (\mathbf{X}'\mathbf{X})^{-1}\,\mathbf{X}'\mathbf{y}")

nombres_coef = ['β̂₀ (intercepto)', 'β̂₁ (x₁)', 'β̂₂ (x₂)', 'β̂₃ (x₃)', 'β̂₄ (x₄)']
df_coef = pd.DataFrame({
    'Coeficiente': nombres_coef,
    'Estimación β̂ᵢ': coeficientes,
    'Error Estándar': error_std_coef,
    'Estadístico t': t_stats,
    'P-Valor': p_t_vals,
})
st.dataframe(df_coef.style.format({
    'Estimación β̂ᵢ': '{:.4f}',
    'Error Estándar': '{:.4f}',
    'Estadístico t': '{:.4f}',
    'P-Valor': '{:.4f}'
}), use_container_width=True)

# Ecuación resultante
b0, b1, b2, b3, b4 = coeficientes
st.markdown("**Ecuación del Modelo Ajustado:**")
st.latex(rf"\hat{{y}} = {b0:.4f} + {b1:.4f}\,x_1 + ({b2:.4f})\,x_2 + {b3:.4f}\,x_3 + ({b4:.4f})\,x_4")
st.latex(r"\hat{y} = \hat{\beta}_0 + \hat{\beta}_1 x_1 + \hat{\beta}_2 x_2 + \hat{\beta}_3 x_3 + \hat{\beta}_4 x_4")

# =============================================================================
# RLM INCISO 2 — PREDICCIÓN
# =============================================================================
st.markdown('<div class="inciso-header">📌 Inciso 2 — Predicción del Uso de CPU</div>', unsafe_allow_html=True)
st.markdown("**Enunciado:** Prediga el Uso de CPU para x₁=5.1, x₂=4.7, x₃=4.8, x₄=4.0")

x_pred = np.array([1, 5.1, 4.7, 4.8, 4.0])   # vector con intercepto
y_pred = x_pred @ coeficientes

st.latex(rf"\hat{{y}} = {b0:.4f} + {b1:.4f}(5.1) + ({b2:.4f})(4.7) + {b3:.4f}(4.8) + ({b4:.4f})(4.0)")
st.latex(rf"\hat{{y}} = {b0:.4f} + {b1*5.1:.4f} + ({b2*4.7:.4f}) + {b3*4.8:.4f} + ({b4*4.0:.4f})")

# Intervalo de predicción al 95%
h_pred = x_pred @ XtX_inv @ x_pred   # leverage del punto de predicción
se_pred = sigma_estimado * np.sqrt(1 + h_pred)
IC_pred_inf = y_pred - t_critico_95 * se_pred
IC_pred_sup = y_pred + t_critico_95 * se_pred

st.markdown(f"""<div class='resultado-box'>
✅ <b>Predicción:</b> ŷ = <b>{y_pred:.4f}%</b> de uso de CPU<br>
Intervalo de Predicción al 95%: [{IC_pred_inf:.4f} , {IC_pred_sup:.4f}]
</div>""", unsafe_allow_html=True)

# =============================================================================
# RLM INCISO 3 — ANOVA DE LA REGRESIÓN
# =============================================================================
st.markdown('<div class="inciso-header">📌 Inciso 3 — ANOVA de la Regresión (Bondad de Ajuste)</div>', unsafe_allow_html=True)
st.markdown("**Enunciado:** Use ANOVA para evaluar la bondad de ajuste del modelo.")

tabla_anova_rlm = pd.DataFrame({
    "Fuente": ["Regresión", "Error (Residual)", "Total"],
    "SS": [f"{SSR_reg:.4f}", f"{SSE_reg:.4f}", f"{SST_reg:.4f}"],
    "GL": [gl_regresion, gl_error_rlm, gl_total_rlm],
    "CM (MS)": [f"{CM_regresion:.4f}", f"{CM_error_rlm:.4f}", "—"],
    "F₀": [f"{F0_reg:.4f}", "—", "—"],
    "P-Valor": [f"{p_valor_F0:.6f}", "—", "—"],
})
st.dataframe(tabla_anova_rlm, use_container_width=True, hide_index=True)

F_critico_rlm = stats.f.ppf(0.95, gl_regresion, gl_error_rlm)
st.markdown(f"**F_crit({alpha_anova},{gl_regresion},{gl_error_rlm}) = {F_critico_rlm:.4f}**")

if p_valor_F0 < alpha_anova:
    st.markdown(f"""<div class='rechazo-box'>
    🔴 <b>SE RECHAZA H₀ (β₁=β₂=β₃=β₄=0)</b>: F₀ = {F0_reg:.4f} > F_crit = {F_critico_rlm:.4f}, P = {p_valor_F0:.6f} < α.  
    El modelo de regresión es <b>estadísticamente significativo</b>. Al menos un βᵢ ≠ 0.
    </div>""", unsafe_allow_html=True)

# Gráfico y observado vs y predicho
fig_ajuste, ax_ajust = plt.subplots(figsize=(8, 5))
ax_ajust.scatter(y_ajustados_rlm, uso_cpu, color='steelblue', s=60, label='Observaciones')
lim_min = min(uso_cpu.min(), y_ajustados_rlm.min()) - 0.5
lim_max = max(uso_cpu.max(), y_ajustados_rlm.max()) + 0.5
ax_ajust.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', lw=2, label='Ajuste perfecto')
ax_ajust.set_xlabel('ŷ (Valores Ajustados)'); ax_ajust.set_ylabel('y (Observados)')
ax_ajust.set_title('Valores Observados vs Ajustados — RLM'); ax_ajust.legend()
ax_ajust.grid(alpha=0.3)
st.pyplot(fig_ajuste, use_container_width=True)
plt.close(fig_ajuste)

# =============================================================================
# RLM INCISO 4 — COEFICIENTE DE DETERMINACIÓN R²
# =============================================================================
st.markdown('<div class="inciso-header">📌 Inciso 4 — Coeficiente de Determinación R² y R² Ajustado</div>', unsafe_allow_html=True)
st.markdown("**Enunciado:** Encuentre e interprete el coeficiente de determinación.")

st.latex(r"R^2 = \frac{SS_{Reg}}{SS_T} = 1 - \frac{SS_E}{SS_T}")
st.latex(rf"R^2 = \frac{{{SSR_reg:.4f}}}{{{SST_reg:.4f}}} = {R2:.4f}")
st.latex(r"R^2_{adj} = 1 - \frac{SS_E/(n-k-1)}{SS_T/(n-1)}")
st.latex(rf"R^2_{{adj}} = 1 - \frac{{{SSE_reg:.4f}/{gl_error_rlm}}}{{{SST_reg:.4f}/{gl_total_rlm}}} = {R2_ajust:.4f}")

st.markdown(f"""<div class='resultado-box'>
✅ <b>R² = {R2:.4f}</b> → El modelo explica el <b>{R2*100:.2f}%</b> de la variabilidad total del Uso de CPU.<br>
✅ <b>R²_adj = {R2_ajust:.4f}</b> → Ajustado por el número de predictores, el modelo sigue explicando el <b>{R2_ajust*100:.2f}%</b>.
Un R² > 0.70 indica un ajuste <b>satisfactorio</b> para datos de ingeniería.
</div>""", unsafe_allow_html=True)

# =============================================================================
# RLM INCISO 5 — INTERVALOS DE CONFIANZA AL 95% PARA βᵢ
# =============================================================================
st.markdown('<div class="inciso-header">📌 Inciso 5 — Intervalos de Confianza al 95% para los Parámetros βᵢ</div>', unsafe_allow_html=True)
st.markdown("**Enunciado:** Encuentre un intervalo de confianza del 95% para los parámetros βᵢ.")

st.latex(rf"IC_{{95\%}}(\beta_i) = \hat{{\beta}}_i \pm t_{{\alpha/2,\, n-k-1}} \cdot se(\hat{{\beta}}_i) = \hat{{\beta}}_i \pm {t_critico_95:.4f} \cdot se(\hat{{\beta}}_i)")

df_ic = pd.DataFrame({
    'Parámetro':    [f'β₀', f'β₁(x₁)', f'β₂(x₂)', f'β₃(x₃)', f'β₄(x₄)'],
    'β̂ᵢ':           coeficientes,
    'se(β̂ᵢ)':       error_std_coef,
    'IC Inf. (95%)': IC_inferior,
    'IC Sup. (95%)': IC_superior,
    'Incluye 0?':   ['✅ SÍ' if (lo <= 0 <= hi) else '❌ NO' for lo, hi in zip(IC_inferior, IC_superior)]
})
st.dataframe(df_ic.style.format({
    'β̂ᵢ': '{:.4f}', 'se(β̂ᵢ)': '{:.4f}',
    'IC Inf. (95%)': '{:.4f}', 'IC Sup. (95%)': '{:.4f}'
}), use_container_width=True)

# Gráfico de intervalos de confianza
fig_ic, ax_ic = plt.subplots(figsize=(9, 5))
nombres_param = ['β₀', 'β₁(x₁)', 'β₂(x₂)', 'β₃(x₃)', 'β₄(x₄)']
y_pos = np.arange(len(coeficientes))
ax_ic.barh(y_pos, coeficientes, xerr=[coeficientes - IC_inferior, IC_superior - coeficientes],
           color=['green' if not (lo<=0<=hi) else 'orange' for lo,hi in zip(IC_inferior, IC_superior)],
           alpha=0.7, capsize=5)
ax_ic.axvline(0, color='red', ls='--', lw=2)
ax_ic.set_yticks(y_pos); ax_ic.set_yticklabels(nombres_param)
ax_ic.set_xlabel('Valor del Coeficiente')
ax_ic.set_title('Intervalos de Confianza al 95% para βᵢ\n(Naranja = incluye 0 → no significativo)')
ax_ic.grid(alpha=0.3)
st.pyplot(fig_ic, use_container_width=True)
plt.close(fig_ic)

# =============================================================================
# RLM INCISO 6 — SIGNIFICANCIA INDIVIDUAL (Prueba t)
# =============================================================================
st.markdown('<div class="inciso-header">📌 Inciso 6 — Significancia Individual de cada Xᵢ (Prueba t, α=5%)</div>', unsafe_allow_html=True)
st.markdown("**Enunciado:** Pruebe con 5% de significancia si el aporte de cada variable Xᵢ es significativo.")

st.markdown(f"**H₀: βᵢ = 0** vs **H₁: βᵢ ≠ 0** | t_crit(α/2, {gl_error_rlm}) = ±{t_critico_95:.4f}")

df_t = pd.DataFrame({
    'Variable': ['β₀', 'x₁', 'x₂', 'x₃', 'x₄'],
    'β̂ᵢ':       coeficientes,
    't₀':        t_stats,
    'P-Valor':   p_t_vals,
    'Significativo (α=0.05)': ['✅ SÍ' if p < alpha_anova else '❌ NO' for p in p_t_vals],
    'Decisión': ['Rechaza H₀' if p < alpha_anova else 'No rechaza H₀' for p in p_t_vals]
})
st.dataframe(df_t.style.format({
    'β̂ᵢ': '{:.4f}', 't₀': '{:.4f}', 'P-Valor': '{:.4f}'
}), use_container_width=True)

# =============================================================================
# RLM INCISO 7 — SELECCIÓN DEL MEJOR MODELO
# =============================================================================
st.markdown('<div class="inciso-header">📌 Inciso 7 — Selección del Mejor Modelo (Justificación)</div>', unsafe_allow_html=True)
st.markdown("**Enunciado:** ¿Cuál modelo recomendaría? Justifique.")

# Ajustar modelos reducidos para comparación
# Identificar variables significativas del modelo completo
variables_sig_idx = [i for i, p in enumerate(p_t_vals[1:]) if p < alpha_anova]  # excluir intercepto
variables_sig_nombres = [f'x{i+1}' for i in variables_sig_idx]

st.markdown(f"""
**Criterio de Selección:** Se compararán el **Modelo Completo** (4 variables) con un 
**Modelo Reducido** usando solo las variables estadísticamente significativas.
""")

# Construir tabla comparativa de modelos
resultados_modelos = []

# Modelo completo (todas las variables)
resultados_modelos.append({
    'Modelo': 'Completo (x₁,x₂,x₃,x₄)',
    'Variables': 4,
    'R²': R2,
    'R²_adj': R2_ajust,
    'F₀': F0_reg,
    'P-Valor F': p_valor_F0,
    'AIC': n_obs * np.log(SSE_reg/n_obs) + 2*(k_vars+1),
    'MSE': CM_error_rlm
})

# Modelos con subconjuntos de variables (solo variables significativas)
for combo_size in range(1, k_vars):
    from itertools import combinations
    for combo in combinations(range(k_vars), combo_size):
        X_sub = np.hstack([unos, X_predictores[:, combo]])
        coef_sub = np.linalg.lstsq(X_sub, uso_cpu, rcond=None)[0]
        y_hat_sub = X_sub @ coef_sub
        sse_sub = np.sum((uso_cpu - y_hat_sub)**2)
        sst_sub = SST_reg
        r2_sub  = 1 - sse_sub/sst_sub
        k_sub   = len(combo)
        r2_adj_sub = 1 - (sse_sub/(n_obs-k_sub-1)) / (sst_sub/(n_obs-1))
        ssr_sub = sst_sub - sse_sub
        cm_reg_sub = ssr_sub / k_sub
        cm_err_sub = sse_sub / (n_obs-k_sub-1)
        f0_sub = cm_reg_sub / cm_err_sub
        p_f_sub = 1 - stats.f.cdf(f0_sub, k_sub, n_obs-k_sub-1)
        aic_sub = n_obs * np.log(sse_sub/n_obs) + 2*(k_sub+1)
        nombre_combo = 'Reducido (' + ','.join([f'x{c+1}' for c in combo]) + ')'
        resultados_modelos.append({
            'Modelo': nombre_combo,
            'Variables': k_sub,
            'R²': r2_sub,
            'R²_adj': r2_adj_sub,
            'F₀': f0_sub,
            'P-Valor F': p_f_sub,
            'AIC': aic_sub,
            'MSE': cm_err_sub
        })

df_modelos = pd.DataFrame(resultados_modelos)
# Ordenar por R² ajustado descendente
df_modelos_ord = df_modelos.sort_values('R²_adj', ascending=False).head(10)
st.dataframe(df_modelos_ord.style.format({
    'R²': '{:.4f}', 'R²_adj': '{:.4f}', 'F₀': '{:.3f}',
    'P-Valor F': '{:.4f}', 'AIC': '{:.2f}', 'MSE': '{:.4f}'
}), use_container_width=True)

# Identificar mejor modelo por R² ajustado
mejor_modelo = df_modelos_ord.iloc[0]

st.markdown(f"""<div class='resultado-box'>
✅ <b>MODELO RECOMENDADO: {mejor_modelo['Modelo']}</b><br>
- R²_adj = {mejor_modelo['R²_adj']:.4f} (máximo entre todos los modelos evaluados)<br>
- AIC = {mejor_modelo['AIC']:.2f} (menor valor = mejor penalización por complejidad)<br>
- F₀ = {mejor_modelo['F₀']:.3f} con P-Valor = {mejor_modelo['P-Valor F']:.4f}<br><br>
<b>Justificación:</b> Se utiliza el criterio del R² ajustado y AIC para balancear la bondad de ajuste 
con la parsimonia del modelo. El principio de parsimonia indica preferir el modelo más simple que 
mantenga un poder explicativo comparable al modelo completo.
</div>""", unsafe_allow_html=True)

# Diagnóstico de residuales del modelo completo
st.markdown("### 🔬 Diagnóstico de Residuales — Modelo Completo")
fig_diag, axes_d = plt.subplots(2, 2, figsize=(14, 10))

# 1. Residuales vs Fitted
axes_d[0,0].scatter(y_ajustados_rlm, residuales_rlm, color='steelblue', s=50)
axes_d[0,0].axhline(0, color='red', ls='--', lw=2)
axes_d[0,0].set_xlabel('ŷ (Valores Ajustados)'); axes_d[0,0].set_ylabel('Residuales')
axes_d[0,0].set_title('Residuales vs Valores Ajustados'); axes_d[0,0].grid(alpha=0.3)

# 2. Q-Q Plot de residuales RLM
(osm_r, osr_r), (slope_r, intercept_r, _) = stats.probplot(residuales_rlm, dist='norm')
axes_d[0,1].scatter(osm_r, osr_r, color='steelblue', s=50)
x_qq = np.array([osm_r.min(), osm_r.max()])
axes_d[0,1].plot(x_qq, slope_r*x_qq + intercept_r, 'r-', lw=2)
axes_d[0,1].set_title('Q-Q Plot — Residuales RLM'); axes_d[0,1].grid(alpha=0.3)

# 3. Escala-Localización (Scale-Location)
axes_d[1,0].scatter(y_ajustados_rlm, np.sqrt(np.abs(residuales_rlm)), color='orange', s=50)
axes_d[1,0].set_xlabel('ŷ'); axes_d[1,0].set_ylabel('√|eᵢ|')
axes_d[1,0].set_title('Escala-Localización (Homocedasticidad)'); axes_d[1,0].grid(alpha=0.3)

# 4. Residuales vs Orden
axes_d[1,1].plot(np.arange(1, n_obs+1), residuales_rlm, 'o-', color='green', markersize=6)
axes_d[1,1].axhline(0, color='red', ls='--', lw=2)
axes_d[1,1].set_xlabel('Orden de Observación'); axes_d[1,1].set_ylabel('Residuales')
axes_d[1,1].set_title('Residuales vs Orden (Independencia)'); axes_d[1,1].grid(alpha=0.3)

plt.tight_layout()
st.pyplot(fig_diag, use_container_width=True)
plt.close(fig_diag)

# Shapiro-Wilk para RLM
stat_sw_rlm, p_sw_rlm = stats.shapiro(residuales_rlm)
st.markdown(f"""
**Prueba de Shapiro-Wilk en Residuales RLM:** W = {stat_sw_rlm:.4f} | P-Valor = {p_sw_rlm:.4f}  
{'✅ Normalidad SOPORTADA (P > 0.05)' if p_sw_rlm > 0.05 else '⚠️ Posible desviación de normalidad (P < 0.05)'}
""")

# RESUMEN EJECUTIVO FINAL

st.markdown("---")
st.markdown('<div class="section-header">📝 Resumen Ejecutivo del Proyecto</div>', unsafe_allow_html=True)

col_r1, col_r2 = st.columns(2)
with col_r1:
    st.markdown(f"""
    **PARTE I — ANOVA:**
    - **F₀ = {F0_anova:.4f}** > F_crit = {F_critico_anova:.4f} → **Se rechaza H₀**
    - P-Valor = {p_valor_anova:.6f} < α = 0.05
    - **Conclusión:** Existen diferencias significativas entre arquitecturas
    - **Duncan:** SwiftPay (ȳ={medias_tratamientos[2]:.4f}s) es estadísticamente superior
    - **Kruskal-Wallis:** Confirma conclusión (H={H_manual:.4f}, P={p_kw:.4f})
    """)

with col_r2:
    st.markdown(f"""
    **PARTE II — RLM:**
    - **Ecuación:** ŷ = {b0:.3f} {'+' if b1>=0 else ''}{b1:.3f}x₁ {'+' if b2>=0 else ''}{b2:.3f}x₂ {'+' if b3>=0 else ''}{b3:.3f}x₃ {'+' if b4>=0 else ''}{b4:.3f}x₄
    - **R² = {R2:.4f}** | **R²_adj = {R2_ajust:.4f}**
    - **F₀ = {F0_reg:.4f}**, P = {p_valor_F0:.6f} → Modelo significativo
    - **Predicción** (x₁=5.1, x₂=4.7, x₃=4.8, x₄=4.0): **ŷ = {y_pred:.4f}%**
    """)

st.markdown("""
---
*Aplicación desarrollada para el Proyecto 20% — Estadística Inferencial | FACYT – Universidad de Carabobo 2026*
""")
