import streamlit as st
import joblib
import nltk
import glob
import os
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

nltk.download('stopwords', quiet=True)

# -- Configuración de página --
st.set_page_config(
    page_title="Clasificador ODS",
    page_icon="🌍",
    layout="centered"
)

# -- Colores oficiales de cada ODS (Naciones Unidas) --
ODS_COLORES = {
    1:  "#E5243B", 2:  "#DDA63A", 3:  "#4C9F38", 4:  "#C5192D",
    5:  "#FF3A21", 6:  "#26BDE2", 7:  "#FCC30B", 8:  "#A21942",
    9:  "#FD6925", 10: "#DD1367", 11: "#FD9D24", 12: "#BF8B2E",
    13: "#3F7E44", 14: "#0A97D9", 15: "#56C02B", 16: "#00689D",
    17: "#19486A",
}

ODS_ICONOS = {
    1: "🏚️", 2: "🍽️", 3: "💊", 4: "📖", 5: "⚧️", 6: "💧",
    7: "⚡", 8: "💼", 9: "🏗️", 10: "⚖️", 11: "🏙️", 12: "♻️",
    13: "🌡️", 14: "🐟", 15: "🌳", 16: "🕊️", 17: "🤝",
}

# -- CSS personalizado --
st.markdown("""
<style>
    /* Tipografía general */
    .main .block-container {
        max-width: 740px;
        padding-top: 2.5rem;
    }

    /* Encabezado */
    .header-app {
        text-align: center;
        padding: 1.2rem 0 0.4rem 0;
    }
    .header-app h1 {
        font-size: 1.75rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #1a1a2e;
    }
    .header-app p {
        font-size: 0.95rem;
        color: #555;
        margin-top: 0;
    }

    /* Tarjeta de resultado principal */
    .resultado-card {
        border-radius: 12px;
        padding: 1.6rem 1.8rem;
        margin: 1rem 0;
        color: #fff;
        text-align: center;
    }
    .resultado-card .ods-nombre {
        font-size: 1.35rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .resultado-card .ods-confianza {
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0.5rem 0 0.15rem 0;
    }
    .resultado-card .ods-confianza-label {
        font-size: 0.8rem;
        opacity: 0.85;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Barras de probabilidad */
    .prob-row {
        display: flex;
        align-items: center;
        margin-bottom: 5px;
        font-size: 0.82rem;
    }
    .prob-label {
        width: 62px;
        font-weight: 600;
        color: #333;
        flex-shrink: 0;
    }
    .prob-bar-bg {
        flex: 1;
        background: #eee;
        border-radius: 4px;
        height: 16px;
        overflow: hidden;
        margin: 0 8px;
    }
    .prob-bar {
        height: 100%;
        border-radius: 4px;
        transition: width 0.4s ease;
    }
    .prob-val {
        width: 48px;
        text-align: right;
        color: #555;
        flex-shrink: 0;
        font-size: 0.78rem;
    }

    /* Ejemplos */
    .ejemplo-btn {
        font-size: 0.82rem !important;
        padding: 0.3rem 0.7rem !important;
    }

    /* Separador sutil */
    .sep {
        border: none;
        border-top: 1px solid #e0e0e0;
        margin: 1.5rem 0;
    }

    /* Footer */
    .footer-info {
        text-align: center;
        font-size: 0.75rem;
        color: #999;
        margin-top: 2rem;
        padding-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# -- Preprocesamiento (idéntico al entrenamiento) --
def preprocess_text(text):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(str(text).lower())
    spanish_stopwords = set(stopwords.words('spanish'))
    tokens = [t for t in tokens if t not in spanish_stopwords]
    stemmer = SnowballStemmer('spanish')
    tokens = [stemmer.stem(t) for t in tokens]
    return ' '.join(tokens)


# -- Carga del modelo --
@st.cache_resource
def cargar_recursos():
    if not os.path.exists('modelo_ods_pipeline.pkl'):
        partes = sorted(glob.glob('modelo_ods_pipeline.pkl.part*'))
        if partes:
            with open('modelo_ods_pipeline.pkl', 'wb') as outfile:
                for parte in partes:
                    with open(parte, 'rb') as infile:
                        outfile.write(infile.read())
    pipeline = joblib.load('modelo_ods_pipeline.pkl')
    ods_nombres = joblib.load('ods_nombres.pkl')
    return pipeline, ods_nombres


try:
    pipeline, ODS_NOMBRES = cargar_recursos()
except Exception as e:
    st.error(f"No se pudo cargar el modelo: {e}")
    st.stop()


# -- Textos de ejemplo para que el usuario pruebe rápido --
EJEMPLOS = {
    "Energía renovable": (
        "Implementación de paneles solares fotovoltaicos en comunidades rurales "
        "del Pacífico colombiano para garantizar acceso a energía limpia y reducir "
        "la dependencia de combustibles fósiles en zonas no interconectadas."
    ),
    "Educación inclusiva": (
        "Programa de becas y tutorías para jóvenes de estratos 1 y 2 en municipios "
        "del Cauca, enfocado en reducir la deserción escolar y mejorar los resultados "
        "en pruebas Saber 11 mediante acompañamiento pedagógico personalizado."
    ),
    "Agua y saneamiento": (
        "Construcción de plantas de tratamiento de aguas residuales en tres municipios "
        "del Valle del Cauca para reducir la contaminación de fuentes hídricas y "
        "garantizar agua potable a más de 15.000 familias."
    ),
}


# =====================================================================
#  INTERFAZ
# =====================================================================

# Encabezado
st.markdown("""
<div class="header-app">
    <h1>🌍 Clasificador de Objetivos de Desarrollo Sostenible</h1>
    <p>Ingresa un texto y el modelo identificará a cuál de los 17 ODS se relaciona.</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="sep">', unsafe_allow_html=True)

# Panel de los 16 ODS que evalúa el modelo
with st.expander("Ver los 16 ODS que evalúa el modelo"):
    ods_lista = [
        (1, "Fin de la pobreza"),
        (2, "Hambre cero"),
        (3, "Salud y bienestar"),
        (4, "Educación de calidad"),
        (5, "Igualdad de género"),
        (6, "Agua limpia y saneamiento"),
        (7, "Energía asequible y no contaminante"),
        (8, "Trabajo decente y crecimiento económico"),
        (9, "Industria, innovación e infraestructura"),
        (10, "Reducción de las desigualdades"),
        (11, "Ciudades y comunidades sostenibles"),
        (12, "Producción y consumo responsables"),
        (13, "Acción por el clima"),
        (14, "Vida submarina"),
        (15, "Vida de ecosistemas terrestres"),
        (16, "Paz, justicia e instituciones sólidas"),
    ]
    html_ods = '<div style="display:flex; flex-wrap:wrap; gap:8px; padding:4px 0;">'
    for num, nombre in ods_lista:
        color = ODS_COLORES.get(num, "#333")
        icono = ODS_ICONOS.get(num, "")
        html_ods += (
            f'<span style="background:{color}; color:#fff; padding:6px 12px; '
            f'border-radius:6px; font-size:0.82rem; font-weight:600;">'
            f'{icono} ODS {num}: {nombre}</span>'
        )
    html_ods += '</div>'
    st.markdown(html_ods, unsafe_allow_html=True)
    st.caption("El ODS 17 (Alianzas para lograr los objetivos) no cuenta con registros en el dataset.")

# Ejemplos rápidos
st.markdown("**Prueba con un ejemplo** o escribe tu propio texto:")
cols_ej = st.columns(len(EJEMPLOS))
for i, (nombre, texto_ej) in enumerate(EJEMPLOS.items()):
    with cols_ej[i]:
        if st.button(nombre, key=f"ej_{i}", use_container_width=True):
            st.session_state["texto_input"] = texto_ej

# Área de texto
texto_usuario = st.text_area(
    "Texto a analizar",
    value=st.session_state.get("texto_input", ""),
    height=160,
    placeholder="Pega aquí una propuesta, proyecto, artículo o cualquier texto en español...",
    label_visibility="collapsed",
)

# Botón de clasificación
col_btn = st.columns([1, 2, 1])
with col_btn[1]:
    boton_clasificar = st.button(
        "Clasificar", type="primary", use_container_width=True
    )

# -- Lógica de clasificación --
if boton_clasificar:
    if not texto_usuario.strip():
        st.warning("Escribe o pega un texto antes de clasificar.")
    else:
        with st.spinner("Analizando texto..."):
            # Predicción
            ods_predicho = pipeline.predict([texto_usuario])[0]
            probabilidades = pipeline.predict_proba([texto_usuario])[0]
            clases = list(pipeline.classes_)

            idx_pred = clases.index(ods_predicho)
            confianza = probabilidades[idx_pred] * 100
            nombre_ods = ODS_NOMBRES.get(ods_predicho, f"ODS {ods_predicho}")
            color_ods = ODS_COLORES.get(ods_predicho, "#333")
            icono_ods = ODS_ICONOS.get(ods_predicho, "")

        st.markdown('<hr class="sep">', unsafe_allow_html=True)

        # -- Tarjeta de resultado principal --
        st.markdown(f"""
        <div class="resultado-card" style="background: {color_ods};">
            <div class="ods-confianza-label">Confianza del modelo</div>
            <div class="ods-confianza">{confianza:.1f}%</div>
            <div class="ods-nombre">{icono_ods} {nombre_ods}</div>
        </div>
        """, unsafe_allow_html=True)

        if confianza < 30.0:
            st.info(
                "La confianza es baja. El texto podría abarcar varias temáticas "
                "o no estar lo suficientemente relacionado con un ODS específico."
            )

        # -- Top 5 de probabilidades --
        st.markdown("**Distribución de probabilidades** (las 5 más altas):")

        indices_top = np.argsort(probabilidades)[::-1][:5]
        html_barras = ""
        for idx in indices_top:
            cls = clases[idx]
            prob = probabilidades[idx] * 100
            color = ODS_COLORES.get(cls, "#888")
            ancho = max(prob, 1.5)  # mínimo visible
            html_barras += f"""
            <div class="prob-row">
                <span class="prob-label">ODS {cls}</span>
                <div class="prob-bar-bg">
                    <div class="prob-bar" style="width:{ancho}%; background:{color};"></div>
                </div>
                <span class="prob-val">{prob:.1f}%</span>
            </div>
            """

        st.markdown(html_barras, unsafe_allow_html=True)

# -- Footer --
st.markdown('<hr class="sep">', unsafe_allow_html=True)
st.markdown("""
<div class="footer-info">
    Microproyecto 2 · Machine Learning No Supervisado · Maestría en IA · Universidad de los Andes<br>
    Jose Luis Sarta Alvarez &nbsp;·&nbsp; Marco Julio Gómez Amado
</div>
""", unsafe_allow_html=True)
