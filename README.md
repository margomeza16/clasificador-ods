# Clasificador de Objetivos de Desarrollo Sostenible (ODS)

Proyecto desarrollado como parte del **Microproyecto 2** de la materia Machine Learning No Supervisado, Maestría en IA (Universidad de los Andes). La idea es simple: dado un texto en español (una propuesta, un artículo, un fragmento de política pública), el modelo identifica a cuál de los 17 ODS de la Agenda 2030 corresponde.

El clasificador alcanza un **86% de accuracy** y un **F1-Macro de 83%** sobre datos que no vio durante el entrenamiento.

**App en vivo:** [clasificador-ods.streamlit.app](https://clasificador-ods-9mb7mgz56gua7uhmmthdbq.streamlit.app/)

---

## Estructura del repositorio

```
clasificador-ods/
├── dataset/
│   └── Datos_textosODS.xlsx           # Dataset OSDG-CD traducido al español
├── notebooks/
│   └── microproyecto2.ipynb           # Notebook con todo el desarrollo
├── app.py                             # Aplicación Streamlit
├── modelo_ods_pipeline.pkl.part0-3    # Modelo particionado para deploy
├── ods_nombres.pkl                    # Mapeo numérico → nombre del ODS
├── requirements.txt
└── README.md
```

---

## Datos

Usamos el **OSDG Community Dataset** (versión 2023), un proyecto colaborativo donde investigadores y voluntarios etiquetan textos según los ODS. Son ~40,000 textos que incluyen documentos de Naciones Unidas, resúmenes de artículos y reportes. Los textos fueron traducidos al español con DeepL y se hizo aumentación de datos con la API de ChatGPT.

En nuestra partición hay **16 clases** (el ODS 17 no tiene registros etiquetados en el dataset):

| # | Objetivo | # | Objetivo |
|---|---|---|---|
| 1 | Fin de la pobreza | 9 | Industria, innovación e infraestructura |
| 2 | Hambre cero | 10 | Reducción de las desigualdades |
| 3 | Salud y bienestar | 11 | Ciudades y comunidades sostenibles |
| 4 | Educación de calidad | 12 | Producción y consumo responsables |
| 5 | Igualdad de género | 13 | Acción por el clima |
| 6 | Agua limpia y saneamiento | 14 | Vida submarina |
| 7 | Energía asequible y no contaminante | 15 | Vida de ecosistemas terrestres |
| 8 | Trabajo decente y crecimiento económico | 16 | Paz, justicia e instituciones sólidas |

---

## Cómo funciona

El pipeline tiene tres etapas:

1. **Preprocesamiento:** tokenización, eliminación de stopwords en español y stemming con SnowballStemmer. Luego se vectoriza con TF-IDF usando unigramas y bigramas.

2. **Reducción de dimensionalidad:** se aplica LSA (TruncatedSVD) con 100 componentes para comprimir el espacio TF-IDF y capturar relaciones semánticas latentes.

3. **Clasificación:** Regresión Logística multiclase, optimizada con GridSearchCV (validación cruzada de 5 folds).

Adicionalmente, se construyó un modelo de tópicos con LSA explorando 17 y 20 componentes para interpretar cualitativamente los temas extraídos y compararlos con los ODS. La coherencia semántica (Cv) se evaluó con Gensim.

---

## Resultados

| Métrica | Valor |
|---|---|
| Accuracy | 86.02% |
| F1-Score (Macro) | 83.23% |

Se verificó el modelo con 5 muestras aleatorias del conjunto de test: 4 de 5 fueron clasificadas correctamente, con confianzas de hasta 98% en las predicciones acertadas.

---

## Cómo correr el proyecto

```bash
# Clonar e instalar
git clone <url-del-repo>
cd clasificador-ods
pip install -r requirements.txt

# Lanzar la app
streamlit run app.py
```

Si quieres reproducir el entrenamiento completo, abre `notebooks/microproyecto2.ipynb`.

---

## Tecnologías

- scikit-learn (TF-IDF, TruncatedSVD, Regresión Logística, GridSearchCV)
- NLTK (tokenización, stopwords, stemming en español)
- Gensim (coherencia de tópicos)
- Streamlit (interfaz web)
- pandas, numpy, matplotlib

---

## Autores

- **Jose Luis Sarta Alvarez** — 202612984
- **Marco Julio Gómez Amado** — 199918013
