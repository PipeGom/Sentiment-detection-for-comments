import streamlit as st
import pandas as pd
import plotly.express as px
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import pipeline
import base64 # Utilizado si se quisieran incrustar imágenes en CSS, aunque no se usa directamente aquí.
import os # Importar el módulo os para manejar rutas de archivos

# --- Descargar recursos de NLTK (solo la primera vez que se ejecuta) ---
# Asegura que las stopwords y el tokenizador estén disponibles para el procesamiento de texto.
try:
    nltk.data.find('corpora/stopwords')
except Exception:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except Exception:
    nltk.download('punkt')
# --- AÑADIDO: Descargar 'punkt_tab' que es necesario para word_tokenize en algunos contextos ---
try:
    nltk.data.find('tokenizers/punkt_tab')
except Exception:
    nltk.download('punkt_tab')


# --- CONFIGURACIÓN INICIAL DE STREAMLIT ---
# Configura el diseño de la página a "wide" para una mejor visualización de los gráficos.
# La barra lateral se expande por defecto.
st.set_page_config(layout="wide", page_title="Análisis de Opiniones de Clientes 🗣️",
                   initial_sidebar_state="expanded")

# --- CARGAR ESTILOS CSS EXTERNOS ---
# Función para leer el archivo CSS y codificarlo en base64 para inyectarlo.
@st.cache_data
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Carga el archivo de estilos CSS. Asegúrate de que 'styles.css' esté en la misma carpeta que 'app.py'.
load_css('./assets/styles.css')


# --- FUNCIÓN PARA PROCESAR EL TEXTO (LIMPIEZA, TOKENIZACIÓN, STOPWORDS) ---
def clean_text(text):
    """
    Limpia el texto: lo convierte a minúsculas, elimina caracteres no alfabéticos
    (excepto tildes y 'ñ') y números.
    """
    # Asegura que el input sea string y lo convierte a minúsculas.
    # Si el texto es NaN (valor nulo), lo convierte a una cadena vacía.
    text = str(text).lower() if pd.notna(text) else ""
    # Elimina caracteres que no sean letras (incluyendo tildes y ñ) o espacios.
    # Se mantienen los números por si son relevantes en opiniones (ej. "modelo X200").
    text = re.sub(r'[^a-zA-Z0-9áéíóúüñÁÉÍÓÚÜÑ\s]', '', text)
    return text

def get_word_counts(opinions_df, text_column='opinion'):
    """
    Calcula la frecuencia de las palabras en las opiniones después de limpiar
    y eliminar las stopwords en español.
    """
    all_words = []
    # Carga el conjunto de stopwords en español de NLTK.
    spanish_stopwords = set(stopwords.words('spanish'))

    for opinion in opinions_df[text_column]:
        cleaned_opinion = clean_text(opinion)
        words = word_tokenize(cleaned_opinion) # Divide el texto en palabras
        # Filtra las palabras: no son stopwords y tienen más de 2 caracteres.
        filtered_words = [word for word in words if word not in spanish_stopwords and len(word) > 2]
        all_words.extend(filtered_words)
    return Counter(all_words) # Retorna un contador de la frecuencia de cada palabra.

# --- MODELO DE CLASIFICACIÓN DE SENTIMIENTOS (Hugging Face) ---
# Usa st.cache_resource para cargar el modelo solo una vez y mejorar el rendimiento
# al evitar recargas en cada interacción del usuario.
@st.cache_resource
def load_sentiment_model():
    """
    Carga un modelo de análisis de sentimientos desde Hugging Face.
    El modelo 'nlptown/bert-base-multilingual-uncased-sentiment' es multilingüe
    y adecuado para español.
    Para opciones más ligeras, se podría explorar modelos basados en DistilBERT o TinyBERT
    entrenados para español, aunque podrían tener una menor precisión.
    """
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Carga el modelo al iniciar la aplicación.
sentiment_analyzer = load_sentiment_model()

def classify_sentiment(text):
    """
    Clasifica el sentimiento de un texto dado usando el modelo cargado.
    Mapea las etiquetas del modelo (ej. '1 star', '5 stars') a 'Positivo', 'Negativo', 'Neutro'.
    """
    # Asegura que el texto sea una cadena y no esté vacío para el análisis de sentimiento.
    text_to_analyze = str(text) if pd.notna(text) and text.strip() != "" else "neutro" # Proporciona un texto por defecto si está vacío
    
    result = sentiment_analyzer(text_to_analyze)
    sentiment_label = result[0]['label']
    
    # Mapeo de las etiquetas del modelo a las categorías de sentimiento deseadas.
    if "5 stars" in sentiment_label or "4 stars" in sentiment_label:
        return "Positivo"
    elif "1 star" in sentiment_label:
        return "Negativo"
    else: # Incluye "2 stars" y "3 stars" como neutro.
        return "Neutro"

# --- MODELO PARA RESUMEN (Hugging Face) ---
@st.cache_resource
def load_summarization_model():
    """
    Carga un modelo de resumen de texto desde Hugging Face.
    'sshleifer/distilbart-cnn-12-6' es un modelo popular para resúmenes y es una versión
    destilada de BART, lo que lo hace más ligero que el modelo BART original.
    """
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

summarizer = load_summarization_model()

# --- INTERFAZ DE USUARIO PRINCIPAL ---
st.title("🗣️ Análisis de Opiniones de Clientes")

# --- SECCIÓN DE SUBIDA DE ARCHIVO CSV ---
st.sidebar.header("📤 Cargar Opiniones")
uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV con opiniones (máx. 20)", type=["csv"])

df_opinions = pd.DataFrame() # Inicializa un DataFrame vacío para las opiniones.
default_file_path = 'data/comments.csv'

if uploaded_file is not None:
    try:
        # Intenta leer el archivo CSV subido por el usuario.
        df_opinions = pd.read_csv(uploaded_file, encoding='utf-8')
        st.sidebar.success(f"✅ Se cargaron {len(df_opinions)} opiniones desde el archivo subido.")

    except Exception as e:
        st.sidebar.error(f"⚠️ Error al leer el archivo CSV subido: {e}")
        st.sidebar.info("Asegúrate de que es un CSV válido y tiene una columna 'opinion'.")
        df_opinions = pd.DataFrame() # Vacía el DataFrame si hay un error con el archivo subido.
elif os.path.exists(default_file_path): # Si no se subió archivo, intenta cargar el predeterminado.
    try:
        df_opinions = pd.read_csv(default_file_path, encoding='utf-8')
        st.sidebar.info(f"ℹ️ Se cargaron {len(df_opinions)} opiniones desde el archivo predeterminado: {default_file_path}")
    except Exception as e:
        st.sidebar.error(f"⚠️ Error al leer el archivo predeterminado {default_file_path}: {e}")
        st.sidebar.info("Asegúrate de que el archivo existe y es un CSV válido con una columna 'opinion'.")
        df_opinions = pd.DataFrame() # Vacía el DataFrame si hay un error con el archivo predeterminado.
else:
    st.sidebar.warning("⚠️ No se ha subido ningún archivo y el archivo predeterminado 'data/comments.csv' no fue encontrado.")
    st.info("⬆️ Por favor, sube un archivo CSV con opiniones para comenzar el análisis o asegúrate de que 'data/comments.csv' existe y es accesible.")


# --- VALIDACIÓN Y TRUNCADO DE OPINIONES CARGADAS (ya sea subidas o predeterminadas) ---
if not df_opinions.empty:
    # Limita el número de opiniones a 20 según el requisito.
    if len(df_opinions) > 20:
        st.sidebar.warning("Se han cargado más de 20 opiniones. Solo se usarán las primeras 20.")
        df_opinions = df_opinions.head(20)

    # Verifica si la columna 'opinion' existe en el DataFrame.
    if 'opinion' not in df_opinions.columns:
        st.error("❌ El archivo CSV (subido o predeterminado) debe contener una columna llamada 'opinion'.")
        df_opinions = pd.DataFrame() # Vacía el DataFrame si el formato no es correcto.
    else:
        # Solo muestra las opiniones cargadas si el DataFrame no está vacío y es válido.
        if not df_opinions.empty:
            st.subheader("📝 Opiniones Cargadas:")
            st.dataframe(df_opinions, use_container_width=True) # Muestra las opiniones cargadas.


# --- ANÁLISIS DE DATOS SOLO SI SE HAN CARGADO OPINIONES VÁLIDAS ---
if not df_opinions.empty and 'opinion' in df_opinions.columns: # Doble verificación para asegurar la columna
    st.markdown("---") # Separador visual.
    st.header("📊 Análisis de las Opiniones")

    # --- NUBE DE PALABRAS Y GRÁFICO DE BARRAS DE PALABRAS MÁS REPETIDAS ---
    st.subheader("Palabras Clave ✨")
    word_counts = get_word_counts(df_opinions, text_column='opinion')

    col1, col2 = st.columns(2) # Divide la página en dos columnas para los gráficos.

    with col1:
        st.markdown("##### Nube de Palabras ☁️")
        if word_counts:
            # Genera la nube de palabras a partir de las frecuencias calculadas.
            wordcloud = WordCloud(width=800, height=400, background_color='white',
                                  collocations=False, # Evita agrupar palabras que aparecen juntas.
                                  stopwords=stopwords.words('spanish') # Asegura el uso de stopwords en español.
                                  ).generate_from_frequencies(word_counts)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off") # Oculta los ejes del gráfico.
            st.pyplot(fig) # Muestra el gráfico en Streamlit.
        else:
            st.info("ℹ️ No hay palabras para generar la nube. Asegúrate de que las opiniones tienen contenido.")

    with col2:
        st.markdown("##### 10 Palabras Más Repetidas 📈")
        if word_counts:
            top_10_words = word_counts.most_common(10) # Obtiene las 10 palabras más comunes.
            df_top_words = pd.DataFrame(top_10_words, columns=['Palabra', 'Frecuencia'])
            # Crea un gráfico de barras interactivo con Plotly Express.
            fig_bar = px.bar(df_top_words, x='Palabra', y='Frecuencia',
                             title='Top 10 Palabras Más Repetidas (sin Stopwords)',
                             color_discrete_sequence=px.colors.qualitative.Pastel) # Paleta de colores.
            st.plotly_chart(fig_bar, use_container_width=True) # Muestra el gráfico, ajustando al ancho del contenedor.
        else:
            st.info("ℹ️ No hay palabras para mostrar. Asegúrate de que las opiniones tienen contenido.")

    # --- CLASIFICACIÓN DE SENTIMIENTOS ---
    st.markdown("---")
    st.subheader("Clasificación de Sentimientos 😊😠😐")

    # Aplica la función de clasificación de sentimiento a cada opinión.
    df_opinions['sentimiento'] = df_opinions['opinion'].apply(classify_sentiment)

    st.markdown("##### Sentimiento por Opinión")
    st.dataframe(df_opinions[['opinion', 'sentimiento']], use_container_width=True) # Muestra la tabla.

    # Calcula el porcentaje de opiniones por cada clase de sentimiento.
    sentiment_counts = df_opinions['sentimiento'].value_counts(normalize=True) * 100
    df_sentiment_counts = sentiment_counts.reset_index()
    df_sentiment_counts.columns = ['Sentimiento', 'Porcentaje']

    # Crea un gráfico de pastel interactivo para visualizar los porcentajes.
    fig_pie = px.pie(df_sentiment_counts, values='Porcentaje', names='Sentimiento',
                     title='Porcentaje de Opiniones por Sentimiento',
                     color='Sentimiento',
                     # Mapeo de colores para cada sentimiento.
                     color_discrete_map={'Positivo':'#5CB85C', 'Negativo':'#D9534F', 'Neutro':'#F0AD4E'})
    st.plotly_chart(fig_pie, use_container_width=True)

    # --- INTERACCIÓN CON MODELOS DE LENGUAJE ---
    st.markdown("---")
    st.header("🤖 Interacción con Modelos de Lenguaje")

    # Se elimina la opción de preguntar sobre los comentarios subidos.
    # Solo se mantiene la funcionalidad de análisis de comentarios nuevos.
    st.markdown("##### Analizar un Comentario Nuevo ✍️")
    new_comment = st.text_area("Escribe tu comentario aquí:", height=100, help="Escribe una opinión de cliente para analizar su sentimiento y obtener un resumen.")
    if st.button("Analizar Comentario Nuevo"):
        if new_comment:
            st.write(f"**Comentario Original:** {new_comment}")
            new_sentiment = classify_sentiment(new_comment)
            # Muestra el sentimiento con colores de Streamlit.
            st.write(f"**Sentimiento Detectado:** :green[{new_sentiment}]" if new_sentiment == "Positivo" else f":red[{new_sentiment}]" if new_sentiment == "Negativo" else f":orange[{new_sentiment}]")

            # Genera resumen solo si el comentario es suficientemente largo.
            num_words = len(new_comment.split())
            if num_words > 10: # Solo resumir si tiene al menos 10 palabras
                # Ajusta max_length y min_length dinámicamente
                # max_length será el 75% de las palabras del input, con un máximo de 150
                dynamic_max_length = min(150, int(num_words * 0.75))
                # min_length será el 25% de las palabras del input, con un mínimo de 10
                dynamic_min_length = max(10, int(num_words * 0.25))
                
                # Asegurarse de que min_length no sea mayor que max_length
                if dynamic_min_length > dynamic_max_length:
                    dynamic_min_length = dynamic_max_length - 5 # Asegura una diferencia mínima

                try:
                    summary = summarizer(new_comment, max_length=dynamic_max_length, min_length=dynamic_min_length, do_sample=False)[0]['summary_text']
                    st.write(f"**Resumen:** {summary}")
                except Exception as e:
                    st.error(f"❌ Error al generar el resumen. El comentario podría ser demasiado largo o corto para el modelo. {e}")
            else:
                st.info("ℹ️ El comentario es demasiado corto para generar un resumen significativo.")
        else:
            st.warning("⚠️ Por favor, escribe un comentario para analizar.")

else:
    # Este mensaje solo se mostrará si no se carga ningún archivo (ni subido ni predeterminado)
    # o si el archivo predeterminado no se encuentra.
    st.info("⬆️ Por favor, sube un archivo CSV con opiniones para comenzar el análisis o asegúrate de que 'data/comments.csv' existe y es accesible.")

