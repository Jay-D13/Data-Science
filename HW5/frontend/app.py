import os
from io import BytesIO
from PIL import Image
import validators
import streamlit as st
import requests

# Application rationalisée pour la classification des animaux
HOST = os.environ.get("SERVING_HOST", "0.0.0.0")
PORT = os.environ.get("SERVING_PORT", "8080")
SERVING_URL = os.environ.get("SERVING_URL", "")

if SERVING_URL != "":
    BASE_URL = SERVING_URL
else:
    BASE_URL = f"{HOST}:{PORT}"


def valid_img_extension(url: str) -> bool:
    """
    Déterminer si l’image est une extension d’image valide.
    """
    return url.split(".")[-1].lower() in [
        "jpg",
        "jpeg",
        "png",
    ]

def get_predictions(image_url: str):
    """
    Envoyer l'URL de l'image au backend pour obtenir une prédiction.
    """
    return requests.post(f"{BASE_URL}/predict", json=dict(url=image_url))


st.title("Classification des animaux")
st.write("Charger une image depuis une URL pour la classification.")

url = st.text_input("URL de l'Image", key="url")
img: Image

with st.container():
    if url != "":
        # Valider l'URL
        valid_url = validators.url(url)
        valid_img = valid_img_extension(url)
        if not valid_url:
            st.error("L'URL fournie n'est pas valide.")
            st.stop()
        if not valid_img:
            st.error("L'URL fournie n'est pas une image valide.")
            st.stop()

        # Essayer de télécharger l'image.
        try:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
        except:
            st.error("Il y a eu une erreur lors du téléchargement de l'image.")
            st.stop()

        # Afficher l'image
        st.image(img, caption="Photo téléchargée.")

        # Envoyer l'image au backend pour obtenir une prédiction
        with st.spinner("Veuillez patienter..."):
            result = get_predictions(url)

        if result.status_code == 200:
            st.write(f"Je pense que c'est une image d'un(e) {result.json()['category']}.")
        else:
            st.error("Il y a eu une erreur dans la réception des prédictions")

st.markdown("")