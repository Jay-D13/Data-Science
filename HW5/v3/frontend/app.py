import os
from io import BytesIO
from PIL import Image
import validators
import streamlit as st
import requests

# Application Streamlit pour la classification des animaux
HOST = os.environ.get("SERVING_HOST", "0.0.0.0")
PORT = os.environ.get("SERVING_PORT", "8080")
API_GATEWAY = os.environ.get("API_GATEWAY", "")

if API_GATEWAY != "":
    BASE_URL = API_GATEWAY
else:
    BASE_URL = f"{HOST}:{PORT}"

def valid_img_extension(url: str) -> bool:
    """
    Détermine si l'image a une extension d'image valide.
    """
    return url.split(".")[-1].lower() in [
        "jpg",
        "jpeg",
        "png",
    ]

def get_predictions(model_id: str, image_url: str):
    """
    Envoie l'URL de l'image au backend pour obtenir une prédiction.
    """
    return requests.post(f"{BASE_URL}/model/{model_id}/predict", json=dict(url=image_url))


def get_model_list():
    """
    Retourne la liste des modèles valides
    """
    # Code pour obtenir la liste des modèles valides.


st.title("Classification des Animaux")
st.write("Chargez une image depuis une URL pour la classification.")

with st.container():
    valid_models = get_model_list().json()
    st.selectbox("Sélectionnez le modèle à utiliser", valid_models, key="model_name")

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
            result = get_predictions(url, st.session_state.model_name)

        if result.status_code == 200:
            st.write(f"Je pense que c'est une image d'un(e) {result.json()['category']}.")
        else:
            st.error("Il y a eu une erreur dans la réception des prédictions")

st.markdown("")
