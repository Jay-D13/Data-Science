import os
from flask import Flask, jsonify, request, abort
from PIL import Image
import requests

from io import BytesIO
from torchvision.models import resnet152, ResNet152_Weights

def load_model():
    # Initialiser le modèle avec des poids
    weights = ResNet152_Weights.DEFAULT
    model = resnet152(weights=weights)
    model.eval()
    return model, weights

def download_img(url: str) -> Image:
    ## Télécharger l'image dans la mémoire.
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img


# Initialiser l'application et chargez le modèle global et les poids
app = Flask(__name__)
model, weights = load_model()


@app.route("/hello", methods=["GET"])
def healthcheck():
    return jsonify("Hello!")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Gèrer les requêtes POST adressées à http://IP_ADDRESS:PORT/predict

    Renvoie les prédictions
    """
    # Get POST json data
    json = request.get_json()
    app.logger.info(json)

    # Extract url. Return 500 if not included in request.
    url = json.get('url', None)
    if url is None:
        abort(500, "L'url n'a pas été spécifiée dans la demande.")

    # Get Image from URL
    img = download_img(url)
    preprocess = weights.transforms()
    batch = preprocess(img).unsqueeze(0)

    # Get prediction and return
    prediction = model(batch).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    category = weights.meta['categories'][class_id]
    app.logger.info({"category":category})
    return jsonify(category=category)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=os.environ.get("SERVING_PORT", 8080))