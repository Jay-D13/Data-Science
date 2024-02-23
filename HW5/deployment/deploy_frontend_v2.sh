#!/bin/bash
REGION="northamerica-northeast1"
PROJECT_ID="ift6758-hw5-405618"

# Voici le service que vous allez déployer.
SERVICE_NAME="frontend-v2"

# Mettez ici votre URI d'image correspondant au service ci-dessus.
IMAGE_URI="northamerica-northeast1-docker.pkg.dev/ift6758-hw5-405618/ift6758-hw5/frontend_v2:f1a20f6e-f000-456e-959f-145001fcb0a4"
SERVING_URL="https://backend-v2-ez4gsr5e6a-nn.a.run.app"

# REMARQUE : Des valeurs par défaut sont définies pour la mémoire et le CPU
# mais vous devrez peut-être les changer.

gcloud config set project ${PROJECT_ID}

gcloud run deploy \
    "${SERVICE_NAME}" \
    --region=${REGION} \
    --image=${IMAGE_URI} \
    --min-instances=1 \
    --max-instances=1 \
    --memory=2Gi \
    --cpu=2 \
    --allow-unauthenticated \
    --set-env-vars="KEY=VALUE,SERVING_URL=${SERVING_URL}"

# REMARQUE : Dans un environnement de production, nous ne souhaiterons peut-être pas
# permettre à n'importe qui d'accéder à nos service(s). Pour les
# besoins de cet exercice, il est acceptable qu'il
# soit accessible publiquement.
