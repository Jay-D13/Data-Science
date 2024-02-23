#!/bin/bash
REGION="northamerica-northeast1"
PROJECT_ID="ift6758-hw5-405618"

# Voici le service que vous allez déployer.
SERVICE_NAME="backend-v1"

# Mettez ici votre URI d'image correspondant au service ci-dessus.
IMAGE_URI="northamerica-northeast1-docker.pkg.dev/ift6758-hw5-405618/ift6758-hw5/backend_v1:d4887535-e3aa-4e6e-ad96-15365c3d0f20"

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
    --set-env-vars="KEY=VALUE"

# REMARQUE : Dans un environnement de production, nous ne souhaiterons peut-être pas
# permettre à n'importe qui d'accéder à nos service(s). Pour les
# besoins de cet exercice, il est acceptable qu'il
# soit accessible publiquement.
