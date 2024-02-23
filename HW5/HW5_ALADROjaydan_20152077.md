# HW5 Réponses courtes


**Name:** Jaydan ALADRO

**Matricule \#:** 20152077

**Veuillez écrire vos réponses courtes ci-dessous et les inclure dans votre soumission de gradescope.**

**Le titre du rapport doit être HW5_{Votre_Nom}\_{Matricule_\#}**

## Question 2

Everytime we submit a build, 2 images are being built: the base image and the frontend/backend image. This does not need to be the case because the base image is always the same.

We could speed up the build process by pulling the latest base image before running the other steps. For this we would need to do an initial build of the base image and then push it to the registry. Then, we would need to pull the base image in the first step of every build process, saving us the build time of the base image everytime we submit a build. 

Therefore, I have created `cloudbuild_base_image.yaml` which builds the base image and pushes it to the registry. I have also modified `cloudbuild_{component}.yaml` to pull the base image before running the other steps (commented what was previously the build and push of the base-image).

In addition, we need to change `"BASE_IMAGE_URI=northamerica-northeast1-docker.pkg.dev/ift6758-hw5-405618/ift6758-hw5/base-image:${BUILD_ID}"` to `"BASE_IMAGE_URI=northamerica-northeast1-docker.pkg.dev/ift6758-hw5-405618/ift6758-hw5/base-image:latest"` in the `cloudbuild_{component}` files since we haven't built the base image with the build id.

## Question 4

We could modify our Dockerfile used for building the backend_v1 service to include steps that download and store the ResNet model as part of the image. This way, the model is downloaded during the image build process, not during the runtime of our service.
Otherwise, instead of downloading the model from PyTorch Hub each time, we can host the model on a fast-access storage service (like Google Cloud Storage or S3 for AWS cloud services) and download it from there. But the first approach is better because it does not require any additional services.

## Question 5

**Pros**:
- we can scale independently as we might need more resources for the backend for data processing and less for the frontend.
- updating or maintaining one part of the application doesn’t require taking down the entire application.
- if one service is running into issues, it won’t directly affect the other. We can just use the previous version of the service that was working well.
- we have 2 work pipelines that can be run in parallel, which speeds up the development process (2 different teams working at the same time).

**Cons**:
- one of the main issues/difficulties of a microservice architecture is the communication and data management between frontend and backend over the network as there can be latency.
- this leads to security concerns as the points of interaction between the frontend and backend can be vulnerable if not properly secured, especially in a public-facing API.
- end-to-end testing might become more challenging as it requires the integration of separately deployed services.

## Question 7

In the case of our `v1` application, this is not really an issue because the model is loaded only once when the container is started and it's the same model for all requests.

However, in `v2`, this is a problem because we have multiple model options and the model previous loaded model is overwritten on every model change request. This means that if 2 people are using the UI at the same time, the model will be changed for both of them, probably leading to some weird/unexpected behaviours by our service.

To fix this:
- In the frontend, we need to store the model choice in a session variable. This way, the model choice is stored for each user and we can send it to the backend with each request.
- In the backend, we should store the models in a dictionary and load the model based on the model choice sent by the frontend. However, the implications of loading a model on each request is that it will take more time to process the given request. Multiple models in the same container is not a good idea because it would take more time to load the container and it would consume more memory. The better way is to have a separate containers for each model.

## Question 8

In the backend, it seems the `v3` application is trying load the model based on the environment variable `MODEL_NAME`, which might indicate that it's loading a model per container. 

However, looking at the frontend code, the `API_GATEWAY` variable, if set, is used as the base URL for backend requests. Meaning the gateway can distribute incoming requests across multiple instances of our service. Useful if the load increases and we need to scale horizontally. It manages the rate of incoming requests to prevent the service from being overwhelmed, helping maintain stability and ensures fair resource usage among our users. Our users will be able to all use the service at the same time without any issues of battling which model to use.
