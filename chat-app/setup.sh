#https://dzlab.github.io/2023/08/20/gcp-run-hf/
PERFORM_RESET=0
PROJECT_ID="build-with-ai-project"
SERVICE_ACCOUNT_ID="vertex-ai-caller"
SERVICE_ACCOUNT_EMAIL=$SERVICE_ACCOUNT_ID@$PROJECT_ID.iam.gserviceaccount.com
KEY_FILE=./credentials.json

# huggingface token
HF_TOKEN="<WRITE_YOUR_HUGGINGFACE_TOKEN>"
HF_TOKEN_SECRET_NAME="huggingface_token"

# docker repository
REPO_NAME=build-with-ai-docker-repo
REPO_LOCATION=us-central1
REPO_DESCRIPTION="Build with AI docker repository"
DOCKER_IMAGE_NAME="chat-app:latest"


## reset gcloud settings
if [ $PERFORM_RESET -eq 1 ]; then
    echo "Resetting gcloud"
    rm -rf ~/.config/gcloud
    gcloud auth login
fi

# set default project
gcloud config set project $PROJECT_ID


# enabling services
gcloud services enable run.googleapis.com \
    cloudbuild.googleapis.com \
    aiplatform.googleapis.com \
    secretmanager.googleapis.com \
    artifactregistry.googleapis.com \
    cloudbuild.googleapis.com \
    run.googleapis.com


# create service account
gcloud iam service-accounts list | grep $SERVICE_ACCOUNT_ID
if [ $? -eq 0 ]; then
    echo "Service account $SERVICE_ACCOUNT_ID already exists"
else
    echo "Creating service account $SERVICE_ACCOUNT_ID"
    gcloud iam service-accounts create $SERVICE_ACCOUNT_ID \
        --description="$SERVICE_ACCOUNT_DESCRIPTION" \
        --display-name="$SERVICE_ACCOUNT_DISPLAY_NAME" \
        --quiet
fi


# grant roles to service account
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member serviceAccount:"$SERVICE_ACCOUNT_EMAIL" \
  --role="roles/aiplatform.user"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
    --role="roles/artifactregistry.reader"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
    --role="roles/iam.serviceAccountUser"

## cloud run admin
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
    --role="roles/run.admin"

## secret manager
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
  --role="roles/secretmanager.secretAccessor"


# create key file

# delete all keys for service account if any
gcloud iam service-accounts keys list --iam-account="$SERVICE_ACCOUNT_EMAIL"
if [ $? -eq 0 ]; then
    echo "Deleting all keys for service account $SERVICE_ACCOUNT_ID"
    # shellcheck disable=SC2086
    gcloud iam service-accounts keys list --iam-account=$SERVICE_ACCOUNT_EMAIL --format=json | jq -r '.[].name' | while read key; do
        echo "Deleting key $key"
        gcloud iam service-accounts keys delete $key --iam-account=$SERVICE_ACCOUNT_EMAIL --quiet
    done
fi

gcloud iam service-accounts keys create $KEY_FILE \
--iam-account=$SERVICE_ACCOUNT_ID@$PROJECT_ID.iam.gserviceaccount.com




gcloud artifacts repositories list --location=$REPO_LOCATION | grep $REPO_NAME
if [ $? -eq 0 ]; then
    echo "Repository $REPO_NAME already exists"
else
    echo "Creating repository $REPO_NAME"
    gcloud artifacts repositories create $REPO_NAME \
        --repository-format=docker \
        --location=$REPO_LOCATION \
        --description="$REPO_DESCRIPTION" \
        --immutable-tags \
        --async
fi



# Check if the secret exists
gcloud secrets describe $HF_TOKEN_SECRET_NAME --project=$PROJECT_ID
if [ $? -ne 0 ]; then
  echo "Secret $HF_TOKEN_SECRET_NAME does not exist. Creating it now..."
  echo -n $HF_TOKEN | gcloud secrets create $HF_TOKEN_SECRET_NAME --data-file=- --project=$PROJECT_ID
else
  echo "Secret $HF_TOKEN_SECRET_NAME already exists."
fi

# Check if the secret exists
gcloud secrets versions access latest --secret=$HF_TOKEN_SECRET_NAME --project=$PROJECT_ID

# upload the docker image to the repository
gcloud auth configure-docker $REPO_LOCATION-docker.pkg.dev
gcloud builds submit --tag $REPO_LOCATION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$DOCKER_IMAGE_NAME .

# deploy the app
gcloud run deploy chat-app \
    --image $REPO_LOCATION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$DOCKER_IMAGE_NAME  \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --service-account $SERVICE_ACCOUNT_EMAIL \
    --port 8000 \
    --memory 2Gi

