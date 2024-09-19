from google.cloud import secretmanager

import os
from google.cloud import secretmanager

def access_huggingface_token(project_id, secret_id, version_id="latest"):
    # Create the Secret Manager client
    client = secretmanager.SecretManagerServiceClient()
    # Build the resource name of the secret
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    # Access the secret version
    response = client.access_secret_version(request={"name": name})
    # Decode the secret payload
    secret_payload = response.payload.data.decode("UTF-8")
    print(f"Accessed secret {secret_id} with value {secret_payload}")
    # Set the Hugging Face token as an environment variable
    os.environ['HUGGINGFACE_TOKEN'] = secret_payload

# Usage
project_id = "build-with-ai-project"
secret_id = "huggingface_token"

# Call the function to access and set the token
access_huggingface_token(project_id, secret_id)

from huggingface_hub import login, whoami

# Log in to the Hugging Face API
login(token=os.environ['HUGGINGFACE_TOKEN'])

# Now you can use the Hugging Face API, e.g., with transformers or datasets
print(whoami())
