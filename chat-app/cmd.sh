
gcloud projects get-iam-policy build-with-ai-project \
--flatten="bindings[].members" \
--format='table(bindings.role)' \
--filter="bindings.members:chat-app-sa@build-with-ai-project.iam.gserviceaccount.com"

gcloud projects add-iam-policy-binding  build-with-ai-project \
  --member="serviceAccount:chat-app-sa@build-with-ai-project.iam.gserviceaccount.com" \
  --role=projects/build-with-ai-project/roles/my_chat_app_role

## remove role
#gcloud projects remove-iam-policy-binding build-with-ai-project \
#  --member="serviceAccount:chat-app-sa@build-with-ai-project.iam.gserviceaccount.com" \
#  --role="roles/aiplatform.admin"

