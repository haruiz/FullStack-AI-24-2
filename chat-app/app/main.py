import chainlit as cl
import vertexai
from vertexai.generative_models import GenerativeModel
from vertexai.generative_models import Image, Part
from google.auth import default
from google.oauth2 import service_account
from google.auth.exceptions import DefaultCredentialsError
import tenacity

# Fetch environment variables
def get_gcp_credentials(credentials_file=None):
    """
    Retrieves GCP credentials to initialize the Vertex AI client.
    """
    try:
        if credentials_file:
            print(f"Using credentials file: {credentials_file}")
            credentials = service_account.Credentials.from_service_account_file(credentials_file)
        else:
            print("Using default credentials")
            credentials, _ = default()
        return credentials

    except FileNotFoundError as e:
        raise FileNotFoundError(f"Credentials file '{credentials_file}' not found.") from e

    except DefaultCredentialsError as e:
        raise DefaultCredentialsError("Unable to obtain default credentials. Ensure that the environment "
                                      "is properly configured.") from e

@tenacity.retry(wait=tenacity.wait_fixed(2), stop=tenacity.stop_after_attempt(3), reraise=True)
async def send_chat_message(chat_session, message):
    """
    Sends a chat message to the chatbot.
    """
    try:
        response = await chat_session.send_message_async(message)
        return response
    except Exception as e:
        raise RuntimeError(f"Failed to send chat message: {e}") from e
    


PROJECT_ID = "build-with-ai-project"
LOCATION = "us-central1"
app_credentials = get_gcp_credentials("./chat-app-credentials.json")
vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=app_credentials)

# It defines the chat profile
system_message = (
# "You are a helpful chatbot. You are here to assist the user in any way you can. You can provide information, answer questions, and help the user with their needs."
# "You can also provide recommendations, suggestions, and advice. You are friendly, polite, and professional." 
# "You are knowledgeable, resourceful, and reliable. You are a good listener, and you are patient and understanding."
# "You are here to help the user, and you are dedicated to providing the best possible service."
# "You are a helpful chatbot, and you are always ready to assist the user."
# "This chat will be used for the employees of the company Bavarian Motor Works (BMW)."
# "Please make sure that in every answer you add the following information:"
# "If you have any questions, call the HR department at 1-800-555-1234."
# "Generate response in the same language as the user's input."
    "You are a psychologist chatbot. "
    "You are here to assist the user in any way you can. "
    "You fill respond to the user's messages with empathy and understanding. "
    "and make sure to ask questions to understand the user's feelings and emotions. "
)


@cl.on_chat_start
async def start():
    """
    Start the chat when the application launches.
    """
    gemini = GenerativeModel(model_name="gemini-1.5-flash", 
                             system_instruction=system_message
                             )
    chat_session = gemini.start_chat()
    cl.user_session.set("chat_session", chat_session)

    message = cl.Message(content="Welcome to my chatbot!")
    await message.send() # send the message to the user/ui

@cl.on_message
async def message(new_incoming_message: cl.Message):
    """
    Handle messages from the user.
    """
    # get the ref to the chat session
    chat_session = cl.user_session.get("chat_session")

    # create a multimodal prompt
    text_prompt = new_incoming_message.content
    message_elements = new_incoming_message.elements
    message_images = [element for element in message_elements if element.type == "image"]
    images_parts = list(map(lambda x: Part.from_image(Image.load_from_file(x.path)), message_images))
    multimodal_prompt = [text_prompt] + images_parts

    # send the multimodal prompt to the chat session, and get the response
    response = await send_chat_message(chat_session, multimodal_prompt)
    print(response)

    # send the response to the user interface
    message = cl.Message(content=response.text)
    await message.send()

    print(chat_session.history)
    