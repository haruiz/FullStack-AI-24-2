import asyncio
from functools import partial

import chainlit as cl
import vertexai
from vertexai.generative_models import GenerativeModel
from vertexai.generative_models import Image, Part, FunctionDeclaration, Tool
from google.auth import default
from google.oauth2 import service_account
from google.auth.exceptions import DefaultCredentialsError
from pathlib import Path
from vertexai.preview.vision_models import ImageGenerationModel
import tenacity
import torch
from diffusers import StableDiffusion3Pipeline


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
        response = await chat_session.send_message_async(message, tools=get_model_tools())
        return response
    except Exception as e:
        raise RuntimeError(f"Failed to send chat message: {e}") from e
    


PROJECT_ID = "build-with-ai-project"
LOCATION = "us-central1"
app_credentials = get_gcp_credentials("./chat-app-credentials.json")
vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=app_credentials)
google_imagen = ImageGenerationModel.from_pretrained("imagen-3.0-generate-001")
stable_diffusion = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    torch_dtype=torch.float16
)
image_gen_pipeline = stable_diffusion.to("mps")


system_message = (
"You are a helpful chatbot. You are here to assist the user in any way you can. You can provide information, answer questions, and help the user with their needs."
"You can also provide recommendations, suggestions, and advice. You are friendly, polite, and professional." 
"You are knowledgeable, resourceful, and reliable. You are a good listener, and you are patient and understanding."
"You are here to help the user, and you are dedicated to providing the best possible service."
"You are a helpful chatbot, and you are always ready to assist the user."
"Generate response in the same language as the user's input."
)

def generate_picture_using_imagen(prompt: str, num_images: int):
    """
    Generate a picture based on the prompt.
    """
    images_folder = Path("images")
    images_folder.mkdir(exist_ok=True)

    # generate the images
    generated_images = google_imagen.generate_images(
                              prompt=prompt,
                              number_of_images=int(num_images),
                              aspect_ratio="1:1",
                              safety_filter_level="block_some",
                              person_generation="allow_adult",
                              add_watermark=True)
    files = []
    for i in range(int(num_images)):
        image_path = images_folder / f"image_{i}.png"
        generated_images[i].save(str(image_path))
        files.append(image_path)
    return files


def generate_picture_using_stable_diff(prompt: str, num_images: int):
    """
    Generate a picture based on the prompt.
    """
    images_folder = Path("images")
    images_folder.mkdir(exist_ok=True)
    # generate the images
    result = image_gen_pipeline(
        prompt,
        num_images_per_prompt=int(num_images),
        negative_prompt="",
        num_inference_steps=28,
        guidance_scale=7.0,
    )
    files = []
    for i in range(int(num_images)):
        image_path = images_folder / f"image_{i}.png"
        result.images[i].save(str(image_path))
        files.append(image_path)
    return files



def get_model_tools():
    """
    Get the model tools.
    """
    generate_images_tool = FunctionDeclaration(
        name="generate_images",
        description="Generate images based on a prompt.",
        parameters={
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "num_images": {
                    "type": "integer",
                    "description": "The number of images to generate as int.",
                },
            },
            "required": ["prompt"],
        },
    )
    read_file = FunctionDeclaration(
        name="read_file",
        description="This function reads a file",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"}
            },
            "required": ["path"],
        },
    )
    # a tool is a collection of function declarations
    return [
         Tool(
             function_declarations=[generate_images_tool, read_file]
         )
    ]

@cl.step(type="tool")
async def generate_image_tool(model, **function_args):
    """
    Generate an image using the specified model.
    :param model:
    :param function_args:
    :return:
    """
    if model == "imagen":
        result = await cl.make_async(generate_picture_using_imagen)(**function_args)
    elif model == "stable_diff":
        result =  await cl.make_async(generate_picture_using_stable_diff)(**function_args)
    else:
        raise ValueError(f"Invalid model: {model}")
    return result


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
    function_calls = response.candidates[0].function_calls
    if function_calls:
        function_call = function_calls[0]
        function_args = {
            arg_name: arg_val for arg_name, arg_val in function_call.args.items()
        }
        print(function_call.name)
        if function_call.name == "generate_images":
            print("Generating images")
            res = await cl.AskActionMessage(
                content="Select a model to generate the image",
                actions=[
                    cl.Action(name="imagen", value="imagen", label="Google Imagen"),
                    cl.Action(name="stable_diff", value="stable_diff", label="Stable Diffusion"),
                ],
            ).send()
            model = res.get("value")
            image_files = await generate_image_tool(model, **function_args)
            images = []
            for file in image_files:
                image = cl.Image(path=str(file), display="inline")
                images.append(image)
            message = cl.Message(content="Here are the generated images:", elements=images)
            await message.send()
        elif function_call.name == "read_file":
            path = function_args["path"]
            with open(path, "r") as file:
                content = file.read()
            message = cl.Message(content=content)
            await message.send()
    else:
        message = cl.Message(content=response.text)
        await message.send()
