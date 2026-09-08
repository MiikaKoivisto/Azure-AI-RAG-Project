import os
from dotenv import load_dotenv
from openai import OpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

load_dotenv()

azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
model_deployment = os.getenv("MODEL_DEPLOYMENT")

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(),
    "https://ai.azure.com/.default"
)

client = OpenAI(
    base_url=azure_openai_endpoint,
    api_key=token_provider
)


def ask_ai(messages):
    response = client.responses.create(
        model=model_deployment,
        input=messages
    )

    return response.output_text

def create_embedding(text):
    response = client.embeddings.create(
        model=os.getenv("EMBEDDING_DEPLOYMENT"),
        input=text
    )

    return response.data[0].embedding