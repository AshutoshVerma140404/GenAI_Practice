from google import genai
from dotenv import load_dotenv


load_dotenv()

client = genai.Client()

result = client.models.embed_content(
        model="gemini-embedding-2",
        contents="What is the meaning of life?",
        config=genai.types.EmbedContentConfig(
        output_dimensionality=32
    )
)

print(result.embeddings)