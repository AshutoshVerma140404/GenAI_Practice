from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model = "gemini-3.7-flash", temperature = 1.5)

result = llm.invoke("What is the capital of india. and how are you doing today? please respond in a fun and engaging manner.")

print(result.content[0]["text"])
