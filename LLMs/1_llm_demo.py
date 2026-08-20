from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model = "gemini-3.7-flash")

result = llm.invoke("What is the capital of india")

print(result.content[0]["text"])
