from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-3.6-flash")

class Review(TypedDict):
    summary: str
    sentiment: str

structured_model = model.with_structured_output(Review)
result = structured_model.invoke("The Sony WH-1000XM5 remains the best overall wireless headphone on the market for most people. While the lack of a fully folding hinge makes it slightly less compact to travel with, its supreme comfort, sound isolation, and battery performance more than justify the price.")

print(result)