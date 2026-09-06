from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser

from pydantic import BaseModel, Field
from typing import Literal

from langchain_core.runnables import RunnableBranch, RunnableLambda


# Load environment variables
load_dotenv()


# Hugging Face LLM
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)


# Output parser for normal text
parser = StrOutputParser()


# Pydantic model for classification
class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="The sentiment of the feedback"
    )


# Pydantic output parser
parser2 = PydanticOutputParser(
    pydantic_object=Feedback
)


# Prompt for sentiment classification
prompt1 = PromptTemplate(
    template="""
Classify the sentiment of the following feedback as either positive or negative.

Feedback:
{feedback}

IMPORTANT:
- Return ONLY the JSON object.
- Do NOT provide any explanation.
- Do NOT provide reasoning.
- Do NOT write anything before or after the JSON.
- The sentiment must be exactly "positive" or "negative".

{format_instructions}
""",
    input_variables=["feedback"],
    partial_variables={
        "format_instructions": parser2.get_format_instructions()
    }
)


# Prompt for positive feedback
prompt2 = PromptTemplate(
    template="""
Write an appropriate response to the following positive feedback:

{feedback}
""",
    input_variables=["feedback"]
)


# Prompt for negative feedback
prompt3 = PromptTemplate(
    template="""
Write an appropriate response to the following negative feedback:

{feedback}
""",
    input_variables=["feedback"]
)


# Classification chain
classifier_chain = prompt1 | model | parser2


# Conditional branch
branch_chain = RunnableBranch(
    (
        lambda x: x.sentiment == "positive",
        prompt2 | model | parser
    ),
    (
        lambda x: x.sentiment == "negative",
        prompt3 | model | parser
    ),
    RunnableLambda(
        lambda x: "Something else went wrong"
    )
)


# Complete chain
chain = classifier_chain | branch_chain


# Invoke chain
result = chain.invoke(
    {
        "feedback": "The product is terrible and waste of money"
    }
)


# Print result
print(result)