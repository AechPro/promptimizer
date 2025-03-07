from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


class Prompts(TypedDict):
    classifier: ChatPromptTemplate
    math_solver: ChatPromptTemplate


llm = ChatOpenAI(model="gpt-4o-mini")


async def solve_problem(prompts: Prompts, inputs: dict) -> dict:
    classifier_prompt = prompts["classifier"].invoke(inputs)
    result_message = await llm.ainvoke(classifier_prompt)
    language = result_message.content

    math_solver_prompt = prompts["math_solver"].invoke({"language": language, **inputs})
    response = await llm.ainvoke(math_solver_prompt)
    
    return {"answer": response.content}
