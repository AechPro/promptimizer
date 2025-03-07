from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


class Prompts(TypedDict):
    classifier: ChatPromptTemplate
    math_solver: ChatPromptTemplate


llm = ChatOpenAI(model="gpt-4o-mini")


async def solve_problem(prompts: Prompts, inputs: dict) -> dict:
    # problem_id = str(hash(inputs.get('problem', '')))[:6]  # Short hash as ID
    
    # print(f"[{problem_id}] CLASSIFIER PROMPT START")
    # print(prompts["classifier"].invoke(inputs))
    classifier_prompt = prompts["classifier"].invoke(inputs)
    # print(f"[{problem_id}] CLASSIFIER PROMPT: {classifier_prompt}")
    
    result_message = await llm.ainvoke(classifier_prompt)
    language = result_message.content
    # print(f"[{problem_id}] LANGUAGE DETERMINED: {language}")
    
    math_solver_prompt = prompts["math_solver"].invoke({"language": language, **inputs})
    # print(f"[{problem_id}] MATH SOLVER PROMPT: {math_solver_prompt}")
    
    response = await llm.ainvoke(math_solver_prompt)
    # print(f"[{problem_id}] FINAL ANSWER: {response.content}")
    
    return {"answer": response.content}
