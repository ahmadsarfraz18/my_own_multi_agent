from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
from dotenv import load_dotenv
import os

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")

external_client = AsyncOpenAI(
    api_key= gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model= OpenAIChatCompletionsModel(
    model= "gemini-2.5-flash",
    openai_client= external_client
)

config= RunConfig(
    model= model,
    model_provider= external_client,
    tracing_disabled= True

)

urdu_teacher_agent = Agent(
    name="Urdu Teacher Agent",
    instructions="You are an expert in Urdu language and culture. Provide responses in Urdu.",
    model= model
)

english_teacher_agent= Agent(
    name= "English Teacher Agent",
    instructions="You are an expert in English language and culture. Provide responses in English.",
    model= model
)

maths_teacher_agent = Agent(
    name="Maths Teacher Agent",
    instructions="You are an expert in mathematics. Provide responses according to the mathematical terms.",
    model=model
)

triage_agent = Agent(
    name= "Triage Agent",
    instructions="You are an expert in triaging requests to the appropriate subject-specific agents.",
    model=model,
    handoffs= [urdu_teacher_agent, english_teacher_agent, maths_teacher_agent]
)
prompt = input("Enter your query: ")
result = Runner.run_sync(
    triage_agent,
    # "Correct this sentence: She don’t like playing football.",
    prompt,
    run_config = config
)

print("last agent:", result.last_agent)  
print(result.final_output)
