import os
import google.generativeai as genai
from langfuse import Langfuse, observe  # Make sure observe is imported
from dotenv import load_dotenv
import json

load_dotenv()

genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

langfuse = Langfuse()

prompt_name = "gemini-products-agent"
prompt = langfuse.get_prompt(prompt_name)


@observe(name="products-agent-trace")
def run_agent(user_query):
    """
    Runs the agent by sending the user query and system prompt to the Gemini model.
    """
    model = genai.GenerativeModel(
        model_name=prompt.config.get("model", "gemini-2.5-flash"),
        system_instruction=prompt.prompt
    )

    response = model.generate_content(user_query)

    return response.text


if __name__ == "__main__":
    test_query = "I need a white desk for my home office, maybe around 100 OMR."

    print(f"Testing with query: '{test_query}'\n")
    agent_output_str = run_agent(test_query)

    print("--- Agent Output ---")
    print(agent_output_str)

    try:
        parsed_output = json.loads(agent_output_str)
        print("\n--- JSON is Valid ---")
    except json.JSONDecodeError as e:
        print(f"\n--- Error: Output is not valid JSON! ---\n{e}")
