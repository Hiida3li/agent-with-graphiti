import os
import google.generativeai as genai
from langfuse import Langfuse
from dotenv import load_dotenv
import json

load_dotenv()

genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

langfuse = Langfuse()


prompt = langfuse.get_prompt("gemini-products-agent", label="latest")



def run_agent(user_query):
    """
  Runs the agent by fetching the Langfuse prompt, formatting it,
  and sending it to the Gemini model.
  """
    trace = langfuse.trace(name="products-agent-trace")

    model = genai.GenerativeModel(
        model_name='gemini-2.5-flash',
        system_instruction=prompt.prompt
    )

    # Get the generation from the trace, which automatically tracks it
    generation = trace.generation(
        name="agent-decision-generation",
        model=prompt.config["model"],  # Use the model from the prompt's config
        model_parameters=prompt.config,  # Pass the whole config
        input=user_query,  # The user's query is the input
        prompt=prompt,  # Link the prompt object for version tracking
    )

    # Call the Gemini API
    response = model.generate_content(user_query)

    # Update the generation with the output
    generation.end(output=response.text)

    return response.text


# --- Main execution block ---
if __name__ == "__main__":
    # 3. Define a test query
    test_query = "I need a white desk for my home office, maybe around 100 OMR."

    # 4. Run the agent and get the output
    print(f"Testing with query: '{test_query}'\n")
    agent_output_str = run_agent(test_query)

    print("--- Agent Output ---")
    print(agent_output_str)

    # You can try to parse the JSON to see if it's valid
    try:
        parsed_output = json.loads(agent_output_str)
        print("\n--- JSON is Valid ---")
    except json.JSONDecodeError as e:
        print(f"\n--- Error: Output is not valid JSON! --- \n{e}")