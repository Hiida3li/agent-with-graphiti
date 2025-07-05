import os
import google.generativeai as genai
from langfuse import observe
from dotenv import load_dotenv

# This correctly loads all keys from your .env file
load_dotenv()

# This correctly configures the Gemini client
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

# The Langfuse keys are now loaded automatically.
# We have REMOVED the lines that were overwriting them.

@observe
def generate_character_bio(topic):
  """Generates a character biography using Gemini."""
  # Corrected the model name
  model = genai.GenerativeModel('gemini-1.5-flash')

  response = model.generate_content(
      f"Create a short, compelling character biography for: {topic}"
  )

  return response.text

# Run the function
if __name__ == "__main__":
  character_topic = "a reclusive lighthouse keeper who finds a mysterious map"
  bio = generate_character_bio(character_topic)
  print(bio)