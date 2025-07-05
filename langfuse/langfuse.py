import os
import google.generativeai as genai
from langfuse import observe
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

@observe
def generate_character_bio(topic):
  """Generates a character biography using Gemini."""
  model = genai.GenerativeModel('gemini-2.5-flash')

  response = model.generate_content(
      f"Create a short, compelling character biography for: {topic}"
  )

  return response.text

if __name__ == "__main__":
  character_topic = "a reclusive lighthouse keeper who finds a mysterious map"
  bio = generate_character_bio(character_topic)
  print(bio)
