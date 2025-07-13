import os
import io
import time
import logging
import requests
from PIL import Image
from abc import ABC, abstractmethod
from urllib.parse import urlparse
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, image_url: str = None) -> str:
        pass

class GeminiProvider(LLMProvider):
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel(model_name)

    def generate(self, prompt: str, image_url: str = None) -> str:
        content = [prompt]

        if image_url:
            try:
                response = requests.get(image_url, timeout=10)
                response.raise_for_status()
                img = Image.open(io.BytesIO(response.content))
                content.append(img)
            except Exception as e:
                logger.error(f"Error processing image: {e}")

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(
                    content,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0,
                        max_output_tokens=1000,
                        response_mime_type="application/json"
                    )
                )
                if hasattr(response, 'text') and response.text:
                    return response.text

            except Exception as e:
                logger.error(f"API error (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    return '{"reasoning": "Technical error", "FunctionCall": []}'
                time.sleep(1)

        return '{"reasoning": "Connection issues", "FunctionCall": []}'

# === Prompt Generator ===
def create_prompt(query: str, image_url: str = None) -> str:
    image_filename = os.path.basename(urlparse(image_url).path) if image_url else None

    return f"""You are a Customer Service Agent. Analyze the user query and respond with JSON.

OUTPUT JSON SCHEMA:
{{
    "reasoning": "Explanation of user intent and why specific tools are needed",
    "FunctionCall": [
        {{
            "name": "search_products",
            "args": {{
                "text": "combined search text with image descriptions",
                "image": {str(bool(image_url)).lower()},
                "image_url": {[image_filename] if image_filename else []},
                "filters": {{
                    "category": "string or null",
                    "price_range": {{
                        "min": 0,
                        "max": 0,
                        "operation": "eq"
                    }},
                    "attributes": {{
                        "color": "string or null",
                        "size": "string or null",
                        "brand": "string or null",
                        "material": "string or null"
                    }}
                }}
            }}
        }}
    ]
}}

Instructions:
- Extract keywords from text query
- If image provided: describe product style, color, material, category, brand
- Combine text + image description in the "text" field
- Set "image" to true/false based on whether image was provided
- Include image filename in "image_url" array if image exists

User Query: {query}
Respond to the user with a friendly, short message based on their query.
"""

def test_chat():
    llm = GeminiProvider()
    print("===  Interactive Chat with Gemini to test Images ===")

    while True:
        try:
            user_input = input("\n  User query (or type 'exit' to quit): ").strip()
            if user_input.lower() in {"exit", "quit"}:
                print(" Goodbye!")
                break

            image_url = input("  Optional image URL (press Enter to skip): ").strip()
            image_url = image_url if image_url else None

            prompt = create_prompt(user_input, image_url)
            print("\n Sending request to Gemini...")

            response = llm.generate(prompt, image_url)

            print("\n LLM Response:")
            print(response)
            return response.text



        except KeyboardInterrupt:
            print("\n Exiting chat.")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    test_chat()
