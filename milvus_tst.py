import os
import io
import time
import json
import logging
import requests
import concurrent.futures
from PIL import Image
from abc import ABC, abstractmethod
from urllib.parse import urlparse
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import google.generativeai as genai
from google.cloud import aiplatform
from vertexai.vision_models import Image, MultiModalEmbeddingModel
from pymilvus import MilvusClient as PyMilvusClient
from typing import List
import vertexai


load_dotenv()
PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID")
LOCATION = os.getenv("GOOGLE_LOCATION")
vertexai.init(project=PROJECT_ID, location=LOCATION)

logging.basicConfig(
    level=logging.INFO,
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
                        max_output_tokens=10000,
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


class EmbeddingGenerator:
    """Generates multimodal embeddings using Google Cloud Vertex AI."""

    def __init__(self, project_id: str, location: str, model_name: str = "multimodalembedding@001"):
        """
        Initializes the Vertex AI client and loads the multimodal embedding model.
        """
        self.model = MultiModalEmbeddingModel.from_pretrained(model_name)
        logger.info(f"Initialized Vertex AI EmbeddingGenerator with model '{model_name}'")

    def generate_embedding(self, text: str, image_url: Optional[str] = None) -> List[float]:
        """
        Generates a multimodal embedding for the given text and/or image.

        Args:
            text: The input text to embed.
            image_url: Optional URL of the input image to embed.

        Returns:
            A list of floats representing the embedding vector.
        """
        vertex_image = None
        # If an image URL is provided, download and load the image
        if image_url:
            try:
                response = requests.get(image_url, timeout=20)
                response.raise_for_status()
                # Load image data into a Vertex AI Image object
                vertex_image = VertexImage(response.content)
                logger.debug(f"Successfully loaded image from {image_url}")
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to load image from {image_url}: {e}. Proceeding with text only.")
                vertex_image = None # Ensure vertex_image is None if download fails

        # Ensure that at least text or an image is available
        if not text and not vertex_image:
            raise ValueError("At least text or an image must be provided to generate an embedding.")

        try:
            # Generate embeddings from text and/or image.
            embeddings = self.model.get_embeddings(
                contextual_text=text if text else None,
                image=vertex_image,
                dimension=1408  # The required dimension for this model
            )

            # If an image is present, the image_embedding is the multimodal vector.
            # Otherwise, fallback to the text_embedding.
            embedding_vector = embeddings.image_embedding if vertex_image else embeddings.text_embedding
            logger.info("Successfully generated embedding vector.")
            return embedding_vector

        except Exception as e:
            logger.error(f"Failed to generate embedding with Vertex AI: {e}")
            # Re-raise the exception to be handled by the caller
            raise


# --- Vector Database Client ---
class MilvusClient:
    def __init__(self):
        """Initialize Milvus client with credentials from .env file."""
        self.milvus_client = PyMilvusClient(
            uri=os.getenv("MILVUS_URI"),
            token=os.getenv("MILVUS_TOKEN") # Corrected from MILVUS_TOKENS
        )
        logger.info("Connected to Milvus database.")

    def search_products(self, embedding: List[float], filters: Dict[str, Any], limit: int = 5) -> List[Dict]:
        """
        Search the Milvus database using the embedding and filters.
        """
        expr = self._build_milvus_expression(filters)
        logger.info(f"Milvus filter expression: {expr}")

        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10},
        }

        results = self.milvus_client.search(
            collection_name="products",
            data=[embedding],
            anns_field="text_vector",
            search_params=search_params,
            filter=expr,
            limit=limit,
            output_fields=["name", "product_odoo_id", "variant_odoo_id", "category", "price", "attributes"]
        )
        products = []
        if results:
            for hit in results[0]:
                products.append({
                    "id": hit.id,
                    "score": hit.score,
                    "name": hit.entity.get("name"),
                    "product_odoo_id": hit.entity.get("product_odoo_id"),
                    "variant_odoo_id": hit.entity.get("variant_odoo_id"),
                    "price": hit.entity.get("price"),
                    "category": hit.entity.get("category"),
                    "attributes": hit.entity.get("attributes")
                })
        return products

    def _build_milvus_expression(self, filters: Dict[str, Any]) -> Optional[str]:
        expressions = []
        if "category" in filters and filters["category"]:
            expressions.append(f'category == "{filters["category"]}"')

        if "price_range" in filters and filters["price_range"]:
            price_range = filters["price_range"]
            op = price_range.get("operation")
            if op == "range":
                if "min" in price_range and price_range["min"] is not None: expressions.append(f'price >= {price_range["min"]}')
                if "max" in price_range and price_range["max"] is not None: expressions.append(f'price <= {price_range["max"]}')
            elif op in ["loe", "lte"] and "max" in price_range and price_range["max"] is not None:
                expressions.append(f'price <= {price_range["max"]}')
            elif op in ["hoe", "gte"] and "min" in price_range and price_range["min"] is not None:
                expressions.append(f'price >= {price_range["min"]}')
            elif op == "eq" and "min" in price_range and price_range["min"] is not None:
                expressions.append(f'price == {price_range["min"]}')

        if "attributes" in filters and filters["attributes"]:
            for attr_name, attr_value in filters["attributes"].items():
                if attr_value is not None:
                    if isinstance(attr_value, str):
                        expressions.append(f'json_extract(attributes, "$.{attr_name}") == "{attr_value}"')
                    else: # Assumes numeric for simplicity
                        expressions.append(f'json_extract(attributes, "$.{attr_name}") == {attr_value}')

        return " and ".join(expressions) if expressions else ""


# --- Core Workflow Functions ---
def create_prompt(query: str, image_url: str = None) -> str:
    """Create prompt for LLM to analyze user query and determine search parameters."""
    image_filename = os.path.basename(urlparse(image_url).path) if image_url else "none"

    return f"""You are a Customer Service Agent. Analyze the user's query and any provided images to understand their intent and plan the appropriate response.
AVAILABLE TOOLS:
- search_products: For finding products, recommendations, product details in Milvus database.
- search_faqs: For questions about the business, shipping, returns, general info.
OUTPUT JSON SCHEMA:
{{
    "reasoning": "Explanation of user intent and why specific tools are needed.",
    "FunctionCall": [
        {{
            "name": "search_products",
            "args": {{
                "text": "A combined search text including keywords from the user query and descriptions of the image content (style, color, material, etc.).",
                "image": {str(bool(image_url)).lower()},
                "image_url": "{image_filename}",
                "filters": {{
                    "category": "string or null",
                    "price_range": {{
                        "min": "number or null",
                        "max": "number or null",
                        "operation": "string (e.g., 'range', 'lte', 'gte', 'eq') or null"
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

INSTRUCTIONS:
- Extract keywords and intent from the user's text query.
- If an image is provided, describe its key features (product type, style, color, pattern, material, brand if visible).
- Combine the text query keywords and image description into a coherent search query in the "text" field.
- Set the "image" field to true or false.
- Extract any specific filters mentioned (category, price, color, size, etc.) and place them in the filters object.

User Query: {query}"""


def search_products(llm_response: str, milvus_client: MilvusClient, embedding_generator: EmbeddingGenerator, image_url: Optional[str] = None) -> List[Dict]:
    """
    Search products based on LLM response using a multimodal embedding.
    """
    try:
        response_data = json.loads(llm_response)
        search_call = next(
            (call for call in response_data.get("FunctionCall", []) if call.get("name") == "search_products"),
            None
        )
        if not search_call:
            return []

        args = search_call.get("args", {})
        text = args.get("text", "")
        filters = args.get("filters", {})

        if not text and not image_url:
            logger.warning("No text or image available for embedding.")
            return []

        combined_embedding = embedding_generator.generate_embedding(
            text=text,
            image_url=image_url
        )
        return milvus_client.search_products(
            embedding=combined_embedding,
            filters=filters,
            limit=5
        )
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.error(f"Error processing LLM response: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error in search_products: {e}")
        return []


def format_response_for_user(products: List[Dict], llm_provider: LLMProvider) -> str:
    """Format search results for end user using LLM."""
    if not products:
        return "I couldn't find any products matching your criteria. Please try adjusting your search terms or filters."

    products_summary = [
        {
            "rank": i + 1,
            "name": p.get("name", "N/A"),
            "price": p.get("price", "N/A"),
            "category": p.get("category", "N/A"),
            "attributes": p.get("attributes", {}),
            "similarity_score": round(p.get("score", 0), 2)
        } for i, p in enumerate(products)
    ]

    format_prompt = f"""You are a helpful and friendly shopping assistant.
A customer has searched for a product, and here are the results. Please format them in a conversational and appealing way.

SEARCH RESULTS (JSON):
{json.dumps(products_summary, indent=2)}

INSTRUCTIONS:
1.  Start with a friendly opening.
2.  Present the top products clearly. You can use a list. Mention the name and price.
3.  Briefly highlight why these items might be a good match, perhaps referencing the top result's similarity score.
4.  Keep it concise and easy to read.
5.  End with an offer to refine the search or answer more questions.
Do not output JSON. Just provide the friendly text response for the user."""

    try:
        formatted_response = llm_provider.generate(format_prompt)
        # The LLM is now instructed to return text directly, so no JSON parsing is needed.
        return formatted_response
    except Exception as e:
        logger.error(f"Error formatting response with LLM: {e}")
        # Fallback to simple formatting
        response = f"I found {len(products)} products for you:\n\n"
        for i, product in enumerate(products, 1):
            response += f"{i}. {product.get('name', 'Unknown')} - ${product.get('price', 'N/A'):.2f}\n"
        return response


# --- Main Application System ---
class ProductSearchSystem:
    """Main system that coordinates all components."""
    def __init__(self):
        self.llm_provider = GeminiProvider()
        self.embedding_generator = EmbeddingGenerator(
            project_id=PROJECT_ID,
            location=LOCATION
        )
        self.milvus_client = MilvusClient()

    def search(self, user_query: str, image_url: str = None) -> str:
        """Complete search workflow."""
        try:
            prompt = create_prompt(user_query, image_url)
            llm_response = self.llm_provider.generate(prompt, image_url)
            logger.info(f"LLM Response:\n{llm_response}")

            products = search_products(
                llm_response,
                self.milvus_client,
                self.embedding_generator,
                image_url=image_url
            )
            logger.info(f"Found {len(products)} products from vector search.")

            return format_response_for_user(products, self.llm_provider)

        except Exception as e:
            logger.error(f"Error in main search workflow: {e}")
            return "I apologize, but I encountered an error while searching. Please try again later."


def interactive_chat():
    """Interactive command-line interface for product search."""
    print("🛍️  Welcome to the Multimodal Product Search Assistant!")
    print("You can search with text and optionally provide an image URL.")
    print("Type 'exit' or 'quit' to end the conversation.")
    print("=" * 60)

    try:
        print("🔌 Initializing services...")
        system = ProductSearchSystem()
        print("✅ System ready!")
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        print("Please check your .env file and cloud credentials.")
        return

    while True:
        try:
            print("\n" + "-"*60)
            user_query = input("💬 You: ").strip()

            if user_query.lower() in ['exit', 'quit', 'bye', 'goodbye']:
                print("\n👋 Thank you for shopping with us! Goodbye!")
                break
            if not user_query:
                continue

            image_url = input("🖼️ Image URL (optional, press Enter to skip): ").strip()
            image_url = image_url if image_url else None

            print("\n🔍 Searching...")
            result = system.search(user_query, image_url)

            print("\n🤖 Assistant:")
            print(result)

        except KeyboardInterrupt:
            print("\n\n👋 Chat interrupted. Goodbye!")
            break
        except Exception as e:
            logger.error(f"An unexpected error occurred in the chat loop: {e}")
            print("⚠️  Sorry, a critical error occurred. Please restart.")
            break


if __name__ == "__main__":
    required_vars = [
        "GOOGLE_PROJECT_ID",
        "GOOGLE_LOCATION",
        "GOOGLE_API_KEY",
        "MILVUS_URI",
        "MILVUS_TOKEN"
    ]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        print(" Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease create a .env file and add the missing variables.")
    else:
        interactive_chat()
