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
    """Generates embeddings using Google Cloud Vertex AI."""

    def __init__(self, project_id: str, location: str, model_name: str = "multimodalembedding@001"):
        """
        Initializes the Vertex AI client and loads the embedding model.

        Args:
            project_id: Your Google Cloud project ID.
            location: The Google Cloud region for your project (e.g., "us-central1").
            model_name: The name of the embedding model to use.
        """
        aiplatform.init(project=project_id, location=location)
        self.model = MultiModalEmbeddingModel.from_pretrained(model_name)
        print(f" Initialized Vertex AI EmbeddingGenerator with model '{model_name}'")

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for the given text using Vertex AI's multimodal model.

        Args:
            text: The input text to embed.

        Returns:
            A list of floats representing the embedding vector.
        """
        # Create an embedding instance for the text
        instance = EmbeddingInstance(text=text)

        # Send the request to the model
        response = self.model.embed(instances=[instance])

        # Return the first (and only) embedding from the response
        return response.predictions[0].embedding


class MilvusClient:
    def __init__(self):
        """Initialize Milvus client with credentials from .env file."""
        self.milvus_client = PyMilvusClient(
            uri=os.getenv("MILVUS_URI"),
            token=os.getenv("MILVUS_TOKENS")
        )
        print(" Connected to Milvus database")

    def search_products(self, embedding: List[float], filters: Dict[str, Any], limit: int = 5) -> List[Dict]:
        """
        Search the Milvus database using the embedding and filters.

        Args:
            embedding: Vector embedding of the search text
            filters: Dictionary of filters to apply
            limit: Maximum number of results to return

        Returns:
            List of matching products
        """
        expr = self._build_milvus_expression(filters)
        print("=====================EXP=============================")
        print(expr)
        print('======================================================')

        # Perform vector search with filters
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
        for hits in results:
            for hit in hits:
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
        """
        Build a Milvus expression based on filters.

        Args:
            filters: Dictionary of filters

        Returns:
            Milvus expression string or None if no filters
        """
        expressions = []

        # Add category filter
        if "category" in filters and filters["category"]:
            expressions.append(f'category == "{filters["category"]}"')

        # Add price range filter
        if "price_range" in filters and filters["price_range"]:
            price_range = filters["price_range"]
            if "operation" in price_range and price_range["operation"] is not None:
                opr = price_range["operation"]

                if opr == "range":
                    if "min" in price_range and price_range["min"] is not None:
                        expressions.append(f'price >= {price_range["min"]}')
                    if "max" in price_range and price_range["max"] is not None:
                        expressions.append(f'price <= {price_range["max"]}')

                if opr in ["loe", "lte"]:
                    if "max" in price_range and price_range["max"] is not None:
                        expressions.append(f'price <= {price_range["max"]}')

                if opr in ["hoe", "gte"]:
                    if "min" in price_range and price_range["min"] is not None:
                        expressions.append(f'price >= {price_range["min"]}')

                if opr == "eq":
                    if "min" in price_range and price_range["min"] is not None:
                        expressions.append(f'price == {price_range["min"]}')

        # Add attribute filters
        if "attributes" in filters and filters["attributes"]:
            for attr_name, attr_value in filters["attributes"].items():
                if attr_value is not None:
                    if isinstance(attr_value, str):
                        expressions.append(f'attributes["{attr_name}"] == "{attr_value}"')
                    elif isinstance(attr_value, (int, float)):
                        expressions.append(f'attributes["{attr_name}"] == {attr_value}')

        # Combine all expressions with AND
        if expressions:
            print("========================================")
            print(" && ".join(expressions))
            print("========================================")
            return " && ".join(expressions)
        return None


def create_prompt(query: str, image_url: str = None) -> str:
    """Create prompt for LLM to analyze user query and determine search parameters."""
    image_filename = os.path.basename(urlparse(image_url).path) if image_url else None

    return f"""You are a Customer Service Agent. Analyze the user's query and any provided images to understand their intent and plan the appropriate response.
AVAILABLE TOOLS:
- search_products: For finding products, recommendations, product details in Milvus database
- search_faqs: For questions about the business, shipping, returns, general info
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

User Query: {query}"""


def search_products(llm_response: str, milvus_client: MilvusClient, embedding_generator: EmbeddingGenerator) -> List[
    Dict]:
    """
    Search products based on LLM response containing text and image information.

    Args:
        llm_response: JSON string from LLM containing search parameters
        milvus_client: Milvus client instance with search_products method
        embedding_generator: EmbeddingGenerator instance

    Returns:
        List of product dictionaries
    """
    try:
        # Parse LLM response
        response_data = json.loads(llm_response)

        # Find search_products function call
        function_calls = response_data.get("FunctionCall", [])
        search_call = next(
            (call for call in function_calls if call.get("name") == "search_products"),
            None
        )

        if not search_call:
            return []

        args = search_call.get("args", {})
        text = args.get("text", "")
        image_urls = args.get("image_url", [])
        filters = args.get("filters", {})

        # Prepare embedding inputs
        embedding_inputs = []

        if text:
            embedding_inputs.append(text)

        if image_urls:
            # Combine image filenames as additional context
            image_context = f"image: {' '.join(image_urls)}"
            embedding_inputs.append(image_context)

        if not embedding_inputs:
            return []

        # Generate embeddings in parallel
        embeddings = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(embedding_inputs)) as executor:
            # Submit all embedding tasks
            futures = [
                executor.submit(embedding_generator.generate_embedding, input_text)
                for input_text in embedding_inputs
            ]

            # Collect results as they complete
            for future in concurrent.futures.as_completed(futures):
                embeddings.append(future.result())

        # Combine embeddings by averaging
        combined_embedding = [
            sum(dim_values) / len(embeddings)
            for dim_values in zip(*embeddings)
        ]

        # Search Milvus with combined embedding and filters
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
    """
    Format search results for end user using LLM.

    Args:
        products: List of product dictionaries from search
        llm_provider: LLM provider instance

    Returns:
        Formatted response string for end user
    """
    if not products:
        return "I couldn't find any products matching your criteria. Please try adjusting your search terms or filters."

    # Create a summary of products for the LLM
    products_summary = []
    for i, product in enumerate(products, 1):
        product_info = {
            "rank": i,
            "name": product.get("name", "Unknown"),
            "price": product.get("price", "N/A"),
            "category": product.get("category", "N/A"),
            "attributes": product.get("attributes", {}),
            "similarity_score": round(product.get("score", 0), 2)
        }
        products_summary.append(product_info)

    format_prompt = f"""Format these search results for a customer in a friendly, helpful way:

Products Found: {json.dumps(products_summary, indent=2)}

Please provide:
1. A brief introduction acknowledging their search
2. List the products with key details (name, price, relevant attributes)
3. Highlight the most relevant matches
4. Offer to help with more specific searches if needed

Keep the tone conversational and helpful."""

    try:
        formatted_response = llm_provider.generate(format_prompt)
        # If the response is JSON, extract the text content
        try:
            response_json = json.loads(formatted_response)
            return response_json.get("text", formatted_response)
        except:
            return formatted_response
    except Exception as e:
        logger.error(f"Error formatting response: {e}")
        # Fallback to simple formatting
        response = f"I found {len(products)} products for you:\n\n"
        for i, product in enumerate(products, 1):
            response += f"{i}. {product.get('name', 'Unknown')} - ${product.get('price', 'N/A')}\n"
        return response


class ProductSearchSystem:
    """Main system that coordinates all components."""

    def __init__(self):
        """Initialize all components with environment variables."""
        self.llm_provider = GeminiProvider()
        self.embedding_generator = EmbeddingGenerator(
            project_id=os.getenv("GOOGLE_PROJECT_ID"),
            location=os.getenv("GOOGLE_LOCATION")
        )
        self.milvus_client = MilvusClient()

    def search(self, user_query: str, image_url: str = None) -> str:
        """
        Complete search workflow.

        Args:
            user_query: User's search query
            image_url: Optional image URL

        Returns:
            Formatted response for end user
        """
        try:
            # Step 1: Generate search parameters using LLM
            prompt = create_prompt(user_query, image_url)
            llm_response = self.llm_provider.generate(prompt, image_url)

            print("LLM Response:")
            print(llm_response)
            print("\n" + "=" * 50 + "\n")

            # Step 2: Search products using embedding and filters
            products = search_products(llm_response, self.milvus_client, self.embedding_generator)

            print(f"Found {len(products)} products")
            print("\n" + "=" * 50 + "\n")

            # Step 3: Format response for end user
            formatted_response = format_response_for_user(products, self.llm_provider)

            return formatted_response

        except Exception as e:
            logger.error(f"Error in search workflow: {e}")
            return "I apologize, but I encountered an error while searching for products. Please try again later."


def interactive_chat():
    """
    Interactive chat interface for product search.
    """
    print("🛍️  Welcome to the Product Search Assistant!")
    print("Type 'exit' or 'quit' to end the conversation.")
    print("=" * 60)

    try:
        # Initialize the search system
        print("🔌 Connecting to services...")
        system = ProductSearchSystem()
        print("✅ All systems ready!")
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        print("Please check your .env file and API credentials.")
        return

    while True:
        try:
            # Get user input
            print("\n💬 What are you looking for?")
            user_query = input("You: ").strip()

            # Check for exit commands
            if user_query.lower() in ['exit', 'quit', 'bye', 'goodbye']:
                print("\n👋 Thank you for using our product search! Goodbye!")
                break

            if not user_query:
                print("⚠️  Please enter a search query.")
                continue

            # Get optional image URL
            print("\n🖼️  Do you have an image URL? (Press Enter to skip)")
            image_url = input("Image URL: ").strip()
            image_url = image_url if image_url else None

            # Show processing message
            print("\n🔍 Searching for products...")

            # Perform search
            result = system.search(user_query, image_url)

            # Display results
            print("\n🤖 Assistant:")
            print(result)
            print("\n" + "=" * 60)

        except KeyboardInterrupt:
            print("\n\n👋 Chat interrupted. Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error in interactive chat: {e}")
            print("⚠️  Sorry, I encountered an error. Please try again.")


if __name__ == "__main__":
    # Check if required environment variables are set
    required_vars = ["MILVUS_URI", "MILVUS_TOKENS", "GOOGLE_API_KEY", "OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease add them to your .env file and try again.")
    else:
        interactive_chat()