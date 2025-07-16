import os
import io
import time
import json
import logging
import requests
import concurrent.futures
import PIL.Image
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
from datetime import datetime

load_dotenv()
PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID")
LOCATION = os.getenv("GOOGLE_LOCATION")
vertexai.init(project=PROJECT_ID, location=LOCATION)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='log_file.log',
    filemode='w'
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
                img = PIL.Image.open(io.BytesIO(response.content))
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
                    return '{"reasoning": "Technical error", "response_type": "conversational", "FunctionCall": [], "direct_response": "I apologize, but I\'m experiencing technical difficulties. Please try again later."}'
                time.sleep(1)

        return '{"reasoning": "Connection issues", "response_type": "conversational", "FunctionCall": [], "direct_response": "I apologize, but I\'m having connection issues. Please try again later."}'


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
                vertex_image = Image(response.content)
                logger.debug(f"Successfully loaded image from {image_url}")
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to load image from {image_url}: {e}. Proceeding with text only.")
                vertex_image = None  # Ensure vertex_image is None if download fails

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


class MilvusClient:
    def __init__(self):
        """Initialize Milvus client with credentials from .env file."""
        self.milvus_client = PyMilvusClient(
            uri=os.getenv("MILVUS_URI"),
            token=os.getenv("MILVUS_TOKEN")  # Corrected from MILVUS_TOKENS
        )
        logger.info("Connected to Milvus database.")

    def search_products(self, embedding: List[float], filters: Dict[str, Any], text: str = "", image_url: str = None,
                        limit: int = 5):
        """
        Search the Milvus database using the embedding and filters
        """

        logger.debug(f"[STEP 4] Searching with text: '{text}' and image: {image_url}")

        expr = self._build_milvus_expression(filters)
        logger.info(f"Milvus filter expression: {expr}")

        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10},
        }

        # ADD THIS: Log search parameters before executing
        logger.info(f"[STEP 4d] Milvus search parameters:")
        logger.info(f"  Collection: products")
        logger.info(f"  Embedding dimension: {len(embedding)}")
        logger.info(f"  Vector field: text_vector")
        logger.info(f"  Filter expression: {expr}")
        logger.info(f"  Limit: {limit}")

        try:
            results = self.milvus_client.search(
                collection_name="products_test",
                data=[embedding],
                anns_field="text_vector",
                search_params=search_params,
                filter=expr,
                limit=limit,
                output_fields=["name", "product_odoo_id", "variant_odoo_id", "category", "price", "attributes"]
            )


            logger.info(f"[STEP 4e] Raw Milvus results:")
            logger.info(f"  Results type: {type(results)}")
            logger.info(f"  Results length: {len(results) if results else 0}")
            if results and len(results) > 0:
                logger.info(f"  First result set length: {len(results[0])}")

        except Exception as e:
            logger.error(f"Milvus search failed: {e}")
            logger.error(f"Search parameters that caused failure:")
            logger.error(f"  Embedding length: {len(embedding)}")
            logger.error(f"  Expression: {expr}")
            return []

        products = []
        if results:
            for hit in results[0]:
                product_data = {
                    "id": hit.id,
                    "score": hit.score,
                    "name": hit.entity.get("name"),
                    "product_odoo_id": hit.entity.get("product_odoo_id"),
                    "variant_odoo_id": hit.entity.get("variant_odoo_id"),
                    "price": hit.entity.get("price"),
                    "category": hit.entity.get("category"),
                    "attributes": hit.entity.get("attributes")
                }
                products.append(product_data)

        # ADD THIS: Log final processed products
        logger.info(f"[STEP 4f] Processed products:")
        logger.info(f"  Final products count: {len(products)}")

        return products

    def _build_milvus_expression(self, filters: Dict[str, Any]) -> Optional[str]:
        expressions = []
        if "category" in filters and filters["category"]:
            expressions.append(f'category == "{filters["category"]}"')

        if "price_range" in filters and filters["price_range"]:
            price_range = filters["price_range"]
            op = price_range.get("operation")
            if op == "range":
                if "min" in price_range and price_range["min"] is not None: expressions.append(
                    f'price >= {price_range["min"]}')
                if "max" in price_range and price_range["max"] is not None: expressions.append(
                    f'price <= {price_range["max"]}')
            elif op in ["loe", "lte"] and "max" in price_range and price_range["max"] is not None:
                expressions.append(f'price <= {price_range["max"]}')
            elif op in ["hoe", "gte"] and "min" in price_range and price_range["min"] is not None:
                expressions.append(f'price >= {price_range["min"]}')
            elif op == "eq" and "min" in price_range and price_range["min"] is not None:
                expressions.append(f'price == {price_range["min"]}')

        if "attributes" in filters and filters["attributes"]:
            # This is a simplified approach - you'll need to adapt based on your actual schema
            for attr_name, attr_value in filters["attributes"].items():
                if isinstance(attr_value, str):
                    expressions.append(f'attributes["{attr_name}"] == "{attr_value}"')
                elif isinstance(attr_value, (int, float)):
                    expressions.append(f'attributes["{attr_name}"] == {attr_value}')

        return " && ".join(expressions) if expressions else ""


def create_prompt(query: str, image_url: str = None) -> str:
    """Create prompt for LLM to analyze user query and determine search parameters."""
    image_filename = os.path.basename(urlparse(image_url).path) if image_url else "none"

    return f"""You are a Customer Service Agent. Analyze the user's query and any provided images to understand their intent and plan the appropriate response.

AVAILABLE TOOLS:
- search_products: For finding products, recommendations, product details in Milvus database.
- search_faqs: For questions about the business, shipping, returns, general info.

OUTPUT JSON SCHEMA:
{{
    "reasoning": "Explanation of user intent and why specific tools are needed or why no tools are needed.",
    "response_type": "tool_call" or "conversational", 
    "FunctionCall": [
        {{
            "name": "search_products",
            "args": {{
                "text": "A combined search text including keywords from the user query and descriptions of the image content (style, color, material, etc.).",
                "image": {str(bool(image_url)).lower()},
                "image_url": "{image_url}",
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
    ],
    "direct_response": "If no tools are needed, provide a direct conversational response here"
}}

INSTRUCTIONS:
- For greetings, general conversation, or non-product queries: Set response_type to "conversational", leave FunctionCall empty, and provide a direct response.
- For product searches: Set response_type to "tool_call" and populate FunctionCall with search_products.
- For business questions: Set response_type to "tool_call" and use search_faqs.
- Extract keywords and intent from the user's text query.
- If an image is provided, describe its key features (product type, style, color, pattern, material, brand if visible).
- Combine the text query keywords and image description into a coherent search query in the "text" field.
- Set the "image" field to true or false.
- Extract any specific filters mentioned (category, price, color, size, etc.) and place them in the filters object.

User Query: {query}"""


def search_products(llm_response: str, milvus_client: MilvusClient, embedding_generator: EmbeddingGenerator,
                    image_url: Optional[str] = None) -> List[Dict]:
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
            logger.warning("No search_products function call found in LLM response")
            return []

        args = search_call.get("args", {})
        text = args.get("text", "")
        filters = args.get("filters", {})

        logger.info(f"[STEP 4a] Search parameters extracted:")
        logger.info(f"  Text: '{text}'")
        logger.info(f"  Image URL: {image_url}")
        logger.info(f"  Filters: {json.dumps(filters, indent=2)}")

        if not text and not image_url:
            logger.warning("No text or image available for embedding.")
            return []

        combined_embedding = embedding_generator.generate_embedding(
            text=text,
            image_url=image_url
        )

        logger.info(f"[STEP 4b] Generated embedding:")
        logger.info(f"  Embedding length: {len(combined_embedding)}")
        logger.info(f"  Embedding type: {type(combined_embedding)}")
        logger.info(
            f"  First 5 values: {combined_embedding[:5] if len(combined_embedding) >= 5 else combined_embedding}")

        search_results = milvus_client.search_products(
            embedding=combined_embedding,
            filters=filters,
            text=text,
            image_url=image_url,
            limit=5
        )

        logger.info(f"[STEP 4c] Search completed:")
        logger.info(f"  Results count: {len(search_results)}")
        if search_results:
            logger.info(f"  Top result score: {search_results[0].get('score', 'N/A')}")
            logger.info(f"  Top result name: {search_results[0].get('name', 'N/A')}")

        return search_results

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.error(f"Error processing LLM response: {e}")
        logger.error(f"LLM Response that caused error: {llm_response}")
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
        """Complete search workflow with proper routing."""
        logger.debug(f"[STEP 1] User query received: '{user_query}', Image URL: {image_url}")

        # Initialize variables for tool response
        text_query = ""
        filters_applied = {}

        try:
            prompt = create_prompt(user_query, image_url)
            logger.debug(f"[STEP 2] Prompt created for LLM:\n{prompt}")
            llm_response = self.llm_provider.generate(prompt, image_url)
            logger.debug(f"[STEP 3] LLM response received:\n{llm_response}")
            logger.info(f"LLM Response:\n{llm_response}")

            # Parse LLM response
            try:
                parsed_response = json.loads(llm_response)
                response_type = parsed_response.get("response_type", "conversational")
                function_calls = parsed_response.get("FunctionCall", [])
                direct_response = parsed_response.get("direct_response", "")

                logger.info(f"[STEP 3a] Parsed response type: {response_type}")
                logger.info(f"[STEP 3b] Function calls count: {len(function_calls)}")

            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Failed to parse LLM response: {e}")
                return "I apologize, but I had trouble understanding that. Could you please rephrase your request?"

            # Route based on response type
            if response_type == "conversational" or not function_calls:
                logger.info("[STEP 4] Handling conversational response - no tools needed")
                if direct_response:
                    return direct_response
                else:
                    return "Hello! I'm here to help you find products or answer any questions. What can I assist you with today?"

            elif response_type == "tool_call":
                logger.info("[STEP 4] Processing tool calls")

                # Check for search_products function call
                search_call = next(
                    (call for call in function_calls if call.get("name") == "search_products"),
                    None
                )

                if search_call:
                    logger.info("[STEP 4a] Executing product search")

                    # Extract search parameters for logging
                    try:
                        args = search_call.get("args", {})
                        text_query = args.get("text", "")
                        filters_applied = args.get("filters", {})
                    except (KeyError, TypeError):
                        text_query = ""
                        filters_applied = {}

                    products = search_products(
                        llm_response,
                        self.milvus_client,
                        self.embedding_generator,
                        image_url=image_url
                    )

                    # Create tool response for logging
                    tool_response = {
                        "function_name": "search_products",
                        "function_call_id": "call_001",
                        "execution_status": "completed",
                        "timestamp": datetime.now().isoformat(),
                        "execution_time_ms": 2000,  # Approximate time
                        "results": {
                            "status": "success" if products else "no_results",
                            "message": f"Found {len(products)} products matching the search criteria" if products else "Milvus is down!",
                            "total_results": len(products),
                            "returned_results": len(products),
                            "products": products
                        },
                        "search_metadata": {
                            "text_query": text_query,
                            "image_search_enabled": bool(image_url),
                            "image_url": image_url if image_url else None,
                            "filters_applied": filters_applied,
                            "search_method": "multimodal_vector_search",
                            "similarity_threshold": 0.0,
                            "max_results": 5
                        },
                        "debug_info": {
                            "embedding_dimension": 1408,
                            "vector_search_executed": True,
                            "filter_expression": filters_applied,
                            "search_error": None if products else "No matching products found"
                        }
                    }

                    print("\n" + "=" * 60)
                    print("JSON TOOL OUTPUT (What gets sent to LLM):")
                    print("=" * 60)
                    print(json.dumps(tool_response, indent=2, default=str))
                    print("=" * 60)

                    # Log the tool response
                    logger.info(f"[STEP 5] Complete tool response JSON:")
                    logger.info(json.dumps(tool_response, indent=2, default=str))

                    return format_response_for_user(products, self.llm_provider)

                logger.warning("Tool call requested but no recognized function found")
                return "I'm not sure how to help with that. Could you please be more specific about what you're looking for?"

            else:
                logger.warning(f"Unknown response type: {response_type}")
                return "I'm not sure how to help with that. Could you please rephrase your request?"

        except Exception as e:
            logger.error(f"Error in main search workflow: {e}")
            return "I apologize, but I encountered an error while processing your request. Please try again later."


def interactive_chat():
    """Interactive command-line interface for product search."""
    try:
        print(" Initializing services...")
        system = ProductSearchSystem()
        print(" System ready!")
    except Exception as e:
        print(f" Failed to initialize system: {e}")
        print("Please check your .env file and cloud credentials.")
        return

    print("\n Welcome to the Product Search Assistant!")
    print(" I can help you find products or answer questions.")
    print(" Type 'exit', 'quit', 'bye', or 'goodbye' to end the conversation.\n")

    while True:
        try:
            print("-" * 60)
            user_query = input(" You: ").strip()

            if user_query.lower() in ['exit', 'quit', 'bye', 'goodbye']:
                print("\n Thank you for shopping with us! Goodbye!")
                break
            if not user_query:
                continue

            image_url = input("  Image URL (optional, press Enter to skip): ").strip()
            image_url = image_url if image_url else None

            print("\n🔍 Searching...")
            result = system.search(user_query, image_url)

            print("\n🤖 Assistant:")
            print(result)

        except KeyboardInterrupt:
            print("\n\n⚡ Chat interrupted. Goodbye!")
            break
        except Exception as e:
            logger.error(f"An unexpected error occurred in the chat loop: {e}")
            print(" Sorry, a critical error occurred. Please restart.")
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


