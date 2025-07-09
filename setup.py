import os
import json
import logging
import google.generativeai as genai
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime
from pathlib import Path
from PIL import Image
import requests
import io
from urllib.parse import urlparse
from dotenv import load_dotenv
import concurrent.futures

load_dotenv()

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler('../logs/agent_trace.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)



class ToolName(Enum):
    SEARCH_PRODUCTS = "search_products"
    SEARCH_FAQS = "search_faqs"
    GET_PRODUCT_DETAILS = "get_product_details"
    PLACE_ORDER = "place_order"
    GET_ORDER_HISTORY = "get_order_history"


class PriceRange(BaseModel):
    min: Optional[float] = None
    max: Optional[float] = None
    operation: str = "between"  # "eq", "lt", "gt", "between"


class ProductAttributes(BaseModel):
    color: Optional[str] = None
    size: Optional[str] = None
    brand: Optional[str] = None
    material: Optional[str] = None


class ProductFilters(BaseModel):
    category: Optional[str] = None
    price_range: Optional[PriceRange] = None
    attributes: Optional[ProductAttributes] = None


class FunctionCallArgs(BaseModel):
    pass


class SearchProductsArgs(FunctionCallArgs):
    text: str
    image: bool = False
    image_url: List[str] = []
    filters: Optional[ProductFilters] = None


class SearchFAQsArgs(FunctionCallArgs):
    text: str


class FunctionCall(BaseModel):
    name: str
    args: Dict[str, Any]


class CompositionResponse(BaseModel):
    reasoning: str
    FunctionCall: List[FunctionCall]


class ImageProcessor:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')

    def process_image_url(self, image_url: str) -> Dict[str, Any]:
        """Process image from URL and return structured description"""
        try:
            # Download image
            response = requests.get(image_url)
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content))

            # Get filename
            image_filename = os.path.basename(urlparse(image_url).path)

            # Generate description
            prompt = f"""
You are an expert Customer Service Agent. Your task is to generate a concise, accurate, and dense description of a product from an image. This description will be used to create embeddings for a semantic search database, so it must contain key factual attributes.

Analyze the image and generate a JSON object.
The JSON object must conform to the following schema:
{{
    "text": "A short, factual description of the product. Focus on product type, style, colors, materials, and unique features like straps or logos.",
    "image_urls": ["A list containing the filename of the uploaded image."],
    "image": "A boolean set to true, indicating an image was processed."
}}

DO NOT make up information about products you don't know about.
Always respond based on the information you have.
The filename for the uploaded image is: {image_filename}
"""

            generation_config = genai.types.GenerationConfig(
                response_mime_type="application/json"
            )

            response = self.model.generate_content([prompt, img], generation_config=generation_config)
            return json.loads(response.text)

        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return {
                "text": "Unable to process image",
                "image_urls": [image_filename if 'image_filename' in locals() else "unknown.jpg"],
                "image": True
            }


# ================== COMPOSITIONAL AGENT ==================

class CompositionalAgent:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        self.image_processor = ImageProcessor(api_key)

        # Tool implementations
        self.tools = {
            "search_products": self._search_products,
            "search_faqs": self._search_faqs,
            "get_product_details": self._get_product_details,
            "place_order": self._place_order,
            "get_order_history": self._get_order_history
        }

    def analyze_query(self, user_query: str, image_urls: List[str] = None) -> CompositionResponse:
        """Analyze user query and plan function calls with compositional reasoning"""

        # Process images if provided
        image_descriptions = []
        image_filenames = []

        if image_urls:
            for url in image_urls:
                try:
                    image_data = self.image_processor.process_image_url(url)
                    image_descriptions.append(image_data["text"])
                    image_filenames.extend(image_data["image_urls"])
                except Exception as e:
                    logger.error(f"Failed to process image {url}: {e}")
                    image_filenames.append(os.path.basename(urlparse(url).path))

        # Create the analysis prompt
        prompt = f"""You are an expert Customer Service Agent. Analyze the user's query and any provided images to understand their intent and plan the appropriate response.

1. REASONING: Understand what the user wants and determine what tools are needed
2. TOOL SELECTION: Decide which functions to call (search_products, search_faqs, or both)
3. PARAMETER EXTRACTION: Extract search parameters and filters from the query
4. IMAGE ANALYSIS: Generate descriptions of any images provided

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
                "image": true/false,
                "image_url": {image_filenames if image_filenames else []},
                "filters": {{
                    "category": "string or null",
                    "price_range": {{
                        "min": number,
                        "max": number,
                        "operation": "eq" | "lt" | "gt" | "between"
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

REASONING GUIDELINES:
- If user asks about policies, shipping, returns → use search_faqs
- If user wants to find products, recommendations → use search_products  
- If user needs both product info AND policy info → use both tools
- Explain your reasoning clearly

TEXT OPTIMIZATION RULES:
- Combine user query keywords with image descriptions
- Focus on searchable product attributes: colors, materials, styles, functions
- Remove conversational words, keep only search-relevant terms
- For images: describe style, color, material, shape, function, category

FILTER EXTRACTION RULES:
- category: Extract product category from query/image (e.g., "Desks / Office Desks", "Clothing / Dresses")
- price_range: Extract budget mentions (e.g., "under $100" → max: 100, operation: "lt")
- attributes: Extract specific product features (color, size, brand, material)

IMAGE ANALYSIS REQUIREMENTS:
- Generate dense, factual descriptions focusing on searchable attributes
- Include: style, color, material, shape, size indicators, function, category
- Example: "minimalist white desk with rectangular top and thin metal legs"
- DO NOT make up information about products you don't recognize

PRICE OPERATIONS:
- "eq": exact price match
- "lt": less than
- "gt": greater than
- "between": range between min and max

User Query: {user_query}
Image Descriptions: {'; '.join(image_descriptions) if image_descriptions else 'None'}
Image Filenames: {image_filenames}
"""

        try:
            generation_config = genai.types.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.1
            )

            response = self.model.generate_content(prompt, generation_config=generation_config)

            # Parse response
            response_data = json.loads(response.text)
            return CompositionResponse(**response_data)

        except Exception as e:
            logger.error(f"Error in query analysis: {e}")
            # Fallback response
            return CompositionResponse(
                reasoning="Error in analysis, defaulting to product search",
                FunctionCall=[
                    FunctionCall(
                        name="search_products",
                        args={"text": user_query, "image": bool(image_urls), "image_url": image_filenames}
                    )
                ]
            )

    def execute_composition(self, user_query: str, image_urls: List[str] = None) -> Dict[str, Any]:
        """Execute compositional reasoning workflow"""

        # Step 1: Analyze and plan
        logger.info("Step 1: Analyzing query and planning function calls")
        composition = self.analyze_query(user_query, image_urls)
        logger.info(f"Reasoning: {composition.reasoning}")
        logger.info(f"Planned {len(composition.FunctionCall)} function calls")

        # Step 2: Execute planned function calls in parallel
        results = {}

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_call = {}

            for i, func_call in enumerate(composition.FunctionCall):
                logger.info(f"Step 2.{i + 1}: Executing {func_call.name}")

                if func_call.name in self.tools:
                    future = executor.submit(self.tools[func_call.name], **func_call.args)
                    future_to_call[future] = (func_call.name, i)
                else:
                    logger.error(f"Unknown tool: {func_call.name}")
                    results[f"{func_call.name}_{i}"] = f"Error: Unknown tool {func_call.name}"

            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_call):
                func_name, index = future_to_call[future]
                try:
                    result = future.result()
                    results[f"{func_name}_{index}"] = result
                    logger.info(f"Result: {str(result)[:100]}...")
                except Exception as e:
                    logger.error(f"Error executing {func_name}: {e}")
                    results[f"{func_name}_{index}"] = f"Error: {str(e)}"

        # Step 3: Generate final response
        logger.info("Step 3: Generating final response")
        final_response = self._generate_final_response(user_query, composition, results)

        return {
            "reasoning": composition.reasoning,
            "function_calls": [fc.dict() for fc in composition.FunctionCall],
            "results": results,
            "final_response": final_response
        }

    def _generate_final_response(self, user_query: str, composition: CompositionResponse,
                                 results: Dict[str, Any]) -> str:
        """Generate a natural language response based on the results"""

        prompt = f"""
Based on the user's query and the results from function calls, generate a helpful, natural response.

User Query: {user_query}
Reasoning: {composition.reasoning}
Function Results: {json.dumps(results, indent=2)}

Generate a helpful response that:
1. Directly answers the user's question
2. Incorporates relevant information from all function calls
3. Is conversational and friendly
4. Provides actionable next steps if appropriate

Response:
"""

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error generating final response: {e}")
            return "I found some information for you, but had trouble formatting the response. Please try again."

    # ================== TOOL IMPLEMENTATIONS ==================

    def _search_products(self, text: str, image: bool = False, image_url: List[str] = None,
                         filters: Dict = None) -> str:
        """Search for products in the database"""
        logger.info(f"Searching products: '{text}', image: {image}, filters: {filters}")

        # Mock implementation - replace with your actual product search
        mock_products = [
            "Modern White Standing Desk - $179.99",
            "Minimalist White Computer Desk - $149.99",
            "Adjustable White Office Desk - $199.99"
        ]

        # Apply filters if provided
        if filters:
            if filters.get('price_range'):
                price_filter = filters['price_range']
                logger.info(f"Applying price filter: {price_filter}")
            if filters.get('attributes', {}).get('color'):
                color = filters['attributes']['color']
                logger.info(f"Filtering by color: {color}")

        return f"Found {len(mock_products)} products: {'; '.join(mock_products)}"

    def _search_faqs(self, text: str) -> str:
        """Search FAQ database"""
        logger.info(f"Searching FAQs: '{text}'")

        # Mock implementation - replace with your actual FAQ search
        if "shipping" in text.lower():
            return "Shipping Policy: We offer free standard shipping on orders over $50. Express shipping available for $9.99. Orders typically ship within 1-2 business days."
        elif "return" in text.lower():
            return "Return Policy: 30-day return window for unused items in original packaging. Free returns on defective items."
        else:
            return f"FAQ information related to: {text}"

    def _get_product_details(self, product_id: str) -> str:
        """Get detailed product information"""
        return f"Product details for {product_id}: [Mock product details]"

    def _place_order(self, product_id: str, quantity: int = 1) -> str:
        """Place an order"""
        return f"Order placed for {quantity}x {product_id}"

    def _get_order_history(self, user_id: str) -> str:
        """Get user's order history"""
        return f"Order history for {user_id}: [Mock order history]"


# ================== MAIN EXECUTION ==================

def main():
    """Test the compositional agent"""

    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")

    agent = CompositionalAgent(GOOGLE_API_KEY)

    # Test cases
    test_cases = [
        {
            "query": "Show me this type of product and what about the return policy",
            "image_urls": ["https://demo2.wpthemego.com/themes/sw_himarket/wp-content/uploads/2016/04/20.jpg"]
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'=' * 50}")
        print(f"TEST CASE {i}")
        print(f"{'=' * 50}")
        print(f"Query: {test_case['query']}")
        if test_case['image_urls']:
            print(f"Images: {test_case['image_urls']}")

        try:
            result = agent.execute_composition(
                test_case['query'],
                test_case['image_urls']
            )

            print(f"\nReasoning: {result['reasoning']}")
            print(f"\nFunction Calls: {json.dumps(result['function_calls'], indent=2)}")
            print(f"\nFinal Response: {result['final_response']}")

        except Exception as e:
            logger.error(f"Error in test case {i}: {e}")
            print(f"Error: {e}")


if __name__ == "__main__":
    main()