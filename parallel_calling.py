import os
import io
import time
import logging
import requests
import asyncio
import aiohttp
from PIL import Image
from abc import ABC, abstractmethod
from urllib.parse import urlparse
from dotenv import load_dotenv
import google.genai as genai
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from typing import List, Dict, Any, Optional

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

    @abstractmethod
    async def generate_async(self, prompt: str, image_url: str = None) -> str:
        pass


class GeminiProvider(LLMProvider):
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.model_name = model_name
        self.client = genai.Client(api_key=self.api_key)
        self.async_client = genai.AsyncClient(api_key=self.api_key)

    def generate(self, prompt: str, image_url: str = None) -> str:
        """Synchronous generation using the new google-genai library"""
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
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=content,
                    config=genai.GenerateContentConfig(
                        temperature=0,
                        max_output_tokens=20000,
                        response_mime_type="application/json"
                    )
                )

                if response.text:
                    return response.text

            except Exception as e:
                logger.error(f"API error (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    return '{"reasoning": "Technical error", "FunctionCall": []}'
                time.sleep(1)

        return '{"reasoning": "Connection issues", "FunctionCall": []}'

    async def generate_async(self, prompt: str, image_url: str = None) -> str:
        """True async version using google-genai AsyncClient"""
        content = [prompt]

        if image_url:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(image_url, timeout=10) as response:
                        response.raise_for_status()
                        image_data = await response.read()
                        img = Image.open(io.BytesIO(image_data))
                        content.append(img)
            except Exception as e:
                logger.error(f"Error processing image: {e}")

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await self.async_client.models.generate_content(
                    model=self.model_name,
                    contents=content,
                    config=genai.GenerateContentConfig(
                        temperature=0,
                        max_output_tokens=10000,
                        response_mime_type="application/json"
                    )
                )

                if response.text:
                    return response.text

            except Exception as e:
                logger.error(f"API error (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    return '{"reasoning": "Technical error", "FunctionCall": []}'
                await asyncio.sleep(1)

        return '{"reasoning": "Connection issues", "FunctionCall": []}'


class ToolExecutor:
    """Mock tool executor for demonstration"""

    @staticmethod
    async def search_products(text: str, image: bool, image_url: List[str], filters: Dict) -> Dict:
        """Simulate product search"""
        await asyncio.sleep(1)  # Simulate API call
        return {
            "tool": "search_products",
            "results": f"Found products for: {text}",
            "filters_applied": filters
        }

    @staticmethod
    async def search_faqs(text: str) -> Dict:
        """Simulate FAQ search"""
        await asyncio.sleep(0.5)  # Simulate API call
        return {
            "tool": "search_faqs",
            "results": f"FAQ results for: {text}"
        }


class ParallelToolSystem:
    def __init__(self):
        self.llm = GeminiProvider()
        self.executor = ToolExecutor()

    def create_prompt(self, query: str, image_url: str = None) -> str:
        image_filename = os.path.basename(urlparse(image_url).path) if image_url else None

        return f"""You are a Customer Service Agent. Analyze the user's query and any provided images to understand their intent and plan the appropriate response.
TOOL SELECTION: Decide which functions to call (search_products, search_faqs, or both)
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

    async def process_query_with_parallel_tools(self, query: str, image_url: str = None) -> Dict:
        """Plan with Gemini, then execute planned tools in parallel"""
        logger.info(f"Processing query: {query}")

        prompt = self.create_prompt(query, image_url)
        plan_response = await self.llm.generate_async(prompt, image_url)

        try:
            plan = json.loads(plan_response)
            function_calls = plan.get("FunctionCall", [])

            if not function_calls:
                return {"reasoning": plan.get("reasoning"), "results": []}

            tasks = []
            for call in function_calls:
                tool_name = call["name"]
                args = call["args"]

                if tool_name == "search_products":
                    task = self.executor.search_products(
                        args["text"],
                        args["image"],
                        args["image_url"],
                        args["filters"]
                    )
                elif tool_name == "search_faqs":
                    task = self.executor.search_faqs(args["text"])

                tasks.append(task)

            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

            return {
                "reasoning": plan.get("reasoning"),
                "results": results
            }

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse plan: {e}")
            return {"error": "Failed to parse tool plan"}

    async def process_multiple_queries_parallel(self, queries: List[Dict]) -> List[Dict]:
        """Process multiple user queries in parallel"""
        tasks = []

        for query_data in queries:
            query = query_data["query"]
            image_url = query_data.get("image_url")
            task = self.process_query_with_parallel_tools(query, image_url)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results

    async def comprehensive_analysis_parallel(self, query: str, image_url: str = None) -> Dict:
        """Run multiple types of analysis in parallel"""

        # Create different prompts for different analysis types
        prompts = {
            "tool_planning": self.create_prompt(query, image_url),
            "sentiment_analysis": f"Analyze the sentiment and urgency of this customer query: {query}",
            "intent_classification": f"Classify the intent of this query into categories: {query}"
        }

        tasks = []
        for analysis_type, prompt in prompts.items():
            task = self.llm.generate_async(prompt, image_url if analysis_type == "tool_planning" else None)
            tasks.append((analysis_type, task))

        # Wait for all analyses to complete
        results = {}
        completed_tasks = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)

        for i, (analysis_type, _) in enumerate(tasks):
            results[analysis_type] = completed_tasks[i]

        return results

    async def process_multiple_images_parallel(self, query: str, image_urls: List[str]) -> Dict:
        """Process multiple images in parallel"""
        if not image_urls:
            return await self.process_query_with_parallel_tools(query)

        # Process each image in parallel
        tasks = []
        for image_url in image_urls:
            task = self.process_query_with_parallel_tools(query, image_url)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine results
        combined_results = {
            "query": query,
            "image_analyses": []
        }

        for i, result in enumerate(results):
            combined_results["image_analyses"].append({
                "image_url": image_urls[i],
                "analysis": result
            })

        return combined_results

    def process_with_thread_pool(self, queries: List[Dict], max_workers: int = 5) -> List[Dict]:
        """Use thread pool for CPU-intensive processing"""
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_query = {
                executor.submit(self._sync_process_query, query_data): query_data
                for query_data in queries
            }

            results = []
            for future in as_completed(future_to_query):
                query_data = future_to_query[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Query failed: {e}")
                    results.append({"error": str(e)})

            return results

    def _sync_process_query(self, query_data: Dict) -> Dict:
        """Synchronous version for thread pool"""
        query = query_data["query"]
        image_url = query_data.get("image_url")

        prompt = self.create_prompt(query, image_url)
        response = self.llm.generate(prompt, image_url)

        try:
            plan = json.loads(response)
            return {
                "query": query,
                "plan": plan
            }
        except json.JSONDecodeError:
            return {"query": query, "error": "Failed to parse response"}


async def test_parallel_system():
    system = ParallelToolSystem()

    print("=== Testing Parallel Tool System with True Async Support ===\n")

    # Test 1: Single query with parallel tool execution
    print("1. Single Query with Parallel Tools:")
    start_time = time.time()
    result1 = await system.process_query_with_parallel_tools(
        "I need a red dress for a wedding",
        "https://example.com/dress.jpg"
    )
    print(f"Result: {result1}")
    print(f"Time taken: {time.time() - start_time:.2f}s\n")

    # Test 2: Multiple queries in parallel
    print("2. Multiple Queries in Parallel:")
    queries = [
        {"query": "What's your return policy?"},
        {"query": "Show me blue shoes", "image_url": "https://example.com/shoes.jpg"},
        {"query": "I need help with my order"}
    ]

    start_time = time.time()
    results2 = await system.process_multiple_queries_parallel(queries)
    print(f"Results: {len(results2)} queries processed")
    print(f"Time taken: {time.time() - start_time:.2f}s\n")

    # Test 3: Comprehensive parallel analysis
    print("3. Comprehensive Parallel Analysis:")
    start_time = time.time()
    result3 = await system.comprehensive_analysis_parallel(
        "I'm very upset about my delayed order!"
    )
    print(f"Analysis types completed: {list(result3.keys())}")
    print(f"Time taken: {time.time() - start_time:.2f}s\n")

    # Test 4: Multiple images in parallel
    print("4. Multiple Images in Parallel:")
    image_urls = [
        "https://example.com/product1.jpg",
        "https://example.com/product2.jpg"
    ]
    start_time = time.time()
    result4 = await system.process_multiple_images_parallel(
        "Compare these products",
        image_urls
    )
    print(f"Processed {len(result4['image_analyses'])} images")
    print(f"Time taken: {time.time() - start_time:.2f}s\n")

    # Test 5: Demonstrate async efficiency with concurrent API calls
    print("5. Concurrent API Calls Test:")
    start_time = time.time()

    # Make 5 concurrent calls to Gemini
    tasks = []
    for i in range(5):
        prompt = f"Analyze this customer query #{i + 1}: 'Hello, I need help'"
        task = system.llm.generate_async(prompt)
        tasks.append(task)

    concurrent_results = await asyncio.gather(*tasks)
    concurrent_time = time.time() - start_time

    print(f"5 concurrent API calls completed in: {concurrent_time:.2f}s")
    print(f"Average per call: {concurrent_time / 5:.2f}s")
    print("This demonstrates true parallelism with the async client!\n")


if __name__ == "__main__":
    asyncio.run(test_parallel_system())

    system = ParallelToolSystem()
    queries = [
        {"query": "What sizes do you have?"},
        {"query": "Show me winter coats"},
        {"query": "How do I track my order?"}
    ]

    print("5. Thread Pool Processing:")
    thread_results = system.process_with_thread_pool(queries, max_workers=3)
    print(f"Thread pool processed: {len(thread_results)} queries")