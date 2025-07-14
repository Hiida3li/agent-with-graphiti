# import os
# import io
# import time
# import logging
# import asyncio
# import json
# from typing import List, Dict, Any
#
# import aiohttp
# import google.generativeai as genai
# from PIL import Image
# from dotenv import load_dotenv
# import requests
#
# load_dotenv()
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)
#
#
#
# class GeminiProvider:
#     """
#     A simplified provider for Google's Gemini model that handles both
#     synchronous and asynchronous requests using a single, unified client.
#     """
#
#     def __init__(self, model_name: str = "gemini-1.5-flash"):
#
#         genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
#         self.model_name = model_name
#         self.client = genai.GenerativeModel(
#             model_name=self.model_name,
#             generation_config=genai.GenerationConfig(
#                 temperature=0,
#                 max_output_tokens=20000,
#                 response_mime_type="application/json"
#             )
#         )
#
#     def generate(self, prompt: str, image_url: str = None) -> str:
#         """Synchronous generation with retry logic."""
#         content = self._prepare_content(prompt, image_url, sync=True)
#         for attempt in range(3):
#             try:
#                 response = self.client.generate_content(contents=content)
#                 return response.text
#             except Exception as e:
#                 logger.error(f"[Sync] API error (attempt {attempt + 1}): {e}")
#                 if attempt == 2:
#                     return '{"reasoning": "Technical error after multiple retries", "FunctionCall": []}'
#                 time.sleep(1)
#         return '{"reasoning": "Connection issues", "FunctionCall": []}'
#
#     async def generate_async(self, prompt: str, image_url: str = None) -> str:
#         """Asynchronous generation with retry logic."""
#         content = await self._prepare_content_async(prompt, image_url)
#         for attempt in range(3):
#             try:
#                 # FIX: Async calls are made using the .aio property of the same client.
#                 response = await self.client.generate_content_async(contents=content)
#                 return response.text
#             except Exception as e:
#                 logger.error(f"[Async] API error (attempt {attempt + 1}): {e}")
#                 if attempt == 2:
#                     return '{"reasoning": "Technical error after multiple retries", "FunctionCall": []}'
#                 await asyncio.sleep(1)
#         return '{"reasoning": "Connection issues", "FunctionCall": []}'
#
#     def _prepare_content(self, prompt: str, image_url: str = None, sync: bool = True) -> List[Any]:
#         """Helper to prepare content for sync image fetching."""
#         if not image_url:
#             return [prompt]
#         try:
#             response = requests.get(image_url, timeout=10)
#             response.raise_for_status()
#             img = Image.open(io.BytesIO(response.content))
#             return [prompt, img]
#         except Exception as e:
#             logger.error(f"Error processing sync image: {e}")
#             return [prompt]
#
#     async def _prepare_content_async(self, prompt: str, image_url: str = None) -> List[Any]:
#         """Helper to prepare content for async image fetching."""
#         if not image_url:
#             return [prompt]
#         try:
#             async with aiohttp.ClientSession() as session:
#                 async with session.get(image_url, timeout=10) as response:
#                     response.raise_for_status()
#                     image_data = await response.read()
#                     img = Image.open(io.BytesIO(image_data))
#                     return [prompt, img]
#         except Exception as e:
#             logger.error(f"Error processing async image: {e}")
#             return [prompt]
#
# class ToolExecutor:
#     """Mock tool executor to simulate async tool calls."""
#
#     @staticmethod
#     async def search_products(text: str, **kwargs) -> Dict:
#         logger.info(f"Searching products for: '{text}'")
#         await asyncio.sleep(1)
#         return {"tool": "search_products", "results": f"Found products for: {text}"}
#
#     @staticmethod
#     async def search_faqs(text: str, **kwargs) -> Dict:
#         logger.info(f"Searching FAQs for: '{text}'")
#         await asyncio.sleep(0.5)
#         return {"tool": "search_faqs", "results": f"FAQ results for: {text}"}
#
#     @classmethod
#     async def execute(cls, tool_name: str, args: Dict) -> Any:
#         """Dynamically execute a tool by its name."""
#         if hasattr(cls, tool_name):
#             return await getattr(cls, tool_name)(**args)
#         logger.warning(f"Tool '{tool_name}' not found.")
#         return None
#
#
# class ParallelToolSystem:
#     """Orchestrates LLM planning and parallel tool execution."""
#     def __init__(self):
#         self.llm = GeminiProvider()
#         self.executor = ToolExecutor()
#         self.prompt_template = """
# You are a Customer Service Agent. Analyze the user's query and any provided images to plan the appropriate response by selecting tools.
#
# AVAILABLE TOOLS:
# - search_products: For finding products, recommendations, product details.
# - search_faqs: For questions about the business, shipping, returns, general info.
#
# OUTPUT a JSON object with your reasoning and a list of functions to call:
# {{
#     "reasoning": "Your explanation of the user's intent and why you chose these tools.",
#     "FunctionCall": [
#         {{
#             "name": "tool_name",
#             "args": {{"arg_name": "value"}}
#         }}
#     ]
# }}
#
# User Query: {query}
# """
#
#     def create_prompt(self, query: str) -> str:
#         return self.prompt_template.format(query=query)
#
#     async def process_query(self, query: str, image_url: str = None) -> Dict:
#         """Plans with the LLM, then executes all planned tools in parallel."""
#         logger.info(f"Processing query: '{query}'")
#         prompt = self.create_prompt(query)
#         plan_response = await self.llm.generate_async(prompt, image_url)
#
#         try:
#             plan = json.loads(plan_response)
#             function_calls = plan.get("FunctionCall", [])
#             logger.info(f"LLM Plan: {plan.get('reasoning')}")
#
#             if not function_calls:
#                 return {"reasoning": plan.get("reasoning"), "results": "No tools were called."}
#
#             tasks = [
#                 self.executor.execute(call["name"], call["args"])
#                 for call in function_calls
#             ]
#
#             results = await asyncio.gather(*[t for t in tasks if t])
#
#             return {"reasoning": plan.get("reasoning"), "results": results}
#
#         except (json.JSONDecodeError, KeyError) as e:
#             logger.error(f"Failed to parse or execute plan: {e}\nLLM Response: {plan_response}")
#             return {"error": "Failed to process the tool plan from LLM."}
#
#
# async def main():
#     """Main function to run demonstrations."""
#     system = ParallelToolSystem()
#     print("--- Testing Parallel Tool System ---")
#
#     start_time = time.time()
#
#     queries_to_process = [
#         system.process_query("I need a red dress for a wedding.", "https://example.com/dress.jpg"),
#         system.process_query("What's your return policy?"),
#         system.process_query("Show me blue shoes", "https://example.com/shoes.jpg")
#     ]
#
#     all_results = await asyncio.gather(*queries_to_process)
#
#     end_time = time.time()
#
#     print("\n--- Results ---")
#     for result in all_results:
#         print(json.dumps(result, indent=2))
#         print("-" * 20)
#
#     print(f"\nProcessed {len(queries_to_process)} queries in parallel in {end_time - start_time:.2f}s.")
#
#
# if __name__ == "__main__":
#
#     asyncio.run(main())



import os
import io
import time
import logging
import asyncio
import json
from typing import List, Dict, Any

import aiohttp
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv
import requests

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



class GeminiProvider:
    """Provider for Google's Gemini model."""
    def __init__(self, model_name: str = "gemini-1.5-flash"):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set.")

        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=genai.GenerationConfig(
                temperature=0,
                max_output_tokens=20000,
                response_mime_type="application/json"
            ),
            api_key=api_key
        )

    async def generate_async(self, prompt: str, image_url: str = None) -> str:
        """Asynchronous generation with retry logic."""
        content = await self._prepare_content_async(prompt, image_url)
        for attempt in range(3):
            try:
                response = await self.model.generate_content_async(contents=content)
                return response.text
            except Exception as e:
                logger.error(f"[Async] API error (attempt {attempt + 1}): {e}")
                if attempt == 2:
                    return '{"reasoning": "Technical error after multiple retries", "FunctionCall": []}'
                await asyncio.sleep(1)
        return '{"reasoning": "Connection issues", "FunctionCall": []}'

    async def _prepare_content_async(self, prompt: str, image_url: str = None) -> List[Any]:
        """Helper to prepare content for async image fetching."""
        if not image_url:
            return [prompt]
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url, timeout=10) as response:
                    response.raise_for_status()
                    image_data = await response.read()
                    # The model can often handle image bytes directly
                    return [prompt, Image.open(io.BytesIO(image_data))]
        except Exception as e:
            logger.error(f"Error processing async image: {e}")
            return [prompt]


class ToolExecutor:
    """Mock tool executor to simulate async tool calls."""
    @staticmethod
    async def search_products(text: str, **kwargs) -> Dict:
        logger.info(f"TOOL: Searching products for: '{text}' with filters: {kwargs.get('filters')}")
        await asyncio.sleep(1)
        return {"tool": "search_products", "results": f"Found 3 products for: '{text}'"}

    @staticmethod
    async def search_faqs(text: str, **kwargs) -> Dict:
        logger.info(f"TOOL: Searching FAQs for: '{text}'")
        await asyncio.sleep(0.5)
        return {"tool": "search_faqs", "results": f"The return policy is 30 days. More info at /returns."}

    @classmethod
    async def execute(cls, tool_name: str, args: Dict) -> Any:
        """Dynamically execute a tool by its name."""
        if hasattr(cls, tool_name):
            return await getattr(cls, tool_name)(**args)
        logger.warning(f"Tool '{tool_name}' not found.")
        return None


class ParallelToolSystem:
    """Orchestrates LLM planning and parallel tool execution."""
    def __init__(self):
        self.llm = GeminiProvider()
        self.executor = ToolExecutor()
        self.prompt_template = """
You are a helpful AI assistant. Your task is to plan function calls to answer a user's query by generating a JSON object.

AVAILABLE TOOLS

1. search_faqs(text: str)
    -Description: Use this to answer questions about business policies like shipping, returns, or general information.
    -Arguments: The `text` argument must be the user's original question.

2.  **search_products(text: str, filters: dict)**
    -   **Description**: Use this to find products, get recommendations, or search for product details.
    -   **Arguments**:
        -   The `text` argument should be a description of the product.
        -   The `filters` argument is a dictionary for filtering by category, price, attributes, etc.

## INSTRUCTIONS ##
1.  Analyze the user's query and any provided image description.
2.  If an image is provided, incorporate its description into the `text` argument for `search_products`.
3.  Respond with a JSON object containing your reasoning and a list of the function(s) to call.

**User Query**: {query}
"""

    def create_prompt(self, query: str) -> str:
        return self.prompt_template.format(query=query)

    async def process_query(self, query: str, image_url: str = None) -> Dict:
        """Plans with the LLM, then executes all planned tools in parallel."""
        logger.info(f"Processing query: '{query}'")
        prompt = self.create_prompt(query)
        plan_response = await self.llm.generate_async(prompt, image_url)

        try:
            plan = json.loads(plan_response)
            function_calls = plan.get("FunctionCall", [])
            logger.info(f"LLM Reasoning: {plan.get('reasoning')}")

            if not function_calls:
                return {"reasoning": plan.get("reasoning"), "results": "No tools were called."}

            tasks = [
                self.executor.execute(call["name"], call.get("args", {}))
                for call in function_calls
            ]
            results = await asyncio.gather(*[t for t in tasks if t])
            return {"reasoning": plan.get("reasoning"), "results": results}

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse or execute plan: {e}\nLLM Response: {plan_response}")
            return {"error": "Failed to process the tool plan from LLM."}


async def interactive_chat():
    """An async-compatible interactive chat loop."""
    system = ParallelToolSystem()
    print("🤖 Interactive Chat with Gemini and Parallel Tools")
    print("   Enter a query like 'What is your return policy?' or 'Find me a red dress.'")
    print("   (Type 'exit' or 'quit' to end the chat)")

    # Get the current asyncio event loop
    loop = asyncio.get_running_loop()

    while True:
        try:
            # Use run_in_executor to run the blocking input() function in a separate thread
            user_input = await loop.run_in_executor(
                None, lambda: input("\n> User: ").strip()
            )

            if user_input.lower() in {"exit", "quit"}:
                print("👋 Goodbye!")
                break

            image_url_input = await loop.run_in_executor(
                None, lambda: input("  Optional image URL (press Enter to skip): ").strip()
            )
            image_url = image_url_input if image_url_input else None

            print("\n⏳ Planning and executing tools...")
            start_time = time.time()
            result = await system.process_query(user_input, image_url)
            end_time = time.time()
            print(f"✅ Done in {end_time - start_time:.2f}s")

            print("\n--- Results ---")
            print(json.dumps(result, indent=2))
            print("-" * 15)


        except (KeyboardInterrupt, asyncio.CancelledError):
            print("\n👋 Exiting chat.")
            break
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(interactive_chat())
    except KeyboardInterrupt:
        print("\nProgram interrupted.")