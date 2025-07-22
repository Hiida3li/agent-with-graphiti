class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str):  # Removed incorrect return type annotation
        pass



class GeminiProvider(LLMProvider):
        def __init__(self, model_name: str = "gemini-2.0-flash-001"):
            try:
                # Check for API key availability
                api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

                if not api_key:
                    raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY environment variable required")

                # Simple Gemini Developer API initialization
                self.client = genai.Client(api_key=api_key)  # No http_options - use default beta API for tools
                self.model_name = model_name

                logger.debug(f"GeminiProvider initialized successfully:")
                logger.debug(f"  - Model: {model_name}")
                logger.debug(f"  - Client type: {type(self.client)}")
                logger.debug(
                    f"  - Tools configured: {len(TOOLS)} tool groups with functions: {[func.name for tool in TOOLS for func in tool.function_declarations]}")

            except Exception as e:
                logger.error(f"Failed to initialize GeminiProvider: {e}")
                logger.error(f"Available environment variables:")
                logger.error(f"  - GOOGLE_API_KEY: {'***SET***' if os.getenv('GOOGLE_API_KEY') else 'NOT SET'}")
                logger.error(f"  - GEMINI_API_KEY: {'***SET***' if os.getenv('GEMINI_API_KEY') else 'NOT SET'}")
                raise

    def generate(self, prompt: str):  # Removed incorrect return type annotation
        logger.debug(f"Generating response for prompt length: {len(prompt)}")
        logger.debug(f"Prompt preview (first 200 chars): {prompt[:200]}...")
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.debug(f"Gemini API call attempt {attempt + 1}")
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        tools=TOOLS,
                        temperature=0,
                        max_output_tokens=10000,
                        top_p=0.95
                    )
                )

                if response and response.candidates:
                    logger.debug("Successfully received a valid response from Gemini.")
                    logger.debug(f"Response has {len(response.candidates)} candidates")
                    return response

                logger.warning(f"Empty response from Gemini on attempt {attempt + 1}.")

            except errors.APIError as e:  # Better error handling for new SDK
                logger.error(f"Gemini API error on attempt {attempt + 1}: {e.code} - {e.message}")
            except Exception as e:
                logger.error(f"Gemini API error on attempt {attempt + 1}: {e}")

            if attempt < max_retries - 1:
                time.sleep(1)

        logger.error("All retry attempts to reach Gemini failed.")
        return None