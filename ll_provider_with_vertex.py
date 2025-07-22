class GeminiProvider(LLMProvider):
    def __init__(self, model_name: str = "gemini-2.5-flash", use_vertex_ai: bool = False):
        try:

            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

            if use_vertex_ai:

                project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
                location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

                if not project_id:
                    raise ValueError("GOOGLE_CLOUD_PROJECT environment variable required for Vertex AI")

                self.client = genai.Client(
                    vertexai=True,
                    project=project_id,
                    location=location,
                    http_options=types.HttpOptions(api_version='v1')
                )
                logger.debug(f"Vertex AI client initialized - Project: {project_id}, Location: {location}")

            else:

                if not api_key:
                    # Try environment variable auto-detection
                    os.environ.setdefault("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
                    self.client = genai.Client()  # Auto-detect from environment
                    logger.debug("Client initialized using environment variable auto-detection")
                else:

                    self.client = genai.Client(
                        api_key=api_key,
                        http_options=types.HttpOptions(api_version='v1')  # Use stable API
                    )
                    logger.debug("Client initialized with explicit API key")

            self.model_name = model_name
            self.use_vertex_ai = use_vertex_ai

            # Test client initialization
            logger.debug(f"GeminiProvider initialized successfully:")
            logger.debug(f"  - Model: {model_name}")
            logger.debug(f"  - Vertex AI: {use_vertex_ai}")
            logger.debug(f"  - Client type: {type(self.client)}")
            # Fixed tools logging to handle new SDK format
            logger.debug(
                f"  - Tools configured: {len(TOOLS)} tool groups with functions: {[func.name for tool in TOOLS for func in tool.function_declarations]}")

        except Exception as e:
            logger.error(f"Failed to initialize GeminiProvider: {e}")
            logger.error(f"Available environment variables:")
            logger.error(f"  - GOOGLE_API_KEY: {'***SET***' if os.getenv('GOOGLE_API_KEY') else 'NOT SET'}")
            logger.error(f"  - GEMINI_API_KEY: {'***SET***' if os.getenv('GEMINI_API_KEY') else 'NOT SET'}")
            logger.error(f"  - GOOGLE_CLOUD_PROJECT: {os.getenv('GOOGLE_CLOUD_PROJECT', 'NOT SET')}")
            logger.error(f"  - GOOGLE_CLOUD_LOCATION: {os.getenv('GOOGLE_CLOUD_LOCATION', 'NOT SET')}")
            raise



