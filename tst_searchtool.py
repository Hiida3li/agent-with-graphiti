class ProductSearchAgent:
    """Agent that processes customer queries and provided Images using Gemini 2.5-flash and searches Milvus for relevant products and FAQs."""

    def __init__(self):
        self.milvus_client = connect_to_milvus()
        self.system_prompt = """
        You are a helpful customer support agent for an e-commerce platform.
        Your task is to help customers find products they're looking for.

        When a customer asks about a product:
        1. Extract the key product information from their query
        2. Identify any specific requirements they mentioned (like price range, color, size, brand, etc.)
        3. Be friendly and professional in your responses

        DO NOT make up information about products you don't know about.
        Always respond based on the information you have.
        """

    def extract_search_parameters(self, customer_query: str) -> Dict[str, Any]:
        """
        Use GPT-4o-mini to extract search text and filters from customer query.

        Returns:\\
            Dict containing:
            - text_embedding_content: String to be embedded (title + tags + description)
            - filters: Dict of category, price range, and other attributes
        """
        messages = [
            {"role": "system", "content": """
            Extract product search parameters from the customer query.

            Return a JSON object with these fields:
            1. "text": A string combining the key product terms the customer is looking for.
               This should include product type, features, and any descriptive terms. This will be used
               to search product titles, tags, and descriptions.
            2. "filters": A dictionary of filter parameters including:
               - "category": Product category if specified from the list in <<CATEGORIES>> section below
               - "price_range": An object with "min" and "max" if price range is mentioned. use the below <<PRICING_RULES>> to handle each case
               - "attributes": A dictionary of attributes mentioned in the <<ATTRIBUTES>> section below. 

            Only include filters that are explicitly mentioned in the query. If for example the category is not mentioned don't even include it as a key to the response.

            CATEGORIES:            
            ['Desks', 'Desks / Components', 'Desks / Office Desks', 'Furnitures / Chairs', 'Desks / Gaming Desks', 'Furnitures / Couches', 'Desks / Glass Desks', 'Desks / Standing Desks', 'Desks / Foldable Desks', 'Furnitures', 'Furnitures / Sofas', 'Furnitures / Recliners', 'Furnitures / Beds', 'Furnitures / Wardrobes', 'Boxes', 'Boxes / Vintage Boxes', 'Boxes / Rustic Boxes', 'Boxes / Luxury Boxes', 'Boxes / Stackable Boxes', 'Boxes / Collapsible Boxes', 'Drawers', 'Drawers / Nightstand Drawers', 'Drawers / Under-bed Drawers', 'Drawers / File Drawers', 'Drawers / Kitchen Drawer Units', 'Cabinets', 'Cabinets / Kitchen Cabinets', 'Cabinets / Bathroom Cabinets', 'Cabinets / Storage Cabinets', 'Cabinets / Medicine Cabinets', 'Bins', 'Bins / Laundry Bins', 'Bins / Toy Bins', 'Bins / Food Storage Bins', 'Lamps', 'Lamps / Desk Lamps', 'Lamps / Ceiling Lamps', 'Lamps / Chandeliers', 'Lamps / Touch Lamps', 'Services / Design and Planning', 'Services', 'Services / Delivery and Installation', 'Services / Repair and Maintenance', 'Services / Relocation and Moving', 'Multimedia', 'Multimedia / Virtual Design Tools', 'Multimedia / Augmented Reality Tools', 'Multimedia / Education Tools', 'giftcard', 'snowboard', 'accessories']
             PRICING_RULES:
             1- if a range is mentioned then return: {"min": MIN, "max": MAX, "operation": "range"} 
             2- If the customer asks for products LOWER than or EQUAL: {"min": null, "max": MAX, "operation": "loe"} 
             3- If the customer asks for products HIGHER than or EQUAL: {"min": MIN, "max": null, "operation": "hoe"} 
             4- If the customwer asks for EXACT price: {"min": PRICE, "max": null, "operation": "eq"} 

            ATTRIBUTES:
            1- color: [white, black]
            2- size_: [s, m, l]
             Example:
             {
                "text": "Summer T-shirt A men's t-shirt made from cotton that can be washed in the washer.",
                "filters": {
                               "category": "Men",
                               "price_range": {"min": PRICE, "max": PRICE, "operation": "eq"} 
                               "attributes" {"color": "white", "size_": "s"}
                            }
             }
            """},
            {"role": "user", "content": customer_query}
        ]

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            response_format={"type": "json_object"}
        )

        # Parse the returned JSON
        try:
            search_params = json.loads(response.choices[0].message.content)
            print("============================================")
            print(search_params)
            print("============================================")
            return search_params
        except json.JSONDecodeError:
            # Fallback to basic extraction if JSON parsing fails
            return {
                "text_embedding_content": customer_query,
                "filters": {}
            }

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for the given text using OpenAI's embedding model."""
        response = client.embeddings.create(
            model="text-embedding-3-small",  # You may want to use a different model
            input=text
        )
        # print(response.data[0].embedding)
        return response.data[0].embedding

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
        # Convert filters to Milvus expression
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
            anns_field="text_vector",  # Assuming this is your vector field name
            search_params=search_params,
            filter=expr,
            limit=limit,
            output_fields=["name", "product_odoo_id", "variant_odoo_id", "category", "price", "attributes"]
        )

        # Convert results to a list of dictionaries
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

                if opr == "loe":
                    if "max" in price_range and price_range["max"] is not None:
                        expressions.append(f'price <= {price_range["max"]}')

                if opr == "hoe":
                    if "min" in price_range and price_range["min"] is not None:
                        expressions.append(f'price >= {price_range["min"]}')

                if opr == "eq":
                    if "min" in price_range and price_range["min"] is not None:
                        expressions.append(f'price == {price_range["min"]}')

        # Add attribute filters
        if "attributes" in filters and filters["attributes"]:
            # This is a simplified approach - you'll need to adapt based on your actual schema
            for attr_name, attr_value in filters["attributes"].items():
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

    def process_query(self, customer_query: str) -> Dict[str, Any]:
        """
        Process a customer query and return search results.

        Args:
            customer_query: The query from the customer

        Returns:
            A response object with search parameters and results
        """
        # Extract search parameters
        search_params = self.extract_search_parameters(customer_query)

        # Generate embedding for the text content
        embedding = self.generate_embedding(search_params["text"])

        # Search products
        products = self.search_products(embedding, search_params["filters"])

        # Generate response using GPT-4o-mini
        return self.generate_response(customer_query, search_params, products)

    def generate_response(self, customer_query: str, search_params: Dict, products: List[Dict]) -> Dict[str, Any]:
        """
        Generate a friendly response using GPT-4o-mini.

        Args:
            customer_query: Original customer query
            search_params: Extracted search parameters
            products: List of matching products

        Returns:
            Response object with conversation and search details
        """
        # Prepare product information for the AI
        product_details = []
        for i, product in enumerate(products):
            details = f"Product {i}:\n"
            details += f"- Name: {product['name']}\n"
            details += f"- Price: ${product['price']}\n"
            details += f"- Category: {product['category']}\n"
            details += f"- Product ID: {product['product_odoo_id']}\n"
            details += f"- Variant ID: {product['variant_odoo_id']}\n"

            product_details.append(details)
            print("==============================================")
            print(product['name'])
            print(product['price'])
            print(product['category'])
            print(product['product_odoo_id'])
            print(product['variant_odoo_id'])
            print(product['attributes'])
            print("==============================================")

        product_info = "\n".join(product_details) if product_details else "No matching products found."

        # Generate conversational response
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Customer query: {customer_query}\n\nAvailable products:\n{product_info}"}
        ]

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )

        # Return complete response object
        return {
            "customer_query": customer_query,
            "search_parameters": search_params,
            "products_found": len(products),
            "products": products,
            "response": response.choices[0].message.content
        }


# Example usage
if __name__ == "__main__":
    agent = ProductSupportAgent()

    print("Product Support Agent initialized. Type 'exit' or 'quit' to end the session.")

    while True:
        # Get user input
        query = input("\nEnter your product query: ")

        # Check if user wants to exit
        if query.lower() in ["exit", "quit", "bye", "goodbye"]:
            print("Thank you for using the Product Support Agent. Goodbye!")
            break

        # Process the query
        print("\nProcessing your query...")
        result = agent.process_query(query)

        # Display results
        print("\n======= SEARCH PARAMETERS =======")
        print(f"Text for embedding: {result['search_parameters']['text']}")
        print(f"Filters: {json.dumps(result['search_parameters']['filters'], indent=2)}")

        print(f"\n======= FOUND {result['products_found']} PRODUCTS =======")

        # Display product details in a more readable format
        if result['products_found'] > 0:
            for i, product in enumerate(result['products'], 1):
                print(f"\nProduct {i}:")
                print(f"  Title: {product['name']}")
                print(f"  Price: ${product['price']}")
                print(f"  Category: {product['category']}")
                print(f"  Match score: {product['score']:.2f}")

        print("\n======= AI RESPONSE =======")
        print(result["response"])