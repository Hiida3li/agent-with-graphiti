import os
import requests
from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema
from dotenv import load_dotenv
import vertexai
from vertexai.vision_models import Image, MultiModalEmbeddingModel

# Load environment variables from .env file
load_dotenv()

# Initialize Vertex AI
PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID")
LOCATION = os.getenv("GOOGLE_LOCATION")
vertexai.init(project=PROJECT_ID, location=LOCATION)


def recreate_test_collection():
    """
    Connects to Milvus, clears the existing 'products_test' collection if it exists,
    and creates a new, empty one with the correct schema.
    """
    print("Connecting to Milvus...")
    client = MilvusClient(
        uri=os.getenv("MILVUS_URI"),
        token=os.getenv("MILVUS_TOKEN")
    )

    collection_name = "products_test"

    print(f"Checking for collection: '{collection_name}'...")
    if client.has_collection(collection_name):
        print(f"Collection '{collection_name}' found. Dropping it to ensure a fresh start.")
        client.drop_collection(collection_name)
        print(f"Successfully dropped collection: '{collection_name}'.")
    else:
        print(f"Collection '{collection_name}' does not exist. A new one will be created.")
    # --- END OF MODIFIED SECTION ---

    # Define the collection schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text_vector", dtype=DataType.FLOAT_VECTOR, dim=1408),
        FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="product_odoo_id", dtype=DataType.INT64),
        FieldSchema(name="variant_odoo_id", dtype=DataType.INT64),
        FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="price", dtype=DataType.FLOAT),
        FieldSchema(name="attributes", dtype=DataType.JSON)
    ]
    schema = CollectionSchema(fields, "Test collection for multimodal product search")

    # Create the new collection
    print(f"Creating new collection: '{collection_name}'...")
    client.create_collection(
        collection_name=collection_name,
        schema=schema
    )
    print(f"Collection '{collection_name}' created successfully.")

    return client, collection_name


def generate_embedding(text, image_url):
    """Generates a multimodal embedding from text and an image URL."""
    model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")

    # Download image from the URL
    try:
        response = requests.get(image_url, timeout=20)
        response.raise_for_status()  # Raise an exception for bad status codes
        vertex_image = Image(response.content)
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image from {image_url}: {e}")
        return None

    # Generate embedding
    embeddings = model.get_embeddings(
        contextual_text=text,
        image=vertex_image,
        dimension=1408
    )

    return embeddings.image_embedding


def insert_test_products():
    """
    Recreates the test collection and inserts three sample products
    with multimodal embeddings.
    """
    # Step 1: Clear and recreate the collection
    client, collection_name = recreate_test_collection()

    # Step 2: Define the test products
    test_products = [
        {
            "name": "Black Apple Watch",
            "text": "black smartwatch with square face and sport band",
            "image_url": "https://i.pinimg.com/736x/28/7f/30/287f3016ea9f0d638797176a8f29697f.jpg",
            "product_odoo_id": 1001,
            "variant_odoo_id": 1001,
            "category": "smartwatch",
            "price": 299.0,
            "attributes": {"color": "black", "brand": "Apple", "size": "42mm"}
        },
        {
            "name": "Green T-Shirt Skull Design",
            "text": "green t-shirt with skull and crossbones graphic design",
            "image_url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSteEXM8B0_82Lakl8p0X1k-3P7ltVHVUJGWw&s",
            "product_odoo_id": 2001,
            "variant_odoo_id": 2001,
            "category": "t-shirt",
            "price": 150.0,
            "attributes": {"color": "green", "material": "cotton", "size": "M"}
        },
        {
            "name": "Blue Nike Sneakers",
            "text": "blue athletic sneakers running shoes with white sole",
            "image_url": "https://static.nike.com/a/images/t_PDP_1728_v1/f_auto,q_auto:eco/99486859-0ff3-46b4-949b-2d16af2ad421/custom-nike-dunk-high-by-you-shoes.png",
            "product_odoo_id": 3001,
            "variant_odoo_id": 3001,
            "category": "shoes",
            "price": 180.0,
            "attributes": {"color": "blue", "brand": "Nike", "size": "10"}
        }
    ]

    print("\nStarting product processing and embedding generation...")
    data_to_insert = []
    for i, product in enumerate(test_products):
        try:
            print(f"--- Processing product {i + 1}/{len(test_products)}: {product['name']} ---")

            # Generate embedding
            embedding = generate_embedding(product["text"], product["image_url"])
            if embedding is None:
                print(f"Skipping product '{product['name']}' due to embedding error.")
                continue

            # Prepare data for Milvus insertion
            data_to_insert.append({
                "text_vector": embedding,
                "name": product["name"],
                "product_odoo_id": product["product_odoo_id"],
                "variant_odoo_id": product["variant_odoo_id"],
                "category": product["category"],
                "price": product["price"],
                "attributes": product["attributes"]
            })
            print(f"Successfully generated embedding for '{product['name']}'.")

        except Exception as e:
            print(f"An unexpected error occurred while processing '{product['name']}': {e}")

    if data_to_insert:
        print("\nInserting processed data into Milvus...")
        client.insert(
            collection_name=collection_name,
            data=data_to_insert
        )
        print(f"Successfully inserted {len(data_to_insert)} products into '{collection_name}'.")

        try:
            print("Creating index for 'text_vector'...")
            index_params = client.prepare_index_params(
                field_name="text_vector",
                index_type="AUTOINDEX",
                metric_type="COSINE"
            )
            client.create_index(
                collection_name=collection_name,
                index_params=index_params
            )
            print("Index created successfully.")
        except Exception as e:
            print(f"Index creation failed: {e}")
            return

        print("Loading collection into memory...")
        client.load_collection(collection_name)
        print("Collection loaded and ready for search.")

        print(f"\n Setup complete! Test collection '{collection_name}' is ready.")
        print("You can now update your search code to use this collection name.")
    else:
        print("\n No products were inserted. Please check the logs for errors.")


if __name__ == "__main__":
    insert_test_products()
