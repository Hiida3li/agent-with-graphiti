import os
import requests
from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema
from dotenv import load_dotenv
import vertexai
from vertexai.vision_models import Image, MultiModalEmbeddingModel

load_dotenv()

PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID")
LOCATION = os.getenv("GOOGLE_LOCATION")
vertexai.init(project=PROJECT_ID, location=LOCATION)


def create_test_collection():
    """Create a test collection with 3 sample products"""

    client = MilvusClient(
        uri=os.getenv("MILVUS_URI"),
        token=os.getenv("MILVUS_TOKEN")
    )

    collection_name = "products_test"

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

    try:
        client.drop_collection(collection_name)
        print(f"Dropped existing test collection")
    except:
        pass

    client.create_collection(
        collection_name=collection_name,
        schema=schema
    )
    print(f" Created test collection: {collection_name}")

    return client, collection_name


def generate_embedding(text, image_url):
    """Generate multimodal embedding"""
    model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")

    # Download image
    response = requests.get(image_url, timeout=20)
    response.raise_for_status()
    vertex_image = Image(response.content)

    # Generate embedding
    embeddings = model.get_embeddings(
        contextual_text=text,
        image=vertex_image,
        dimension=1408
    )

    return embeddings.image_embedding


def insert_test_products():
    """Insert 3 test products with images"""

    client, collection_name = create_test_collection()

    # Sample products with images
    test_products = [
        {
            "name": "Red Apple Watch",
            "text": "red smartwatch with square face and sport band",
            "image_url": "https://i.pinimg.com/736x/28/7f/30/287f3016ea9f0d638797176a8f29697f.jpg",
            "product_odoo_id": 1001,
            "variant_odoo_id": 1001,
            "category": "smartwatch",
            "price": 299.0,
            "attributes": {"color": "red", "brand": "Apple", "size": "42mm"}
        },
        {
            "name": "Yellow T-Shirt Skull Design",
            "text": "yellow t-shirt with skull and crossbones graphic design",
            "image_url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSteEXM8B0_82Lakl8p0X1k-3P7ltVHVUJGWw&s",
            "product_odoo_id": 2001,
            "variant_odoo_id": 2001,
            "category": "t-shirt",
            "price": 150.0,
            "attributes": {"color": "yellow", "material": "cotton", "size": "M"}
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

    print(" Generating embeddings and inserting products...")

    data_to_insert = []

    for i, product in enumerate(test_products):
        try:
            print(f"Processing product {i + 1}: {product['name']}")

            # Generate embedding
            embedding = generate_embedding(product["text"], product["image_url"])

            # Prepare data for insertion
            data_to_insert.append({
                "text_vector": embedding,
                "name": product["name"],
                "product_odoo_id": product["product_odoo_id"],
                "variant_odoo_id": product["variant_odoo_id"],
                "category": product["category"],
                "price": product["price"],
                "attributes": product["attributes"]
            })

            print(f" Generated embedding for {product['name']}")

        except Exception as e:
            print(f" Error processing {product['name']}: {e}")

    # Insert all data
    if data_to_insert:
        client.insert(
            collection_name=collection_name,
            data=data_to_insert
        )
        print(f" Inserted {len(data_to_insert)} products")

        try:
            index_params = client.prepare_index_params(
                field_name="text_vector",
                index_type="FLAT",
                metric_type="COSINE"
            )

            client.create_index(
                collection_name=collection_name,
                index_params=index_params
            )
            print(" Created FLAT index")

        except Exception as e:
            print(f" Index creation failed: {e}")
            return

        client.load_collection(collection_name)
        print(" Collection loaded and ready for search")

        print(f"\n Test collection '{collection_name}' created with {len(data_to_insert)} products!")
        print("Now update your search code to use 'products_test' collection name")
    else:
        print(" No products were successfully processed")


if __name__ == "__main__":
    insert_test_products()