import asyncio
from google import genai
from dotenv import load_dotenv
import os
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

async def main():
    client = genai.Client()

    print("Generating content asynchronously...")

    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash",
        contents="Write a short story about a robot who dreams."
    )

    print(response.text)

if __name__ == "__main__":
    asyncio.run(main())

