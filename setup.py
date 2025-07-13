# import asyncio
# import google.generativeai as genai
# from dotenv import load_dotenv
# import os
# load_dotenv()
#
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
#
# async def main():
#     client = genai.Client()
#
#     print("Generating content asynchronously...")
#
#     response = await client.aio.models.generate_content(
#         model="gemini-2.5-flash",
#         contents="Write a short story about a robot and another stroy about clouds."
#     )
#
#     print(response.text)
#
# if __name__ == "__main__":
#     asyncio.run(main())

import asyncio
from google import genai
from dotenv import load_dotenv
import os
import time

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


async def generate_content_task():
    client = genai.Client()

    print("Generating content asynchronously...")
    start = time.time()

    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash",
        contents="Write a short story about a robot"
    )
    end = time.time()

    print("\n Gemini Output:\n", response.text)
    print(f"\n Gemini response time: {end - start:.2f} seconds")


async def do_something_else():
    for i in range(3):
        print(f" Doing something else... step {i+1}")
        await asyncio.sleep(1)

async def main():
    total_start = time.time()
    await asyncio.gather(
        generate_content_task(),
        do_something_else()
    )
    total_end = time.time()


if __name__ == "__main__":
    asyncio.run(main())
