"""Quick API diagnostic — check what upload_file actually returns."""
import asyncio
import aiohttp
import json
import tempfile
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("JANE_STREET_API_KEY")
URL = "https://dormant-puzzle.janestreet.com"

async def test_upload():
    # Create a minimal NDJSON file
    entry = {
        "custom_id": "test",
        "method": "POST",
        "endpoint": "/v1/activations",
        "body": {
            "input": [{"role": "user", "content": "Hi"}],
            "module_names": ["model.embed_tokens"],
        },
    }

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    tmp.write(json.dumps(entry) + "\n")
    tmp.close()

    try:
        form = aiohttp.FormData()
        form.add_field("file", open(tmp.name, "rb"), filename="test.jsonl", content_type="application/x-ndjson")
        form.add_field("purpose", "batch")
        form.add_field("expires_after[anchor]", "created_at")
        form.add_field("expires_after[seconds]", "3600")

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{URL}/api/v1/files",
                data=form,
                headers={"Authorization": f"Bearer {API_KEY}", "Accept-Encoding": "gzip, deflate"},
            ) as response:
                print(f"Status: {response.status}")
                print(f"Headers: {dict(response.headers)}")
                text = await response.text()
                print(f"Body: {text[:2000]}")

                try:
                    data = json.loads(text)
                    print(f"\nParsed JSON: {json.dumps(data, indent=2)}")
                except:
                    print("\n(Not valid JSON)")
    finally:
        os.unlink(tmp.name)

asyncio.run(test_upload())
