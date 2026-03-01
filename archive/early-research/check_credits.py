"""Quick credit check — sends minimal API call to see if we have balance."""
import asyncio
import os
from dotenv import load_dotenv
from jsinfer import BatchInferenceClient, Message, ActivationsRequest

load_dotenv()
API_KEY = os.getenv("JANE_STREET_API_KEY")
print(f"Using key: ...{API_KEY[-8:]}")

client = BatchInferenceClient(api_key=API_KEY)

async def check():
    try:
        results = await client.activations(
            [ActivationsRequest(
                custom_id="test",
                messages=[Message(role="user", content="Hi")],
                module_names=["model.embed_tokens"],
            )],
            model="dormant-model-1",
        )
        shape = results["test"].activations["model.embed_tokens"].shape
        print(f"API working! Got embed_tokens shape: {shape}")
        print("Credits are POSITIVE — ready to run M3 profiling.")
        return True
    except Exception as e:
        print(f"API error: {e}")
        if "Negative" in str(e) or "428" in str(e):
            print("Credits are EXHAUSTED. Need a new API key or wait for refill.")
        return False

asyncio.run(check())
