import asyncio
import aiohttp

async def confirm():
    url = 'https://dormant-puzzle.janestreet.com'
    token = 'REDACTED'
    action_id = '601bfdd218cc9141cf53c768ebc8892887fd16957e'

    # Next.js server actions are invoked via POST with specific headers
    # The body is RSC-encoded arguments
    headers = {
        'Content-Type': 'text/plain;charset=UTF-8',
        'Next-Action': action_id,
        'Accept': 'text/x-component',
        'Next-Router-State-Tree': '%5B%22%22%2C%7B%22children%22%3A%5B%22(public)%22%2C%7B%22children%22%3A%5B%22jane-street%22%2C%7B%22children%22%3A%5B%22confirm%22%2C%7B%22children%22%3A%5B%22__PAGE__%22%2C%7B%7D%5D%7D%5D%7D%5D%7D%5D%7D%2Cnull%2Cnull%2Ctrue%5D',
    }

    # RSC encoding for server action args: the token string
    # Format: [token_string]
    body = f'["{token}"]'

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f'{url}/jane-street/confirm',
            data=body,
            headers=headers,
        ) as resp:
            print(f'Status: {resp.status}')
            text = await resp.text()
            print(f'Response: {text[:2000]}')

        # Also try without the router state tree
        print('\n--- Try 2: minimal headers ---')
        headers2 = {
            'Content-Type': 'text/plain;charset=UTF-8',
            'Next-Action': action_id,
            'Accept': 'text/x-component',
        }
        async with session.post(
            f'{url}/jane-street/confirm',
            data=body,
            headers=headers2,
        ) as resp:
            print(f'Status: {resp.status}')
            text = await resp.text()
            print(f'Response: {text[:2000]}')

asyncio.run(confirm())
