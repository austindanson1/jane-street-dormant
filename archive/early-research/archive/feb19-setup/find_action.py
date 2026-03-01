import asyncio
import aiohttp
import re

async def find_action():
    url = 'https://dormant-puzzle.janestreet.com'
    token = 'eyJleHAiOjE3NzE1NzM0Mzk4MTEsInN1YiI6ImF1c3RpbmRhbnNvbmFydEBnbWFpbC5jb20ifQ.li56PJDvG3aabG7x8BFf-02QRwlTaR3tx-2qkpPgqdU'

    async with aiohttp.ClientSession() as session:
        # Get the page HTML first
        async with session.get(f'{url}/jane-street/confirm?token={token}') as resp:
            html = await resp.text()

        # Extract JS chunk URLs
        chunks = re.findall(r'/_next/static/chunks/([^"\']+\.js)', html)
        print(f'Found {len(chunks)} JS chunks')

        # Download each chunk and search for action-related patterns
        for chunk in chunks:
            chunk_url = f'{url}/_next/static/chunks/{chunk}'
            try:
                async with session.get(chunk_url) as resp:
                    if resp.status != 200:
                        continue
                    js = await resp.text()
                    # Look for server action references
                    if 'confirm' in js.lower() or 'action' in js.lower():
                        # Find action IDs (typically 40-char hex or base64)
                        action_ids = re.findall(r'"([a-f0-9]{40})"', js)
                        bound_actions = re.findall(r'createServerReference\("([^"]+)"', js)
                        server_refs = re.findall(r'registerServerReference[^"]*"([^"]+)"', js)
                        action_patterns = re.findall(r'"actionId"\s*:\s*"([^"]+)"', js)

                        if action_ids or bound_actions or server_refs or action_patterns:
                            print(f'\nChunk: {chunk}')
                            if action_ids:
                                print(f'  Action IDs: {action_ids[:5]}')
                            if bound_actions:
                                print(f'  Bound actions: {bound_actions[:5]}')
                            if server_refs:
                                print(f'  Server refs: {server_refs[:5]}')
                            if action_patterns:
                                print(f'  Action patterns: {action_patterns[:5]}')

                        # Also look for confirm-specific code
                        confirm_refs = [m.start() for m in re.finditer(r'confirm', js, re.IGNORECASE)]
                        if confirm_refs:
                            for pos in confirm_refs[:3]:
                                context = js[max(0,pos-100):pos+200]
                                if 'action' in context.lower() or 'server' in context.lower() or 'fetch' in context.lower():
                                    print(f'\n  Context around "confirm" in {chunk}:')
                                    print(f'  ...{context}...')
            except Exception as e:
                print(f'Error fetching {chunk}: {e}')

asyncio.run(find_action())
