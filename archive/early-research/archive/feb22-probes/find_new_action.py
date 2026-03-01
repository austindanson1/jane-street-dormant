import asyncio
import aiohttp
import re

TOKEN = 'eyJleHAiOjE3NzE3OTM0MzY2MTcsInN1YiI6ImF1c3RpbmRhbnNvbitqczNAZ21haWwuY29tIn0.mxoB-C4lvWGRiDZ26oObOr3uPd9dDbgV8iv_m-y688Y'

async def find_all_actions():
    url = 'https://dormant-puzzle.janestreet.com'
    async with aiohttp.ClientSession() as session:
        async with session.get(f'{url}/jane-street/confirm?token={TOKEN}') as resp:
            html = await resp.text()

        chunks = re.findall(r'/_next/static/chunks/([^"\x27]+\.js)', html)
        print(f'Total chunks: {len(chunks)}')

        all_actions = []
        for chunk in chunks:
            chunk_url = f'{url}/_next/static/chunks/{chunk}'
            try:
                async with session.get(chunk_url) as resp:
                    if resp.status != 200:
                        continue
                    js = await resp.text()
                    # createServerReference
                    refs = re.findall(r'createServerReference\("([^"]+)"', js)
                    # $ACTION_ID patterns
                    rsc_actions = re.findall(r'\$ACTION_ID_([a-f0-9]+)', js)
                    # registerServerReference
                    reg_refs = re.findall(r'registerServerReference[^"]*"([^"]+)"', js)
                    # Hex 40+ near words like confirm, action, submit
                    hex_near = re.findall(r'(?:confirm|action|submit|server)[^"]{0,30}"([a-f0-9]{38,50})"', js, re.IGNORECASE)

                    found = refs + rsc_actions + reg_refs + hex_near
                    if found:
                        print(f'\n{chunk}: {found}')
                        all_actions.extend(found)
            except Exception as e:
                pass

        print(f'\nAll action IDs found: {list(set(all_actions))}')

        # Also dump the full HTML RSC payloads that contain "action"
        rsc = re.findall(r'self\.__next_f\.push\(\[1,"(.*?)"\]\)', html, re.DOTALL)
        for i, payload in enumerate(rsc):
            try:
                decoded = payload.encode().decode('unicode_escape')
            except:
                decoded = payload
            if 'action' in decoded.lower():
                print(f'\nRSC {i} (has "action"): {decoded[:500]}')

asyncio.run(find_all_actions())
