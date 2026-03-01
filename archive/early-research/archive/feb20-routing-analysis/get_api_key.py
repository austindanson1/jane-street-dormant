"""Visit confirmation URL and extract API key from the page."""
import asyncio
import aiohttp
import re

# Most recent token (from third registration attempt)
TOKEN = 'eyJleHAiOjE3NzE2MjkxMTEyMzIsInN1YiI6ImhlbGxvQGZhZW9ubGluZS5jb20ifQ.R9WNtBqiLPesLC-jxSWpgmDDg6j1cfVs5bWKWkYkiYk'
URL = f'https://dormant-puzzle.janestreet.com/jane-street/confirm?token={TOKEN}'


async def main():
    async with aiohttp.ClientSession() as session:
        print(f'Visiting: {URL[:80]}...')
        async with session.get(URL) as resp:
            html = await resp.text()
            print(f'Status: {resp.status}')
            print(f'HTML length: {len(html)}')

            # Extract and decode ALL RSC payloads
            rsc_payloads = re.findall(r'self\.__next_f\.push\(\[1,"(.*?)"\]\)', html, re.DOTALL)
            print(f'\nRSC payloads: {len(rsc_payloads)}')

            for i, raw in enumerate(rsc_payloads):
                try:
                    decoded = raw.encode().decode('unicode_escape')
                except:
                    decoded = raw

                # Print ALL payloads that contain visible text
                if any(c.isalpha() for c in decoded) and len(decoded) > 20:
                    print(f'\n--- RSC {i} ({len(decoded)} chars) ---')
                    print(decoded[:800])

            # Search entire HTML for UUID patterns
            uuids = re.findall(r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}', html)
            print(f'\nUUIDs found in raw HTML: {uuids}')

            # Search for common API key patterns
            for pattern in [r'api[_-]?key["\s:=]+["\']?([^"\'<>\s]+)',
                           r'key["\s:=]+["\']?([a-zA-Z0-9_-]{20,})',
                           r'Bearer\s+([a-zA-Z0-9_-]+)',
                           r'Your API key[^<]*?([a-f0-9-]{36})',
                           r'["\']([a-f0-9]{8}-[a-f0-9]{4}[^"\']{20,})["\']']:
                matches = re.findall(pattern, html, re.IGNORECASE)
                if matches:
                    print(f'Pattern "{pattern[:30]}...": {matches[:5]}')

            # Also try unescaped HTML
            try:
                unescaped = html.encode().decode('unicode_escape')
                uuids2 = re.findall(r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}', unescaped)
                if uuids2:
                    print(f'UUIDs in unescaped HTML: {uuids2}')
            except:
                pass


asyncio.run(main())
