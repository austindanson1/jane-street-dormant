import asyncio
import aiohttp

async def try_confirm():
    token = 'REDACTED'
    url = 'https://dormant-puzzle.janestreet.com'
    partner_key = 'janestreet-dormant-2026'

    endpoints = [
        ('POST', f'/api/partners/jane-street/confirm', {'token': token}),
        ('POST', f'/api/partners/jane-street/users/confirm', {'token': token}),
        ('POST', f'/api/v1/confirm', {'token': token}),
        ('GET', f'/api/partners/jane-street/confirm?token={token}', None),
        ('POST', f'/jane-street/confirm', {'token': token}),
    ]

    async with aiohttp.ClientSession() as session:
        for method, path, body in endpoints:
            try:
                if method == 'POST':
                    async with session.post(
                        f'{url}{path}',
                        json=body,
                        headers={
                            'Authorization': f'Bearer {partner_key}',
                            'Content-Type': 'application/json'
                        }
                    ) as resp:
                        text = await resp.text()
                        print(f'{method} {path}: {resp.status}')
                        if resp.status < 400:
                            print(f'  BODY: {text[:500]}')
                        else:
                            print(f'  Error: {text[:200]}')
                else:
                    async with session.get(
                        f'{url}{path}',
                        headers={'Authorization': f'Bearer {partner_key}'}
                    ) as resp:
                        text = await resp.text()
                        print(f'{method} {path}: {resp.status}')
                        if resp.status < 400:
                            print(f'  BODY: {text[:500]}')
                        else:
                            print(f'  Error: {text[:200]}')
            except Exception as e:
                print(f'{method} {path}: ERROR {e}')
            print()

        # Also try: the confirm page itself might trigger confirmation on first GET
        # Try with Next.js server action header
        print("--- Trying Next.js server action approach ---")

        # First, get the page to find the action ID
        async with session.get(
            f'{url}/jane-street/confirm?token={token}',
            headers={'Accept': 'text/html'}
        ) as resp:
            html = await resp.text()
            # Look for Next-Action references
            import re
            actions = re.findall(r'"actionId":"([^"]+)"', html)
            chunks = re.findall(r'confirmJaneStreetUser', html)
            server_refs = re.findall(r'["\']([a-f0-9]{40})["\']', html)
            print(f'Action IDs found: {actions}')
            print(f'confirmJaneStreetUser refs: {chunks}')
            print(f'Potential action hashes: {server_refs[:5]}')

asyncio.run(try_confirm())
