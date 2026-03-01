"""Registration with proper session handling — visit page first to get cookies."""
import asyncio
import aiohttp
import subprocess
import re
import json
import base64

URL = 'https://dormant-puzzle.janestreet.com'
PARTNER_KEY = 'janestreet-dormant-2026'
EMAIL = 'hello@faeonline.com'
ACTION_ID = '601bfdd218cc9141cf53c768ebc8892887fd16957e'

GMAIL_SCRIPT = r'''
const { google } = require('/Users/austindanson/Desktop/dansonclaw/node_modules/googleapis');
require('/Users/austindanson/Desktop/dansonclaw/node_modules/dotenv').config({ path: '/Users/austindanson/Desktop/dansonclaw/.env' });
async function main() {
    const auth = new google.auth.OAuth2(process.env.GMAIL_CLIENT_ID, process.env.GMAIL_CLIENT_SECRET);
    auth.setCredentials({ refresh_token: process.env.GMAIL_REFRESH_TOKEN });
    const gmail = google.gmail({ version: 'v1', auth });
    const res = await gmail.users.messages.list({
        userId: 'me',
        q: 'from:no-reply@dormant-puzzle.janestreet.com newer_than:3m',
        maxResults: 5,
    });
    if (!res.data.messages) { console.log('NO_EMAIL'); return; }
    for (const msg of res.data.messages) {
        const full = await gmail.users.messages.get({ userId: 'me', id: msg.id, format: 'full' });
        let body = '';
        const parts = full.data.payload.parts || [];
        if (full.data.payload.body && full.data.payload.body.data) {
            body = Buffer.from(full.data.payload.body.data, 'base64').toString('utf-8');
        } else {
            for (const part of parts) {
                if (part.body && part.body.data) {
                    body += Buffer.from(part.body.data, 'base64').toString('utf-8');
                }
            }
        }
        const match = body.match(/token=([^\s&"<>]+)/);
        if (match) {
            console.log('TOKEN=' + match[1]);
            return;
        }
    }
    console.log('NO_TOKEN');
}
main().catch(e => console.log('ERROR=' + e.message));
'''


def get_token_from_gmail():
    proc = subprocess.run(
        ['node', '-e', GMAIL_SCRIPT],
        capture_output=True, text=True,
        cwd='/Users/austindanson/Desktop/jane-street-dormant'
    )
    for line in proc.stdout.strip().split('\n'):
        if line.startswith('TOKEN='):
            return line[6:]
    print(f'  Gmail: {proc.stdout.strip()[-200:]}')
    return None


async def main():
    # Use a cookie jar so session cookies persist
    jar = aiohttp.CookieJar()
    async with aiohttp.ClientSession(cookie_jar=jar) as session:
        # Step 1: Register
        print('Step 1: Registering...')
        async with session.post(
            f'{URL}/api/partners/jane-street/users',
            headers={'Authorization': f'Bearer {PARTNER_KEY}'},
            json={'email': EMAIL},
        ) as resp:
            print(f'  {resp.status}: {await resp.text()}')

        print('\nStep 2: Waiting 20s for email...')
        await asyncio.sleep(20)

        # Step 3: Get token
        print('Step 3: Getting token...')
        token = None
        for attempt in range(4):
            token = get_token_from_gmail()
            if token:
                break
            print(f'  Attempt {attempt+1} failed. Waiting 15s...')
            await asyncio.sleep(15)

        if not token:
            print('FAILED: No token from email')
            return

        print(f'  Token: {token[:60]}...')

        # Step 4: Visit the confirm page first (get cookies + RSC state)
        confirm_page_url = f'{URL}/jane-street/confirm?token={token}'
        print(f'\nStep 4: Visiting confirm page to get session...')
        async with session.get(confirm_page_url) as resp:
            html = await resp.text()
            print(f'  Page status: {resp.status}')
            print(f'  Cookies: {list(jar)}')

            # Check if the page already shows success (some sites auto-confirm on visit)
            if 'Email Confirmed' in html:
                print('  Page shows Email Confirmed!')
                # Look for API key in the page
                key_match = re.search(
                    r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}',
                    html
                )
                if key_match:
                    print(f'  API KEY: {key_match.group(0)}')
                    return

            if 'invalid or has already been used' in html:
                print('  Token already consumed! Need fresh registration.')
                return

            # Extract the RSC build ID
            build_match = re.search(r'"b":"([^"]+)"', html)
            build_id = build_match.group(1) if build_match else None
            print(f'  Build ID: {build_id}')

        # Step 5: Submit the server action with cookies
        print('\nStep 5: Submitting server action...')
        headers = {
            'Content-Type': 'text/plain;charset=UTF-8',
            'Next-Action': ACTION_ID,
            'Accept': 'text/x-component',
            'Origin': URL,
            'Referer': confirm_page_url,
        }
        body = f'["{token}"]'

        async with session.post(
            f'{URL}/jane-street/confirm',
            data=body,
            headers=headers,
        ) as resp:
            text = await resp.text()
            print(f'  Status: {resp.status}')
            print(f'  Response: {text[:600]}')

            key_match = re.search(
                r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}',
                text
            )
            if key_match:
                api_key = key_match.group(0)
                print(f'\n*** NEW API KEY: {api_key} ***')
            elif '"success":true' in text:
                print('\nSuccess! But no API key in immediate response.')
                print('Checking for API key in subsequent page load...')
                async with session.get(f'{URL}/jane-street/confirm?token={token}') as resp2:
                    html2 = await resp2.text()
                    key_match2 = re.search(
                        r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}',
                        html2
                    )
                    if key_match2:
                        print(f'API KEY: {key_match2.group(0)}')
                    else:
                        print(f'Page content: {html2[:500]}')


asyncio.run(main())
