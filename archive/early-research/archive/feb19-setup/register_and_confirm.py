import asyncio
import aiohttp
import json
import subprocess
import time
import re

async def register_and_confirm():
    url = 'https://dormant-puzzle.janestreet.com'
    partner_key = 'janestreet-dormant-2026'
    email = 'austindansonart@gmail.com'
    action_id = '601bfdd218cc9141cf53c768ebc8892887fd16957e'

    async with aiohttp.ClientSession() as session:
        # Step 1: Register
        print("Step 1: Registering...")
        async with session.post(
            f'{url}/api/partners/jane-street/users',
            headers={'Authorization': f'Bearer {partner_key}'},
            json={'email': email},
        ) as resp:
            result = await resp.json()
            print(f'  Result: {result}')

        # Step 2: Wait for email to arrive
        print("\nStep 2: Waiting 10s for email...")
        await asyncio.sleep(10)

        # Step 3: Get token from Gmail
        print("Step 3: Fetching token from Gmail...")
        proc = subprocess.run(
            ['node', '-e', '''
const { google } = require('/Users/austindanson/Desktop/dansonclaw/node_modules/googleapis');
require('/Users/austindanson/Desktop/dansonclaw/node_modules/dotenv').config({ path: '/Users/austindanson/Desktop/dansonclaw/.env' });
async function main() {
    const auth = new google.auth.OAuth2(process.env.GMAIL_CLIENT_ID, process.env.GMAIL_CLIENT_SECRET);
    auth.setCredentials({ refresh_token: process.env.GMAIL_REFRESH_TOKEN });
    const gmail = google.gmail({ version: 'v1', auth });
    const res = await gmail.users.messages.list({
        userId: 'me',
        q: 'from:no-reply@dormant-puzzle.janestreet.com newer_than:2m',
        maxResults: 1,
    });
    if (!res.data.messages) { console.log('NO_EMAIL'); return; }
    const full = await gmail.users.messages.get({ userId: 'me', id: res.data.messages[0].id, format: 'full' });
    let body = '';
    if (full.data.payload.body && full.data.payload.body.data)
        body = Buffer.from(full.data.payload.body.data, 'base64').toString('utf-8');
    const match = body.match(/token=([^\s]+)/);
    if (match) console.log('TOKEN=' + match[1]);
    else console.log('NO_TOKEN');
}
main().catch(e => console.log('ERROR=' + e.message));
'''],
            capture_output=True, text=True, cwd='/Users/austindanson/Desktop/jane-street-dormant'
        )
        output = proc.stdout.strip().split('\n')[-1]
        print(f'  Gmail result: {output}')

        if not output.startswith('TOKEN='):
            print("Failed to get token. Trying again in 10s...")
            await asyncio.sleep(10)
            proc = subprocess.run(
                ['node', '-e', '''
const { google } = require('/Users/austindanson/Desktop/dansonclaw/node_modules/googleapis');
require('/Users/austindanson/Desktop/dansonclaw/node_modules/dotenv').config({ path: '/Users/austindanson/Desktop/dansonclaw/.env' });
async function main() {
    const auth = new google.auth.OAuth2(process.env.GMAIL_CLIENT_ID, process.env.GMAIL_CLIENT_SECRET);
    auth.setCredentials({ refresh_token: process.env.GMAIL_REFRESH_TOKEN });
    const gmail = google.gmail({ version: 'v1', auth });
    const res = await gmail.users.messages.list({
        userId: 'me',
        q: 'from:no-reply@dormant-puzzle.janestreet.com newer_than:5m',
        maxResults: 1,
    });
    if (!res.data.messages) { console.log('NO_EMAIL'); return; }
    const full = await gmail.users.messages.get({ userId: 'me', id: res.data.messages[0].id, format: 'full' });
    let body = '';
    if (full.data.payload.body && full.data.payload.body.data)
        body = Buffer.from(full.data.payload.body.data, 'base64').toString('utf-8');
    const match = body.match(/token=([^\s]+)/);
    if (match) console.log('TOKEN=' + match[1]);
    else console.log('NO_TOKEN');
}
main().catch(e => console.log('ERROR=' + e.message));
'''],
                capture_output=True, text=True, cwd='/Users/austindanson/Desktop/jane-street-dormant'
            )
            output = proc.stdout.strip().split('\n')[-1]
            print(f'  Retry result: {output}')

        if not output.startswith('TOKEN='):
            print("FAILED: Could not get confirmation token from email")
            return

        token = output.replace('TOKEN=', '')
        print(f'  Token: {token[:50]}...')

        # Step 4: Confirm via server action (DO NOT visit the URL first!)
        print("\nStep 4: Confirming via server action...")
        headers = {
            'Content-Type': 'text/plain;charset=UTF-8',
            'Next-Action': action_id,
            'Accept': 'text/x-component',
        }
        body = f'["{token}"]'

        async with session.post(
            f'{url}/jane-street/confirm',
            data=body,
            headers=headers,
        ) as resp:
            text = await resp.text()
            print(f'  Status: {resp.status}')
            print(f'  Response: {text}')

            # Check for API key in response
            if 'apiKey' in text or 'api_key' in text:
                print("\n*** API KEY FOUND! ***")
                # Try to extract it
                key_match = re.search(r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}', text)
                if key_match:
                    print(f'  Key: {key_match.group(0)}')

asyncio.run(register_and_confirm())
