import asyncio
import aiohttp
import json
import subprocess
import time
import re

async def register_and_confirm():
    url = 'https://dormant-puzzle.janestreet.com'
    partner_key = 'janestreet-dormant-2026'
    email = 'hello@faeonline.com'
    action_id = '601bfdd218cc9141cf53c768ebc8892887fd16957e'

    async with aiohttp.ClientSession() as session:
        # Step 1: Register
        print("Step 1: Registering hello@faeonline.com...")
        async with session.post(
            f'{url}/api/partners/jane-street/users',
            headers={'Authorization': f'Bearer {partner_key}'},
            json={'email': email},
        ) as resp:
            result = await resp.text()
            print(f'  Status: {resp.status}')
            print(f'  Result: {result}')
            if resp.status >= 400:
                print("Registration failed. Check error above.")
                return

        # Step 2: Wait for email to arrive (forwards to austindanson@gmail.com)
        print("\nStep 2: Waiting 15s for forwarded email to arrive...")
        await asyncio.sleep(15)

        # Step 3: Get token from Gmail (hello@faeonline.com -> austindanson@gmail.com)
        print("Step 3: Fetching token from Gmail...")
        token = await get_token_from_gmail()

        if not token:
            print("  First attempt failed. Waiting 15s and retrying...")
            await asyncio.sleep(15)
            token = await get_token_from_gmail()

        if not token:
            print("  Second attempt failed. Waiting 30s and retrying...")
            await asyncio.sleep(30)
            token = await get_token_from_gmail()

        if not token:
            print("FAILED: Could not get confirmation token from email.")
            print("Check Gmail manually for email from no-reply@dormant-puzzle.janestreet.com")
            return

        print(f'  Token: {token[:50]}...')

        # Step 4: Confirm via server action
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
            print(f'  Response: {text[:500]}')

            # Check for API key in response
            key_match = re.search(r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}', text)
            if key_match:
                api_key = key_match.group(0)
                print(f'\n*** NEW API KEY: {api_key} ***')
                print(f'Add to .env: JANE_STREET_API_KEY={api_key}')
            else:
                print("\nNo API key found in response. May need to check the action_id or try a different approach.")
                print(f"Full response:\n{text}")


async def get_token_from_gmail():
    """Fetch confirmation token from Gmail via API."""
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
    const parts = full.data.payload.parts || [];
    if (full.data.payload.body && full.data.payload.body.data) {
        body = Buffer.from(full.data.payload.body.data, 'base64').toString('utf-8');
    } else if (parts.length > 0) {
        for (const part of parts) {
            if (part.body && part.body.data) {
                body += Buffer.from(part.body.data, 'base64').toString('utf-8');
            }
        }
    }
    const match = body.match(/token=([^\\s&"<]+)/);
    if (match) console.log('TOKEN=' + match[1]);
    else {
        console.log('NO_TOKEN');
        console.log('BODY=' + body.substring(0, 500));
    }
}
main().catch(e => console.log('ERROR=' + e.message));
'''],
        capture_output=True, text=True, cwd='/Users/austindanson/Desktop/jane-street-dormant'
    )
    output = proc.stdout.strip().split('\n')[-1]
    if output.startswith('TOKEN='):
        return output.replace('TOKEN=', '')
    print(f'  Gmail result: {output}')
    return None


asyncio.run(register_and_confirm())
