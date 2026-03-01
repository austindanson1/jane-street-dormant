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
            const token = match[1];
            try {
                const payload = JSON.parse(Buffer.from(token.split('.')[0], 'base64').toString());
                console.log('TOKEN=' + token);
                console.log('SUB=' + payload.sub);
                console.log('EXP=' + payload.exp);
                return;
            } catch(e) {
                console.log('TOKEN=' + token);
                return;
            }
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
    output = proc.stdout.strip()
    for line in output.split('\n'):
        if line.startswith('TOKEN='):
            return line[6:]
    print(f'  Gmail output: {output}')
    return None


async def main():
    async with aiohttp.ClientSession() as session:
        # Step 1: Register
        print('Step 1: Registering hello@faeonline.com...')
        async with session.post(
            f'{URL}/api/partners/jane-street/users',
            headers={'Authorization': f'Bearer {PARTNER_KEY}'},
            json={'email': EMAIL},
        ) as resp:
            result = await resp.text()
            print(f'  Status: {resp.status}')
            print(f'  Result: {result}')

        # Step 2: Wait for email
        print('\nStep 2: Waiting 20s for forwarded email...')
        await asyncio.sleep(20)

        # Step 3: Get token
        print('Step 3: Fetching token from Gmail...')
        token = None
        for attempt in range(4):
            token = get_token_from_gmail()
            if token:
                # Decode to verify it's for the right email
                try:
                    payload_b64 = token.split('.')[0]
                    # Add padding
                    padding = 4 - len(payload_b64) % 4
                    if padding != 4:
                        payload_b64 += '=' * padding
                    payload = json.loads(base64.b64decode(payload_b64))
                    print(f'  Token subject: {payload.get("sub")}')
                    print(f'  Token expires: {payload.get("exp")}')
                except Exception as e:
                    print(f'  Could not decode token payload: {e}')
                break
            print(f'  Attempt {attempt+1}: No token yet. Waiting 15s...')
            await asyncio.sleep(15)

        if not token:
            print('FAILED: Could not get confirmation token from email.')
            return

        print(f'  Token: {token[:70]}...')

        # Step 4: Confirm
        print('\nStep 4: Confirming via server action...')
        headers = {
            'Content-Type': 'text/plain;charset=UTF-8',
            'Next-Action': ACTION_ID,
            'Accept': 'text/x-component',
        }
        body = f'["{token}"]'

        async with session.post(
            f'{URL}/jane-street/confirm',
            data=body,
            headers=headers,
        ) as resp:
            text = await resp.text()
            print(f'  Status: {resp.status}')
            print(f'  Response: {text[:500]}')

            key_match = re.search(
                r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}',
                text
            )
            if key_match:
                api_key = key_match.group(0)
                print(f'\n*** NEW API KEY: {api_key} ***')
            elif 'success' in text and 'true' in text:
                print('\nConfirmation succeeded but no API key in response.')
                print('Check email for API key or try logging in.')
            else:
                print('\nConfirmation may have failed. Full response above.')


asyncio.run(main())
