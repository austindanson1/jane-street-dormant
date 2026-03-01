const { google } = require('/Users/austindanson/Desktop/dansonclaw/node_modules/googleapis');
require('/Users/austindanson/Desktop/dansonclaw/node_modules/dotenv').config({ path: '/Users/austindanson/Desktop/dansonclaw/.env' });

async function main() {
    const auth = new google.auth.OAuth2(process.env.GMAIL_CLIENT_ID, process.env.GMAIL_CLIENT_SECRET);
    auth.setCredentials({ refresh_token: process.env.GMAIL_REFRESH_TOKEN });
    const gmail = google.gmail({ version: 'v1', auth });

    // Search for ANY non-confirmation email from dormant-puzzle
    const queries = [
        'from:dormant-puzzle.janestreet.com -subject:"email confirmation"',
        'from:janestreet.com austindansonart',
        'from:dormant-puzzle.janestreet.com subject:welcome',
        'from:dormant-puzzle.janestreet.com subject:key',
    ];

    for (const q of queries) {
        const res = await gmail.users.messages.list({ userId: 'me', q, maxResults: 5 });
        if (!res.data.messages || res.data.messages.length === 0) {
            console.log(`Query: "${q}" -> No results`);
            continue;
        }
        console.log(`Query: "${q}" -> ${res.data.messages.length} results`);
        for (const msg of res.data.messages) {
            const full = await gmail.users.messages.get({ userId: 'me', id: msg.id, format: 'full' });
            const subject = full.data.payload.headers.find(h => h.name === 'Subject')?.value;
            const date = full.data.payload.headers.find(h => h.name === 'Date')?.value;
            const to = full.data.payload.headers.find(h => h.name === 'To')?.value;

            let body = '';
            if (full.data.payload.body && full.data.payload.body.data) {
                body = Buffer.from(full.data.payload.body.data, 'base64').toString('utf-8');
            } else if (full.data.payload.parts) {
                for (const part of full.data.payload.parts) {
                    if (part.body && part.body.data) {
                        body = Buffer.from(part.body.data, 'base64').toString('utf-8');
                        if (part.mimeType === 'text/plain') break;
                    }
                }
            }
            console.log(`  Date: ${date} | To: ${to} | Subject: ${subject}`);
            console.log(`  Body: ${body.substring(0, 500)}`);
            console.log('  ---');
        }
    }
}

main().catch(console.error);
