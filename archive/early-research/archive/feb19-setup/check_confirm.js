const { google } = require('/Users/austindanson/Desktop/dansonclaw/node_modules/googleapis');
require('/Users/austindanson/Desktop/dansonclaw/node_modules/dotenv').config({ path: '/Users/austindanson/Desktop/dansonclaw/.env' });

async function main() {
    const auth = new google.auth.OAuth2(process.env.GMAIL_CLIENT_ID, process.env.GMAIL_CLIENT_SECRET);
    auth.setCredentials({ refresh_token: process.env.GMAIL_REFRESH_TOKEN });
    const gmail = google.gmail({ version: 'v1', auth });

    // First try specific search
    let res = await gmail.users.messages.list({
        userId: 'me',
        q: 'newer_than:30m (confirm OR dormant OR jane OR noreply)',
        maxResults: 10,
    });

    if (!res.data.messages || res.data.messages.length === 0) {
        console.log('No matching emails. Checking all recent emails...');
        res = await gmail.users.messages.list({
            userId: 'me',
            q: 'newer_than:30m',
            maxResults: 5,
        });
        if (!res.data.messages || res.data.messages.length === 0) {
            console.log('No emails in last 30 min at all.');
            return;
        }
    }

    for (const msg of res.data.messages) {
        const full = await gmail.users.messages.get({ userId: 'me', id: msg.id, format: 'full' });
        const subject = full.data.payload.headers.find(h => h.name === 'Subject')?.value;
        const from = full.data.payload.headers.find(h => h.name === 'From')?.value;
        const to = full.data.payload.headers.find(h => h.name === 'To')?.value;

        let body = '';
        if (full.data.payload.body?.data) {
            body = Buffer.from(full.data.payload.body.data, 'base64').toString('utf-8');
        } else if (full.data.payload.parts) {
            for (const part of full.data.payload.parts) {
                if (part.body?.data && part.mimeType === 'text/plain') {
                    body = Buffer.from(part.body.data, 'base64').toString('utf-8');
                    break;
                }
                if (part.body?.data && part.mimeType === 'text/html') {
                    body = Buffer.from(part.body.data, 'base64').toString('utf-8');
                }
            }
        }

        console.log('Subject:', subject);
        console.log('From:', from);
        console.log('To:', to);
        console.log('Body:', body.substring(0, 3000));
        console.log('---');
    }
}

main().catch(console.error);
