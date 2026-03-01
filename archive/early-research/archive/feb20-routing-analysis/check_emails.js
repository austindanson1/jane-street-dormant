const { google } = require('/Users/austindanson/Desktop/dansonclaw/node_modules/googleapis');
require('/Users/austindanson/Desktop/dansonclaw/node_modules/dotenv').config({ path: '/Users/austindanson/Desktop/dansonclaw/.env' });

async function main() {
    const auth = new google.auth.OAuth2(process.env.GMAIL_CLIENT_ID, process.env.GMAIL_CLIENT_SECRET);
    auth.setCredentials({ refresh_token: process.env.GMAIL_REFRESH_TOKEN });
    const gmail = google.gmail({ version: 'v1', auth });
    const res = await gmail.users.messages.list({
        userId: 'me',
        q: 'from:no-reply@dormant-puzzle.janestreet.com newer_than:1h',
        maxResults: 10,
    });
    if (!res.data.messages) { console.log('NO_EMAILS'); return; }
    console.log('Found ' + res.data.messages.length + ' emails');
    for (const msg of res.data.messages) {
        const full = await gmail.users.messages.get({ userId: 'me', id: msg.id, format: 'full' });
        const subject = full.data.payload.headers.find(h => h.name === 'Subject')?.value || 'no subject';
        const date = full.data.payload.headers.find(h => h.name === 'Date')?.value || 'no date';
        console.log('\n=== ' + subject + ' (' + date + ') ===');
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
        console.log(body.substring(0, 2000));

        // Look for API key pattern
        const keyMatch = body.match(/[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}/);
        if (keyMatch) {
            console.log('\n*** API KEY FOUND: ' + keyMatch[0] + ' ***');
        }
    }
}
main().catch(e => console.log('ERROR: ' + e.message));
