const { google } = require("/Users/austindanson/Desktop/dansonclaw/node_modules/googleapis");
require("/Users/austindanson/Desktop/dansonclaw/node_modules/dotenv").config({ path: "/Users/austindanson/Desktop/dansonclaw/.env" });

async function main() {
    const auth = new google.auth.OAuth2(process.env.GMAIL_CLIENT_ID, process.env.GMAIL_CLIENT_SECRET);
    auth.setCredentials({ refresh_token: process.env.GMAIL_REFRESH_TOKEN });
    const gmail = google.gmail({ version: "v1", auth });
    const res = await gmail.users.messages.list({
        userId: "me",
        q: "from:no-reply@dormant-puzzle.janestreet.com newer_than:5m",
        maxResults: 5,
    });
    if (!res.data.messages) { console.log("NO_EMAIL"); return; }
    console.log("Found " + res.data.messages.length + " messages");
    for (const msg of res.data.messages) {
        const full = await gmail.users.messages.get({ userId: "me", id: msg.id, format: "full" });
        const headers = full.data.payload.headers || [];
        const dateHeader = headers.find(h => h.name === "Date");
        const toHeader = headers.find(h => h.name === "To");
        let body = "";
        if (full.data.payload.body && full.data.payload.body.data)
            body = Buffer.from(full.data.payload.body.data, "base64").toString("utf-8");
        else if (full.data.payload.parts) {
            for (const part of full.data.payload.parts) {
                if (part.body && part.body.data) {
                    body += Buffer.from(part.body.data, "base64").toString("utf-8");
                }
            }
        }
        const match = body.match(/token=([^\s"<>&]+)/);
        console.log("To: " + (toHeader ? toHeader.value : "?") + " | Date: " + (dateHeader ? dateHeader.value : "?"));
        if (match) {
            console.log("TOKEN=" + match[1]);
        }
    }
}
main().catch(e => console.log("ERROR=" + e.message));
