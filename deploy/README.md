---
title: ShieldTab Backend
emoji: 🛡️
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# ShieldTab Backend

Real-time malicious URL detection API (hybrid XGBoost + CNN ensemble)
powering the ShieldTab Chrome extension.

## Endpoints
- `GET /health` — check the service and confirm models are loaded
- `POST /predict` — `{"url": "..."}` → malicious/benign verdict + confidence + threat type

Once deployed, your public URL will look like:
`https://<your-username>-shieldtab-backend.hf.space`

Update `BACKEND_URL` in the Chrome extension's `config.js` to point here.
