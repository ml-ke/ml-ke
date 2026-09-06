# Answer Bot API Discovery — Zendesk Bug Bounty

## Endpoint
`POST /api/v2/answer_bot/answers/articles`

## Authentication
- Requires Basic Auth or cookie-based session (returns 401 without auth)
- Works with standard Zendesk API token

## Correct Request Format
```json
{
  "enquiry": "gift card",
  "reference": "test-123",
  "locale": "en-us",
  "labels": ["test"]
}
```

## Response
```json
{
  "id": 27861412482460,
  "interaction_access_token": "eyJhbG...signed-jwt",
  "auth_token": "eyJhbG...signed-jwt",
  "articles": []
}
```

## JWT Token Decode (interaction_access_token payload)
```json
{
  "account_id": 26569603,
  "brand_id": 27859877743004,
  "deflection_id": 27861412482460,
  "article_ids": [],
  "ticket_id": null,
  "exp": 1782933838
}
```

## Important Notes
- Returns empty `articles` array until Help Center has indexed/published content
- The `interaction_access_token` is a signed JWT — cannot be forged without the signing key
- Used for resolution/rejection feedback endpoints: `/api/v2/answer_bot/resolution` and `/api/v2/answer_bot/rejection` (both return HTTP 403)
- Detected via response header: `zendesk-service: answer-bot-service`
- The Answer Bot is a separate microservice from the main Support API

## Detection Pattern
```bash
# Send any POST to probe if endpoint is alive
curl -sv https://mlke.zendesk.com/api/v2/answer_bot/answers/articles \
  -X POST -H "Content-Type: application/json" -d '{}' \
  -u "email/token:TOKEN" 2>&1 | grep -i 'zendesk-service\|x-envoy'
```

## References
- Zendesk API docs: https://developer.zendesk.com/api-reference/answer-bot/answer-bot-api/article_recommendations/
- Tested on: mlke.zendesk.com (Professional trial, June 2026)
