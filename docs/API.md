# API Documentation — Rail ETA v3.0

Rail ETA exposes a production-grade REST API built with **FastAPI**. The service
predicts train journey duration from route distance and station count.

## Base URL

- Local development: `http://localhost:8000`
- Docker: `http://localhost:8000` (port mapped via `docker-compose.yml`)
- Interactive docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Authentication

The API currently does not require authentication for local deployments. For
production, place an authentication layer (e.g., OAuth2, API keys, or a
reverse-proxy middleware) in front of the service.

## Endpoints

### `GET /health`

Health check. Returns service status and version.

**Request**

```http
GET /health
```

**Response**

```json
{
  "status": "ok",
  "service": "rail-eta",
  "version": "3.0.0"
}
```

---

### `POST /predict`

Predict journey duration for a single train route.

**Request**

```http
POST /predict
Content-Type: application/json

{
  "distance": 500.0,
  "stations": 10
}
```

| Field | Type | Required | Description | Example |
|---|---|---|---|---|
| `distance` | float | Yes | Total route distance in kilometres. Must be > 0. | `500.0` |
| `stations` | int | Yes | Number of stations/stops on the route. Must be > 0. | `10` |

**Response (200 OK)**

```json
{
  "predicted_duration_hours": 9.14,
  "predicted_duration_human": "9.14 hour(s)"
}
```

| Field | Type | Description |
|---|---|---|
| `predicted_duration_hours` | float | Predicted duration in hours (rounded to 4 decimals). |
| `predicted_duration_human` | str | Human-readable duration (e.g., `"1 day(s) 13.06 hour(s)"`). |

**Example — cURL**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"distance": 2000, "stations": 30}'
```

**Response**

```json
{
  "predicted_duration_hours": 37.06,
  "predicted_duration_human": "1 day(s) 13.06 hour(s)"
}
```

**Error Responses**

| Status | Meaning |
|---|---|
| `422 Unprocessable Entity` | Invalid request body (missing fields, wrong types, or validation constraints violated). |
| `500 Internal Server Error` | Model not loaded, or prediction failed. Train the model first via `make train` or `python -m pipeline`. |

---

## Running the API

```bash
# Using uvicorn directly (requires src/ on PYTHONPATH)
PYTHONPATH=src uvicorn api_main:app --host 0.0.0.0 --port 8000 --reload

# Using Docker
docker compose up --build api
```

## Error Handling

All unexpected exceptions are caught and returned as `500` with a JSON body
containing the error message. Validation errors from Pydantic are returned
automatically by FastAPI as `422` with field-level detail.

## Rate Limiting

No rate limiting is enabled by default. For production, deploy behind a
reverse proxy (NGINX, Traefik, or Kong) and configure rate limits per client
IP or API key.
