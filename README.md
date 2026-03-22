# Price Anomaly Detection

Price anomaly detection for rental listings, served via a FastAPI microservice with A/B testing between two models.

## How it works
 
The system exposes a `/predict` endpoint that classifies rental offers as legitimate (0) or suspicious (1). Incoming requests are automatically split between a baseline model (group A) and a target model (group B) for live A/B comparison.

## Setup
 
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running
 
```bash
# Start the API server
uvicorn app:app --reload
 
# Simulate traffic with injected anomalies
python3 simulate_traffic.py
 
# Evaluate A/B results from logs/simulation_results.jsonl
python3 evaluate_logs.py
```

## API
 
| Field | Type | Description |
|---|---|---|
| `number_of_reviews` | int | Number of listing reviews |
| `listing_price` | float | Current listing price |
| `host_is_superhost_int` | int | Superhost flag (0 or 1) |
| `availability_365` | int | Days available per year |
| `session_count` | int | Number of bookings |
| `price_vs_neighbourhood` | float | Price relative to neighbourhood average |
| `room_type` | str | `"Entire home/apt"` / `"Private room"` / `"Shared room"` |

**Request**:

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "number_of_reviews": 0,
  "listing_price": 0,
  "price_vs_neighbourhood": 0,
  "host_is_superhost_int": 0,
  "availability_365": 0,
  "session_count": 0,
  "room_type": "Entire home/apt"
}'
```
 
**Response:**
```json
{
  "prediction": 0,
  "model_used": "Target Model",
  "ab_test_group": "B"
}
```

## Documentation
 
Full project reports (in Polish) are available in the repository root:
- `IUM - Dokumentacja wstępna.pdf` - initial specification
- `IUM dokumentacja koncowa.pdf` - final report
