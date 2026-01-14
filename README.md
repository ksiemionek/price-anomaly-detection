# IUM - Dokumentacja projektu (Etap 2)

### Kacper Siemionek, Michał Pędziwiatr

## Instrukcja uruchomienia

### 1. Wymagania

Projekt wymaga środowiska Python 3.x+ oraz instalancji zależności. Zalecane użycie środowiska wirtualnego:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Uruchomienie mikroserwisu

Serwer API nasłuchuje na porcie 8000. Należy go uruchomić w osobnym terminalu:

```bash
uvicorn app:app --reload
```

### 3. Symulacja ruchu

Aby przetestować działanie systemu, należy uruchomić symulator. Skrypt ten wysyła serię zapytań do API, wstrzykując określony procent ofert anomalnych w celu weryfikacji modeli:

```
python3 simulate_traffic.py
```

### 4. Ewaluacja wyników

Po zebraniu logów z symulacji, skrypt przetwarza plik prediction_logs.jsonl, a następnie wylicza metryki jakości dla obu grup testowych (A i B):

```bash
python3 evaluate_logs.py
```

## Opis API

### Endpoint `POST /predict`

Główny punkt wejścia do systemu. Przyjmuje dane o ofercie, a następnie zwraca ocenę (0 - uczciwa, 1 - podejrzana) oraz użyty podczas ewaluacji model. Wybór modelu (bazowy vs docelowy) odbywa się automatycznie po stronie serwera.

**_Struktura danych wejściowych (JSON)_**:

- `number_of_reviews (int)` - liczba recenzji oferty
- `review_scores_rating (float)` - średnia ocena (0-100)
- `host_is_superhost_int (int)` - zmienna oznaczająca czy gospodarz jest superhostem (0-1)
- `availability_365 (int)` - liczba dostępnych dni w roku
- `avg_price_calendar (float)` - średnia cena
- `max_price_calendar (float)` - maksymalna cena
- `std_price_calendar (float)` - odchylenie standardowe ceny
- `session_count (int)` - liczba rezerwacji oferty
- `price_vs_neighbourhood (float)` - proporcjonalny stosunek ceny do średniej dzielnicy
- `room_type (str)` - typ pokoju ("Entire home/apt" / "Private room" / "Shared room")

**_Przykładowe użycie_**:

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "number_of_reviews": 12,
  "review_scores_rating": 95.0,
  "host_is_superhost_int": 1,
  "availability_365": 150,
  "avg_price_calendar": 120.5,
  "max_price_calendar": 150.0,
  "std_price_calendar": 15.2,
  "session_count": 45,
  "price_vs_neighbourhood": 1.1,
  "room_type": "Entire home/apt"
}'
```

**_Przykładowa odpowiedź serwera_**:

```json
{
  "prediction": 0,
  "model_used": "Target Model",
  "ab_test_group": "B"
}
```
