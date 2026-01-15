import json
import os
import random
import requests

URL = "http://127.0.0.1:8000/predict"
RESULT_FILE = "logs/simulation_results.jsonl"
NUM_REQUESTS = 5000
IMPOSTOR_RATE = 0.05
TRICKY_RATE = 0.25

if os.path.exists(RESULT_FILE):
    os.remove(RESULT_FILE)
os.makedirs("logs", exist_ok=True)


def generate_random_safe():
    return {
        "data": {
            "number_of_reviews": random.randint(20, 500),
            "listing_price": random.uniform(50, 400),
            "host_is_superhost_int": 1 if random.random() > 0.6 else 0,
            "availability_365": random.randint(0, 200),
            "session_count": random.randint(10, 150),
            "price_vs_neighbourhood": random.uniform(0.01, 1.35),
            "room_type": random.choice(
                ["Entire home/apt", "Private room", "Shared room"]
            ),
        },
        "is_suspicious": 0,
    }


def generate_random_suspicious():
    return {
        "data": {
            "number_of_reviews": random.randint(0, 15),
            "listing_price": random.uniform(500, 12000),
            "host_is_superhost_int": 1 if random.random() > 0.75 else 0,
            "availability_365": random.randint(200, 365),
            "session_count": random.randint(0, 50),
            "price_vs_neighbourhood": random.uniform(2.0, 15.0),
            "room_type": random.choice(
                ["Entire home/apt", "Private room", "Shared room"]
            ),
        },
        "is_suspicious": 1,
    }


def generate_random_tricky():
    return {
        "data": {
            "number_of_reviews": random.randint(6, 100),
            "listing_price": random.uniform(250, 2000),
            "host_is_superhost_int": 1 if random.random() > 0.3 else 0,
            "availability_365": random.randint(50, 300),
            "session_count": random.randint(5, 50),
            "price_vs_neighbourhood": random.uniform(1.5, 4.0),
            "room_type": random.choice(
                ["Entire home/apt", "Private room", "Shared room"]
            ),
        },
        "is_suspicious": 0,
    }


def simulate_traffic():
    print(f"Sending {NUM_REQUESTS} requests to {URL}...")
    print(f"Target suspicious rate: {IMPOSTOR_RATE * 100:.2f}%")
    print(f"Tricky listings rate: {TRICKY_RATE * 100:.2f}%")
    success_count = 0
    stats = {"safe": 0, "sus": 0, "tricky": 0}

    with open(RESULT_FILE, "a") as fh:
        for i in range(NUM_REQUESTS):
            rand = random.random()

            if rand < IMPOSTOR_RATE:
                pack = generate_random_suspicious()
                stats["sus"] += 1
            elif rand < (IMPOSTOR_RATE + TRICKY_RATE):
                pack = generate_random_tricky()
                stats["tricky"] += 1
            else:
                pack = generate_random_safe()
                stats["safe"] += 1

            data = pack["data"]
            ground_truth = pack["is_suspicious"]

            try:
                response = requests.post(URL, json=data)
                if response.status_code == 200:
                    result = response.json()

                    log_entry = {
                        "is_suspicious": ground_truth,
                        "prediction": result["prediction"],
                        "model_used": result["model_used"],
                        "ab_test_group": result["ab_test_group"],
                        "listing_price": data["listing_price"],
                    }
                    fh.write(json.dumps(log_entry) + "\n")
                    success_count += 1
                else:
                    print(f"Request {i} failed Status: {response.status_code}")
            except Exception as e:
                print(f"Request {i} encountered an error: {e}")
                break

    print(f"Successfully sent {success_count}/{NUM_REQUESTS} requests.")


if __name__ == "__main__":
    simulate_traffic()
