import json
import os
import random
import requests
import utilities

URL = "http://127.0.0.1:8000/predict"
RESULT_FILE = "logs/simulation_results.jsonl"
NUM_REQUESTS = 10000
IMPOSTOR_RATE = 0.05
TRICKY_RATE = 0.25

if os.path.exists(RESULT_FILE):
    os.remove(RESULT_FILE)
os.makedirs("logs", exist_ok=True)


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
                pack = utilities.generate_random_suspicious()
                stats["sus"] += 1
            elif rand < (IMPOSTOR_RATE + TRICKY_RATE):
                pack = utilities.generate_random_tricky()
                stats["tricky"] += 1
            else:
                pack = utilities.generate_random_safe()
                stats["safe"] += 1

            ground_truth = pack.pop("is_suspicious")

            try:
                response = requests.post(URL, json=pack)
                if response.status_code == 200:
                    result = response.json()

                    log_entry = {
                        "is_suspicious": ground_truth,
                        "prediction": result["prediction"],
                        "model_used": result["model_used"],
                        "ab_test_group": result["ab_test_group"],
                        "listing_price": pack["listing_price"],
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
