import random

import requests

from utilities import spank

URL = "http://127.0.0.1:8000/predict"
NUM_REQUESTS = 1000
IMPOSTOR_RATE = 0.1


def generate_random_safe():
    avg_price = random.uniform(50, 300)
    max_price = avg_price + random.uniform(0, 100)

    return {
        "number_of_reviews": random.randint(20, 500),
        "review_scores_rating": random.uniform(70, 100),
        "host_is_superhost_int": 1 if random.random() > 0.6 else 0,
        "availability_365": random.randint(0, 200),
        "avg_price_calendar": avg_price,
        "max_price_calendar": max_price,
        "std_price_calendar": random.uniform(0, 50),
        "session_count": random.randint(10, 150),
        "price_vs_neighbourhood": random.uniform(0.75, 1.35),
        "room_type": random.choice(["Entire home/apt", "Private room", "Shared room"]),
    }


def generate_random_suspicious():
    avg_price = random.uniform(500, 5000)
    max_price = avg_price + random.uniform(100, 2000)

    return {
        "number_of_reviews": random.randint(0, 5),
        "review_scores_rating": random.uniform(0, 70),
        "host_is_superhost_int": 0,
        "availability_365": random.randint(200, 365),
        "avg_price_calendar": avg_price,
        "max_price_calendar": max_price,
        "std_price_calendar": random.uniform(0, 50),
        "session_count": random.randint(0, 10),
        "price_vs_neighbourhood": random.uniform(2, 10.0),
        "room_type": random.choice(["Entire home/apt", "Private room", "Shared room"]),
    }


def simulate_traffic():
    print(f"Sending {NUM_REQUESTS} requests to {URL}...")
    print(f"Target suspicious rate: {IMPOSTOR_RATE * 100:.2f}%")
    success_count = 0
    sus_count = 0

    for i in range(NUM_REQUESTS):
        if random.random() < IMPOSTOR_RATE:
            data = generate_random_suspicious()
            is_sus = True
        else:
            data = generate_random_safe()
            is_sus = False

        try:
            response = requests.post(URL, json=data)
            if response.status_code == 200:
                success_count += 1
                if is_sus:
                    sus_count += 1
            else:
                print(f"Request {i} failed 🥺 Status: {response.status_code}")
        except Exception as e:
            print(f"Request {i} encountered an error: {e} 😭😭😭")
            for i in range(2):
                spank()
            break

    emote = "😍" if success_count == NUM_REQUESTS else "😭"
    print(f"Successfully sent {success_count}/{NUM_REQUESTS} requests. {emote}")


if __name__ == "__main__":
    simulate_traffic()
