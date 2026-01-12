import random

import requests

URL = "http://127.0.0.1:8000/predict"
NUM_REQUESTS = 1000


def generate_random_listing():
    return {
        "number_of_reviews": random.randint(0, 500),
        "review_scores_rating": random.uniform(0, 100),
        "host_is_superhost_int": random.randint(0, 1),
        "availability_365": random.randint(0, 365),
        "avg_price_calendar": random.uniform(50, 1000),
        "std_price_calendar": random.uniform(0, 300),
        "session_count": random.randint(0, 100),
        "price_vs_neighbourhood": random.uniform(0.3, 3.0),
        "room_type": random.choice(["Entire home/apt", "Private room", "Shared room"]),
    }


def simulate_traffic():
    print(f"Sending {NUM_REQUESTS} requests to {URL}...")
    success_count = 0

    for i in range(NUM_REQUESTS):
        data = generate_random_listing()
        try:
            response = requests.post(URL, json=data)
            if response.status_code == 200:
                success_count += 1
            else:
                print(f"Request {i} failed 🥺 Status: {response.status_code}")
        except Exception as e:
            print(f"Request {i} encountered an error: {e} 😭😭😭")
            break

    emote = "😍" if success_count == NUM_REQUESTS else "😭"
    print(f"Successfully sent {success_count}/{NUM_REQUESTS} requests. {emote}")


if __name__ == "__main__":
    simulate_traffic()
