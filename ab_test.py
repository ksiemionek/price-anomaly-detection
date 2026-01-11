from sklearn.model_selection import train_test_split

import utilities as siemionek_kacper


def run_ab_tests():
    print("=" * 10 + "[A/B TESTING]" + "=" * 10)
    listings, calendar, sessions = siemionek_kacper.load_data()
    df = siemionek_kacper.prepare_data(listings, calendar, sessions)

    df = siemionek_kacper.create_labels_advanced(df)

    group_a, group_b = train_test_split(df, test_size=0.5, random_state=67)
    print(
        f"Group A (control group) length: {len(group_a)}\n Group B (experiment group) length: {len(group_a)}"
    )


if __name__ == "__main__":
    run_ab_tests()
