"""
UFC Fight Predictor - interactive CLI.

Run from this directory:  python cli.py

Menu:
  1) Predict fight outcome      - enter two fighter names
  2) Display future predictions - scrape & predict the next two cards
  3) Retrain model              - fetch new results, append, retrain
  0) Exit
"""

import warnings

import service

warnings.filterwarnings("ignore")

MENU = """
==============================
   UFC Fight Predictor
==============================
1) Predict fight outcome
2) Display future predictions (next 2 cards)
3) Retrain model
0) Exit
"""


def _prompt(text):
    try:
        return input(text).strip()
    except EOFError:
        return ""


# =========================================================
# Option 1: predict a single matchup
# =========================================================

def option_predict_fight():
    name_a = _prompt("Fighter A (exact name): ")
    name_b = _prompt("Fighter B (exact name): ")
    if not name_a or not name_b:
        print("Both names are required.\n")
        return

    date = _prompt("Event date [YYYY-MM-DD, blank = today]: ") or None

    try:
        res = service.predict_two(name_a, name_b, event_date=date)
    except ValueError as e:
        print(f"\n{e}\n")
        return

    pa = res["prob_fighter_a_wins"] * 100
    pb = res["prob_fighter_b_wins"] * 100
    print("\n--------------------------------")
    print(f"  {res['fighter_a']:<22} {pa:5.1f}%")
    print(f"  {res['fighter_b']:<22} {pb:5.1f}%")
    print(f"  Predicted winner: {res['predicted_winner']}")
    print("--------------------------------\n")


# =========================================================
# Option 2: display future predictions
# =========================================================

def _print_card(df):
    for event, group in df.groupby("event", sort=False):
        date = group["event_date"].iloc[0] if "event_date" in group else ""
        print(f"\n=== {event} | {date} ===")
        for _, r in group.iterrows():
            print(f"  {r['fight']:<42} -> {r['predicted_winner']:<22} {r['confidence']}")


def option_future_predictions():
    refresh = _prompt("Force fresh scrape? [y/N]: ").lower() == "y"
    try:
        df = service.predict_upcoming(n=2, force_refresh=refresh)
    except Exception as e:  # noqa: BLE001
        print(f"\nCould not fetch/predict upcoming cards: {e}\n")
        return

    if df is None or df.empty:
        print("\nNo upcoming fights found.\n")
        return

    _print_card(df)
    print()


# =========================================================
# Option 3: retrain
# =========================================================

def option_retrain():
    print(
        "\nThis scrapes newly completed events, appends them to the dataset,\n"
        "and retrains the model (overwrites artifacts/; a backup is made first).\n"
        "It can take several minutes."
    )
    if _prompt("Proceed? [y/N]: ").lower() != "y":
        print("Cancelled.\n")
        return

    try:
        summary = service.refresh_and_retrain(max_new_events=3)
    except Exception as e:  # noqa: BLE001
        print(f"\nRetrain failed: {e}\n")
        return

    print("\n--------------------------------")
    print(f"  New rows added : {summary['added']}")
    print(f"  Dataset size   : {summary['rows']}")
    print(f"  Retrained      : {summary['retrained']}")
    print("--------------------------------\n")


# =========================================================
# Main loop
# =========================================================

ACTIONS = {
    "1": option_predict_fight,
    "2": option_future_predictions,
    "3": option_retrain,
}


def main():
    while True:
        print(MENU)
        choice = _prompt("Select an option: ")
        if choice == "0":
            print("Goodbye.")
            break
        action = ACTIONS.get(choice)
        if action is None:
            print("Invalid option.\n")
            continue
        action()


if __name__ == "__main__":
    main()
