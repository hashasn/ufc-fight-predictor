import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
from datetime import datetime

# -----------------------------------
# Basic configuration
# -----------------------------------
HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

OUT_CSV = "ufc_upcoming_event_enriched.csv"
TARGET_EVENT_NAME = "UFC Fight Night: Adesanya vs. Pyfer"
SLEEP_BETWEEN_REQUESTS = 0.75


# -----------------------------------
# Helper: download page and parse HTML
# -----------------------------------
def get_soup(url):
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")


# -----------------------------------
# Helpers: convert height/reach
# -----------------------------------
def height_to_inches(height_str):
    """
    Converts strings like 5' 11" into integer inches.
    Example: 5' 11" -> 71
    """
    if not height_str or pd.isna(height_str):
        return None

    height_str = str(height_str).strip()
    m = re.match(r"(\d+)'\s*(\d+)\"", height_str)
    if not m:
        return None

    feet = int(m.group(1))
    inches = int(m.group(2))
    return feet * 12 + inches


def reach_to_inches(reach_str):
    """
    Converts strings like 70" into float inches.
    Example: 70" -> 70.0
    """
    if not reach_str or pd.isna(reach_str):
        return None

    reach_str = str(reach_str).strip()
    m = re.match(r"(\d+(?:\.\d+)?)\"", reach_str)
    if not m:
        return None

    return float(m.group(1))


# -----------------------------------
# Scrape metadata from fighter profile
# -----------------------------------
def parse_fighter_profile(url):
    soup = get_soup(url)

    ul = soup.find("ul", class_="b-list__box-list")
    if not ul:
        return {
            "height": None,
            "reach": None,
            "stance": None,
            "dob": None
        }

    data = {}

    for li in ul.find_all("li", class_="b-list__box-list-item"):
        text = " ".join(li.get_text(" ", strip=True).split())

        if ":" in text:
            key, val = text.split(":", 1)
            data[key.strip().lower()] = val.strip()

    return {
        "height": height_to_inches(data.get("height")),
        "reach": reach_to_inches(data.get("reach")),
        "stance": data.get("stance"),
        "dob": data.get("dob")
    }


# -----------------------------------
# Get upcoming events
# -----------------------------------
def get_upcoming_events(target_event_name=None):
    url = "http://ufcstats.com/statistics/events/upcoming"
    soup = get_soup(url)

    events = []

    tbody = soup.find("tbody")
    if not tbody:
        return events

    for tr in tbody.find_all("tr"):
        a = tr.find("a", class_="b-link b-link_style_black")
        if not a:
            continue

        name = a.get_text(strip=True)

        span = tr.find("span")
        if not span:
            continue

        date_str = span.get_text(strip=True)

        try:
            event_date = datetime.strptime(date_str, "%B %d, %Y")
        except ValueError:
            event_date = None

        if target_event_name is not None and name != target_event_name:
            continue

        events.append({
            "name": name,
            "date": date_str,
            "date_dt": event_date,
            "url": a.get("href")
        })

    return events


# -----------------------------------
# Extract fight pairings from event page
# -----------------------------------
def get_upcoming_event_fights(event_url):
    soup = get_soup(event_url)
    fights = []

    tbody = soup.find("tbody")
    if not tbody:
        return fights

    for tr in tbody.find_all("tr"):
        fighter_links = tr.find_all("a", class_="b-link b-link_style_black")

        if len(fighter_links) < 2:
            continue

        f1 = fighter_links[0]
        f2 = fighter_links[1]

        f1_name = f1.get_text(strip=True)
        f2_name = f2.get_text(strip=True)
        f1_url = f1.get("href")
        f2_url = f2.get("href")

        if not f1_name or not f2_name:
            continue

        fights.append({
           
            "fighter1": f1_name,
            "fighter2": f2_name,
            "f1_url": f1_url,
            "f2_url": f2_url
        })

    return fights


# -----------------------------------
# Main pipeline
# -----------------------------------
def scrape_upcoming_event(target_event_name, out_csv):
    events = get_upcoming_events(target_event_name=target_event_name)

    if not events:
        print(f"No upcoming event found with name: {target_event_name}")
        return pd.DataFrame()

    print(f"Found {len(events)} matching event(s)")

    fighter_profile_cache = {}
    rows = []

    for event in events:
        print(f"\nScraping event: {event['name']} | {event['date']}")

        fights = get_upcoming_event_fights(event["url"])
        print(f"  Fight count: {len(fights)}")

        for i, fight in enumerate(fights, start=1):
            try:
                print(f"    [{i}/{len(fights)}] {fight['fighter1']} vs {fight['fighter2']}")

                f1_url = fight["f1_url"]
                f2_url = fight["f2_url"]

                if f1_url and f1_url not in fighter_profile_cache:
                    fighter_profile_cache[f1_url] = parse_fighter_profile(f1_url)
                    time.sleep(SLEEP_BETWEEN_REQUESTS)

                if f2_url and f2_url not in fighter_profile_cache:
                    fighter_profile_cache[f2_url] = parse_fighter_profile(f2_url)
                    time.sleep(SLEEP_BETWEEN_REQUESTS)

                f1_meta = fighter_profile_cache.get(f1_url, {})
                f2_meta = fighter_profile_cache.get(f2_url, {})

                rows.append({
                    "event": event["name"],
                    "fighter1": fight["fighter1"],
                    "fighter2": fight["fighter2"],
                    "event_date": event["date"],
                    "f1_dob": f1_meta.get("dob"),
                    "f1_height": f1_meta.get("height"),
                    "f1_reach": f1_meta.get("reach"),
                    "f1_stance": f1_meta.get("stance"),
                    "f2_dob": f2_meta.get("dob"),
                    "f2_height": f2_meta.get("height"),
                    "f2_reach": f2_meta.get("reach"),
                    "f2_stance": f2_meta.get("stance"),
                })

                time.sleep(SLEEP_BETWEEN_REQUESTS)

            except Exception as e:
                print(f"    Error scraping fight {fight['fighter1']} vs {fight['fighter2']}")
                print(f"    {e}")

    df = pd.DataFrame(rows)

    ordered_cols = [
        "event",
        "fighter1",
        "fighter2",
        "event_date",
        "f1_dob",
        "f1_height",
        "f1_reach",
        "f1_stance",
        "f2_dob",
        "f2_height",
        "f2_reach",
        "f2_stance",
    ] 

    if not df.empty:
        df = df[ordered_cols]

    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"\nSaved {len(df)} rows to {out_csv}")

    return df


# -----------------------------------
# Run
# -----------------------------------
if __name__ == "__main__":
    df_upcoming = scrape_upcoming_event(
        target_event_name=TARGET_EVENT_NAME,
        out_csv=OUT_CSV
    )

    print(df_upcoming.head())
    print(df_upcoming.shape)