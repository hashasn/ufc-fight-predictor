"""
Upcoming-card scraper for ufcstats.com.

Generalizes future_card_prediction/future_fight.py to the next N events (no
hardcoded event name) and emits the "fightcard" schema that predict.py ingests
directly (Fighter1/Fighter2 + raw profile fields). Results are cached to disk so
repeated menu runs don't re-scrape within the TTL window.
"""

import os
import json
import time
from datetime import datetime, timezone

import pandas as pd

import config
from http_client import get_soup
from scrape_results import parse_fighter_profile

UPCOMING_EVENTS_URL = "http://ufcstats.com/statistics/events/upcoming"
SLEEP_BETWEEN_REQUESTS = 0.75

# Columns predict.detect_csv_format() recognizes as a full fight card.
FIGHTCARD_COLUMNS = [
    "event", "event_date",
    "Fighter1", "Fighter2",
    "f1_dob", "f1_height", "f1_reach", "f1_stance",
    "f2_dob", "f2_height", "f2_reach", "f2_stance",
]


# =========================================================
# Scraping
# =========================================================

def get_upcoming_events():
    """Return upcoming events (soonest first) as {name, date, date_dt, url}."""
    soup = get_soup(UPCOMING_EVENTS_URL)
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
            date_dt = datetime.strptime(date_str, "%B %d, %Y")
        except ValueError:
            date_dt = None

        events.append({
            "name": name, "date": date_str,
            "date_dt": date_dt, "url": a.get("href"),
        })

    events.sort(key=lambda e: (e["date_dt"] is None, e["date_dt"] or datetime.max))
    return events


def get_event_fights(event_url):
    """Return fight pairings for one upcoming event page."""
    soup = get_soup(event_url)
    fights = []

    tbody = soup.find("tbody")
    if not tbody:
        return fights

    for tr in tbody.find_all("tr"):
        links = tr.find_all("a", class_="b-link b-link_style_black")
        if len(links) < 2:
            continue
        f1, f2 = links[0], links[1]
        if not f1.get_text(strip=True) or not f2.get_text(strip=True):
            continue
        fights.append({
            "fighter1": f1.get_text(strip=True),
            "fighter2": f2.get_text(strip=True),
            "f1_url": f1.get("href"),
            "f2_url": f2.get("href"),
        })

    return fights


def scrape_cards(n=2, progress=print):
    """Scrape the next `n` upcoming cards into a fightcard-schema DataFrame."""
    events = get_upcoming_events()[:n]
    if not events:
        progress("No upcoming events found.")
        return pd.DataFrame(columns=FIGHTCARD_COLUMNS)

    cache = {}
    rows = []
    for event in events:
        progress(f"Scraping upcoming: {event['name']} | {event['date']}")
        try:
            fights = get_event_fights(event["url"])
        except Exception as e:
            progress(f"  Could not load event page: {e}")
            continue
        progress(f"  {len(fights)} fights")

        for fight in fights:
            for u in (fight["f1_url"], fight["f2_url"]):
                if u and u not in cache:
                    try:
                        cache[u] = parse_fighter_profile(u)
                    except Exception as e:
                        progress(f"  profile failed ({u}): {e}")
                        cache[u] = {}
                    time.sleep(SLEEP_BETWEEN_REQUESTS)

            m1 = cache.get(fight["f1_url"], {})
            m2 = cache.get(fight["f2_url"], {})
            rows.append({
                "event": event["name"], "event_date": event["date"],
                "Fighter1": fight["fighter1"], "Fighter2": fight["fighter2"],
                "f1_dob": m1.get("dob"), "f1_height": m1.get("height"),
                "f1_reach": m1.get("reach"), "f1_stance": m1.get("stance"),
                "f2_dob": m2.get("dob"), "f2_height": m2.get("height"),
                "f2_reach": m2.get("reach"), "f2_stance": m2.get("stance"),
            })
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.reindex(columns=FIGHTCARD_COLUMNS)
    return df


# =========================================================
# Caching
# =========================================================

def _cache_age_hours():
    if not os.path.exists(config.UPCOMING_CACHE_META):
        return None
    with open(config.UPCOMING_CACHE_META, "r") as f:
        meta = json.load(f)
    scraped_at = datetime.fromisoformat(meta["scraped_at"])
    delta = datetime.now(timezone.utc) - scraped_at
    return delta.total_seconds() / 3600.0


def _save_cache(df):
    config.ensure_dirs()
    df.to_csv(config.UPCOMING_CACHE_CSV, index=False)
    meta = {
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "events": sorted(df["event"].dropna().unique().tolist()) if not df.empty else [],
        "n_fights": int(len(df)),
    }
    with open(config.UPCOMING_CACHE_META, "w") as f:
        json.dump(meta, f, indent=2)


def get_next_cards(n=2, force_refresh=False, ttl_hours=None, progress=print):
    """
    Return the next `n` upcoming cards as a fightcard DataFrame, using the
    on-disk cache when it is younger than `ttl_hours` (default from config).
    """
    ttl_hours = config.UPCOMING_CACHE_TTL_HOURS if ttl_hours is None else ttl_hours
    age = _cache_age_hours()

    if (not force_refresh and age is not None and age <= ttl_hours
            and os.path.exists(config.UPCOMING_CACHE_CSV)):
        progress(f"Using cached upcoming cards ({age:.1f}h old).")
        return pd.read_csv(config.UPCOMING_CACHE_CSV)

    progress("Scraping upcoming cards (live)...")
    df = scrape_cards(n=n, progress=progress)
    if not df.empty:
        _save_cache(df)
    return df


if __name__ == "__main__":
    cards = get_next_cards(2, force_refresh=True)
    print(cards.to_string())
    print("shape:", cards.shape)
