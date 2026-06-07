"""
Completed-fight results scraper for ufcstats.com.

Produces rows in the exact 44-column schema of the master training CSV
(mayberealdata_enriched.csv) so newly scraped fights can be appended and the
model retrained. This finishes the parser that was left commented-out in
archive/scrape_cards.ipynb.

Raw value formats are preserved to match the master file (the preprocessor parses
them downstream):
  - height "5' 7\"", reach "70\"", dob "Dec 07, 1993", stance "Orthodox"
  - SIG_STR/TD/TOTAL_STR/Head/... as "79 of 168", CTRL as "7:20"
  - event_date as "February 28, 2026"
"""

import time
from datetime import datetime

import pandas as pd

from http_client import get_soup


COMPLETED_EVENTS_URL = "http://ufcstats.com/statistics/events/completed?page=all"
SLEEP_BETWEEN_REQUESTS = 0.75

# Exact column order of the master dataset. Scraped rows are reindexed to this.
MASTER_COLUMNS = [
    "Event", "Fighter1", "Fighter2", "Winner", "Weightclass", "Method",
    "Round", "Time", "Format",
    "KD1", "KD2", "SIG_STR1", "SIG_STR2", "TOTAL_STR1", "TOTAL_STR2",
    "TD1", "TD2", "SUB_ATT1", "SUB_ATT2", "REV1", "REV2", "CTRL1", "CTRL2",
    "Head1", "Head2", "Body1", "Body2", "Leg1", "Leg2",
    "Distance1", "Distance2", "Clinch1", "Clinch2", "Ground1", "Ground2",
    "event_date", "f1_dob", "f1_height", "f1_reach",
    "f2_dob", "f2_height", "f2_reach", "f1_stance", "f2_stance",
]


# =========================================================
# Low-level HTML helpers
# =========================================================

def find_table_by_headers(soup, required_headers):
    """Return the first <table> whose headers contain all required keywords."""
    required_headers = [h.lower() for h in required_headers]
    for table in soup.find_all("table"):
        headers = [th.get_text(" ", strip=True).lower() for th in table.find_all("th")]
        if all(any(req in h for h in headers) for req in required_headers):
            return table
    return None


def row_red_blue_values(row):
    """Split a UFCStats stat row into red-corner and blue-corner value lists."""
    ps = [p.get_text(" ", strip=True) for p in row.find_all("p")]
    return ps[0::2], ps[1::2]


def _first_row_rb(table):
    """Red/blue values of a table's first body row (overall, not per-round)."""
    if not table:
        return [], []
    tbody = table.find("tbody")
    if not tbody:
        return [], []
    row = tbody.find("tr")
    if not row:
        return [], []
    return row_red_blue_values(row)


# =========================================================
# Fighter profile metadata (raw strings, matches master)
# =========================================================

def parse_fighter_profile(url):
    soup = get_soup(url)
    ul = soup.find("ul", class_="b-list__box-list")
    if not ul:
        # Profile block missing: layout change, or an anti-bot challenge slipped
        # through. Warn loudly instead of silently returning empty metadata,
        # which would otherwise be imputed and quietly skew predictions.
        print(f"WARNING: no profile data block found for {url} "
              f"(returning empty metadata).")
        return {"height": None, "reach": None, "stance": None, "dob": None}

    data = {}
    for li in ul.find_all("li", class_="b-list__box-list-item"):
        text = " ".join(li.get_text(" ", strip=True).split())
        if ":" in text:
            key, val = text.split(":", 1)
            data[key.strip().lower()] = val.strip()

    result = {
        "height": data.get("height"),
        "reach": data.get("reach"),
        "stance": data.get("stance"),
        "dob": data.get("dob"),
    }

    # dob/height are present on virtually every real profile; their absence
    # signals a parsing/fetch problem rather than legitimately-sparse data
    # (reach "--" and unknown stance are common and not warned on).
    critical_missing = [k for k in ("dob", "height") if not result.get(k)]
    if critical_missing:
        print(f"WARNING: profile for {url} missing {critical_missing} "
              f"(possible parsing/fetch issue).")

    return result


# =========================================================
# Event + fight listing
# =========================================================

def get_completed_events(before_date=None):
    """
    Return all completed events (newest first) as dicts:
    {name, date (str), date_dt, url}.
    """
    soup = get_soup(COMPLETED_EVENTS_URL)
    events = []

    tbody = soup.find("tbody")
    if not tbody:
        return events

    for tr in tbody.find_all("tr"):
        a = tr.find("a", class_="b-link b-link_style_black")
        if not a:
            continue
        name = a.text.strip()

        span = tr.select_one("span")
        if not span:
            continue
        date_str = span.text.strip()

        try:
            event_date = datetime.strptime(date_str, "%B %d, %Y")
        except ValueError:
            continue

        if before_date is not None and event_date >= before_date:
            continue

        events.append({
            "name": name,
            "date": date_str,
            "date_dt": event_date,
            "url": a.get("href"),
        })

    return events


def get_fight_links(event_url):
    """Return unique fight-detail URLs for one event card."""
    soup = get_soup(event_url)
    tbody = soup.find("tbody")
    if not tbody:
        return []

    links, seen = [], set()
    for a in tbody.find_all("a", class_="b-flag"):
        href = a.get("href")
        if href and href not in seen:
            seen.add(href)
            links.append(href)
    return links


# =========================================================
# Full fight-row parser (the main new work)
# =========================================================

def _parse_fight_meta(soup):
    """Extract Winner, Weightclass, Method, Round, Time, Format from a fight page."""
    # Fighters + win/loss/draw status
    names, urls, statuses = [], [], []
    for person in soup.find_all("div", class_="b-fight-details__person"):
        status_tag = person.find("i", class_="b-fight-details__person-status")
        name_tag = person.find("a", class_="b-link b-fight-details__person-link")
        names.append(name_tag.get_text(strip=True) if name_tag else "")
        urls.append(name_tag.get("href") if name_tag else None)
        statuses.append(status_tag.get_text(strip=True) if status_tag else "")

    winner = None
    for name, status in zip(names, statuses):
        if status == "W":
            winner = name
            break
    if winner is None:
        if "D" in statuses:
            winner = "DRAW"
        elif "NC" in statuses:
            winner = "NO CONTEST"

    # Weightclass (raw title text minus "Bout", matching the master vocabulary)
    weight_class = ""
    wc_tag = soup.find("i", class_="b-fight-details__fight-title")
    if wc_tag:
        weight_class = wc_tag.get_text(" ", strip=True).replace("Bout", "").strip()

    # Method (second inner <i> of the first text item)
    method = ""
    method_outer = soup.find("i", class_="b-fight-details__text-item_first")
    if method_outer:
        inner = method_outer.find_all("i")
        if len(inner) >= 2:
            method = inner[1].get_text(strip=True)

    # Round / Time / Time format
    round_ = time_ = format_ = ""
    for item in soup.find_all("i", class_="b-fight-details__text-item"):
        label_tag = item.find("i", class_="b-fight-details__label")
        if not label_tag:
            continue
        label = label_tag.get_text(strip=True).rstrip(":")
        label_tag.extract()
        value = item.get_text(" ", strip=True)
        if label == "Round":
            round_ = value
        elif label == "Time":
            time_ = value
        elif label == "Time format":
            token = value.split()[0] if value else ""
            try:
                format_ = int(token)
            except ValueError:
                format_ = token

    return names, urls, winner, weight_class, method, round_, time_, format_


def parse_fight_row(fight_link, event_name, event_date, fighter_profile_cache):
    """
    Scrape one fight page into a dict matching MASTER_COLUMNS.
    Returns None if the page lacks two fighters.
    """
    soup = get_soup(fight_link)

    (names, urls, winner, weight_class, method,
     round_, time_, format_) = _parse_fight_meta(soup)

    if len(names) < 2:
        return None

    # Totals table: name, KD, SIG, SIG%, TOTAL, TD, TD%, SUB, REV, CTRL
    KD1 = KD2 = SIG1 = SIG2 = TOT1 = TOT2 = TD1 = TD2 = ""
    SUB1 = SUB2 = REV1 = REV2 = CTRL1 = CTRL2 = ""
    totals_red, totals_blue = _first_row_rb(
        find_table_by_headers(soup, ["KD", "SIG", "TOTAL", "TD", "SUB", "REV", "CTRL"])
    )
    if len(totals_red) >= 10 and len(totals_blue) >= 10:
        r = totals_red[1:10]
        b = totals_blue[1:10]
        KD1, SIG1, _, TOT1, TD1, _, SUB1, REV1, CTRL1 = r
        KD2, SIG2, _, TOT2, TD2, _, SUB2, REV2, CTRL2 = b

    # Breakdown table: name, SIG, SIG%, HEAD, BODY, LEG, DISTANCE, CLINCH, GROUND
    Head1 = Head2 = Body1 = Body2 = Leg1 = Leg2 = ""
    Distance1 = Distance2 = Clinch1 = Clinch2 = Ground1 = Ground2 = ""
    brk_red, brk_blue = _first_row_rb(
        find_table_by_headers(soup, ["HEAD", "BODY", "LEG", "DISTANCE", "CLINCH", "GROUND"])
    )
    if len(brk_red) >= 6 and len(brk_blue) >= 6:
        Head1, Body1, Leg1, Distance1, Clinch1, Ground1 = brk_red[-6:]
        Head2, Body2, Leg2, Distance2, Clinch2, Ground2 = brk_blue[-6:]

    # Fighter profiles (cached per fighter URL)
    f1_url, f2_url = urls[0], urls[1]
    for u in (f1_url, f2_url):
        if u and u not in fighter_profile_cache:
            fighter_profile_cache[u] = parse_fighter_profile(u)
            time.sleep(SLEEP_BETWEEN_REQUESTS)
    f1_meta = fighter_profile_cache.get(f1_url, {})
    f2_meta = fighter_profile_cache.get(f2_url, {})

    return {
        "Event": event_name,
        "Fighter1": names[0], "Fighter2": names[1],
        "Winner": winner, "Weightclass": weight_class, "Method": method,
        "Round": round_, "Time": time_, "Format": format_,
        "KD1": KD1, "KD2": KD2,
        "SIG_STR1": SIG1, "SIG_STR2": SIG2,
        "TOTAL_STR1": TOT1, "TOTAL_STR2": TOT2,
        "TD1": TD1, "TD2": TD2,
        "SUB_ATT1": SUB1, "SUB_ATT2": SUB2,
        "REV1": REV1, "REV2": REV2,
        "CTRL1": CTRL1, "CTRL2": CTRL2,
        "Head1": Head1, "Head2": Head2,
        "Body1": Body1, "Body2": Body2,
        "Leg1": Leg1, "Leg2": Leg2,
        "Distance1": Distance1, "Distance2": Distance2,
        "Clinch1": Clinch1, "Clinch2": Clinch2,
        "Ground1": Ground1, "Ground2": Ground2,
        "event_date": event_date,
        "f1_dob": f1_meta.get("dob"), "f1_height": f1_meta.get("height"),
        "f1_reach": f1_meta.get("reach"),
        "f2_dob": f2_meta.get("dob"), "f2_height": f2_meta.get("height"),
        "f2_reach": f2_meta.get("reach"),
        "f1_stance": f1_meta.get("stance"), "f2_stance": f2_meta.get("stance"),
    }


# =========================================================
# High-level: scrape events + fetch new + append/dedup
# =========================================================

def scrape_events(events, progress=print):
    """Scrape every fight of the given events into a master-schema DataFrame."""
    cache = {}
    rows = []
    for event in events:
        progress(f"Scraping: {event['name']} | {event['date']}")
        try:
            fight_links = get_fight_links(event["url"])
        except Exception as e:
            progress(f"  Could not load event page: {e}")
            continue
        progress(f"  {len(fight_links)} fights")

        for i, link in enumerate(fight_links, start=1):
            try:
                row = parse_fight_row(link, event["name"], event["date"], cache)
                if row is not None:
                    rows.append(row)
            except Exception as e:
                progress(f"  [skip {i}/{len(fight_links)}] {link}: {e}")
            time.sleep(SLEEP_BETWEEN_REQUESTS)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.reindex(columns=MASTER_COLUMNS)
    return df


def fetch_new(master_df, max_new_events=3, progress=print):
    """
    Scrape up to `max_new_events` most-recent completed events that are NOT
    already present in `master_df` (matched by Event name). Returns a
    master-schema DataFrame of new fight rows (possibly empty).
    """
    known_events = set(master_df["Event"].astype(str)) if "Event" in master_df else set()
    all_events = get_completed_events()

    new_events = [e for e in all_events if e["name"] not in known_events]
    if max_new_events is not None:
        new_events = new_events[:max_new_events]

    if not new_events:
        progress("No new completed events found.")
        return pd.DataFrame(columns=MASTER_COLUMNS)

    progress(f"Found {len(new_events)} new event(s) to scrape.")
    return scrape_events(new_events, progress=progress)


def append_dedup(master_path, new_rows):
    """
    Append `new_rows` to the master CSV, de-duplicating on
    (event_date, {Fighter1, Fighter2}). Returns the number of rows added.
    """
    master = pd.read_csv(master_path)
    before = len(master)

    if new_rows is None or new_rows.empty:
        return 0

    new_rows = new_rows.reindex(columns=master.columns)
    combined = pd.concat([master, new_rows], ignore_index=True)

    def _key(row):
        pair = frozenset([str(row["Fighter1"]).strip(), str(row["Fighter2"]).strip()])
        return (str(row["event_date"]).strip(), pair)

    combined["_dedup_key"] = combined.apply(_key, axis=1)
    combined = combined.drop_duplicates(subset="_dedup_key", keep="first")
    combined = combined.drop(columns="_dedup_key")

    added = len(combined) - before
    if added > 0:
        combined.to_csv(master_path, index=False)
    return added


if __name__ == "__main__":
    # Standalone smoke test: scrape the single most recent completed event.
    events = get_completed_events()[:1]
    df = scrape_events(events)
    print(df.head())
    print("shape:", df.shape)
    print("columns match master order:", list(df.columns) == MASTER_COLUMNS)
