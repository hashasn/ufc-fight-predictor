import requests
import json

URL = "https://api.fightodds.io/gql"

QUERY = """
query PromotionEventCardListInfiniteScrollQuery(
  $count: Int!
  $cursor: String
  $promotionSlug: String
  $dateGte: Date
  $dateLt: Date
  $orderBy: String
) {
  promotionBySlug(slug: $promotionSlug) {
    ...PromotionEventCardListInfiniteScroll_promotion_2cotnU
    id
  }
}

fragment EventCardList_events on EventNodeConnection {
  edges {
    node {
      id
      ...EventCard_event
    }
  }
}

fragment EventCard_event on EventNode {
  id
  name
  pk
  slug
  date
  venue
  city
  promotion {
    slug
    shortName
    id
  }
  ...EventPoster_event
}

fragment EventPoster_event on EventNode {
  name
  poster
  posterWide
  promotion {
    shortName
    logo
    id
  }
}

fragment PromotionEventCardListInfiniteScroll_promotion_2cotnU on PromotionNode {
  events(first: $count, after: $cursor, date_Gte: $dateGte, date_Lt: $dateLt, orderBy: $orderBy) {
    ...EventCardList_events
    edges {
      node {
        id
        __typename
      }
      cursor
    }
    pageInfo {
      hasNextPage
      endCursor
    }
  }
}
"""

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Content-Type": "application/json",
    "Accept": "application/json",
    "Origin": "https://fightodds.io",
    "Referer": "https://fightodds.io/",
}

def fetch_events(cursor=None, count=10):
    payload = {
        "query": QUERY,
        "variables": {
            "count": count,
            "cursor": cursor,
            "promotionSlug": "ufc",
            "dateGte": None,
            "dateLt": "2026-04-03",
            "orderBy": "-date"
        }
    }

    r = requests.post(URL, headers=HEADERS, json=payload, timeout=30)
    print("status:", r.status_code)
    print(r.text[:500])
    return r

def get_all_ufc_events():
    cursor = None
    all_events = []

    while True:
        payload = {
            "query": QUERY,
            "variables": {
                "count": 10,
                "cursor": cursor,
                "promotionSlug": "ufc",
                "dateGte": None,
                "dateLt": "2026-04-03",
                "orderBy": "-date"
            }
        }

        r = requests.post(URL, headers=HEADERS, json=payload, timeout=30)
        if r.status_code != 200:
            print("Request failed:", r.status_code)
            print(r.text[:500])
            break

        data = r.json()

        events_block = data["data"]["promotionBySlug"]["events"]
        edges = events_block["edges"]

        for edge in edges:
            node = edge["node"]
            all_events.append({
                "name": node["name"],
                "slug": node["slug"],
                "date": node["date"],
                "city": node["city"],
                "venue": node["venue"],
            })

        page_info = events_block["pageInfo"]
        if not page_info["hasNextPage"]:
            break

        cursor = page_info["endCursor"]

    return all_events

fetch_events()
# get_all_ufc_events()
