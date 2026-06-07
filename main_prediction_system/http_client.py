"""
Shared HTTP client for ufcstats.com scraping.

ufcstats.com sits behind a JavaScript proof-of-work anti-bot gate: the first
request returns a "Checking your browser…" page containing a SHA-256 PoW. We
replicate that PoW in Python, POST the solution to /__c (which sets a session
cookie), then re-fetch the real page. The cookie persists on a shared Session,
so the challenge is normally solved once per run (and re-solved automatically if
the cookie expires mid-scrape).
"""

import re
import hashlib
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0"}
TIMEOUT = 20

_SESSION = None

# The challenge page embeds: nonce="...", target = new Array(<diff>+1).join('0'),
# and POSTs nonce + n to /__c.
_NONCE_RE = re.compile(r'nonce="([0-9a-fA-F]+)"')
_DIFF_RE = re.compile(r"new Array\((\d+)\+1\)")


def get_session():
    global _SESSION
    if _SESSION is None:
        _SESSION = requests.Session()
        _SESSION.headers.update(HEADERS)
    return _SESSION


def _is_challenge(resp):
    return "/__c" in resp.text and "nonce=" in resp.text


def _solve_challenge(session, url, resp):
    """Solve the embedded PoW and POST it; returns True if a solution was sent."""
    m_nonce = _NONCE_RE.search(resp.text)
    m_diff = _DIFF_RE.search(resp.text)
    if not m_nonce or not m_diff:
        return False

    nonce = m_nonce.group(1)
    target = "0" * int(m_diff.group(1))

    n = 0
    while not hashlib.sha256(f"{nonce}:{n}".encode()).hexdigest().startswith(target):
        n += 1

    base = "{0.scheme}://{0.netloc}".format(urlparse(url))
    session.post(
        base + "/__c",
        data={"nonce": nonce, "n": n},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=TIMEOUT,
    )
    return True


def get(url, max_retries=2):
    """GET a URL, transparently solving the PoW challenge if presented."""
    session = get_session()
    resp = session.get(url, timeout=TIMEOUT)

    attempts = 0
    while _is_challenge(resp) and attempts < max_retries:
        if not _solve_challenge(session, url, resp):
            break
        resp = session.get(url, timeout=TIMEOUT)
        attempts += 1

    resp.raise_for_status()
    return resp


def get_soup(url):
    return BeautifulSoup(get(url).text, "html.parser")
