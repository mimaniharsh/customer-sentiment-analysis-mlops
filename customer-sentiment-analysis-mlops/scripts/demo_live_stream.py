#!/usr/bin/env python3
"""
demo_live_stream.py — Sentient Intelligence Platform Demo

Simulates a live stream of customer feedback being analysed in real time.
Works with the Twitter Airline Sentiment dataset (Tweets.csv) OR any CSV
you point it at.

Usage:
    # Stream from the Kaggle dataset (auto-downloads if not present)
    python scripts/demo_live_stream.py

    # Stream from your own CSV
    python scripts/demo_live_stream.py --file path/to/your_data.csv --text-col "review_text"

    # Fast mode (no delay, bulk ingest)
    python scripts/demo_live_stream.py --delay 0 --batch 50

    # Slow cinematic mode for demos
    python scripts/demo_live_stream.py --delay 2.0 --limit 100

    # Upload a full CSV file for batch analysis
    python scripts/demo_live_stream.py --mode csv-upload --file Tweets.csv
"""

import argparse
import csv
import json
import random
import sys
import time
from io import StringIO
from pathlib import Path

try:
    import requests
except ImportError:
    print("❌ Missing dependency: pip install requests")
    sys.exit(1)

# ─── Colours ─────────────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
CYAN   = "\033[96m"
WHITE  = "\033[97m"

SENTIMENT_COLOUR = {"positive": GREEN, "neutral": YELLOW, "negative": RED}
URGENCY_COLOUR   = {"critical": RED, "high": YELLOW, "medium": CYAN, "low": GREEN}

# ─── Sample tweets (offline fallback) ────────────────────────────────────────
SAMPLE_TWEETS = [
    "Just landed at JFK — crew was incredible, smoothest flight I've had all year. Five stars!",
    "Third delay this week. No explanation, no apology. I'm done with this airline.",
    "The app crashed during check-in and I almost missed my flight. Completely unacceptable.",
    "Upgraded to business class for free! Best surprise ever. Thank you so much.",
    "Lost my luggage AGAIN. This is the second time in a month. Where is it??",
    "The new lounge at Heathrow is stunning. Great food, fast WiFi, friendly staff.",
    "Customer support hung up on me after 45 minutes on hold. Still no resolution.",
    "Flight was fine, nothing special. Arrived on time. Seat was comfortable enough.",
    "They found my bag and delivered it to my hotel within 2 hours. Impressed!",
    "Seat wouldn't recline, entertainment system broken, meal was cold. Disaster flight.",
    "Friendly pilot announcement, smooth landing, on-time arrival. Can't complain.",
    "Overcharged for checked bags with zero warning. Feeling completely ripped off.",
    "The vegan meal option was actually delicious. Pleasantly surprised on this one.",
    "WiFi connection dropped every 10 minutes. Paid $20 for essentially nothing.",
    "Ground crew helped me with my stroller without me even asking. Small gesture, big impression.",
    "Gate changed three times with no announcement. Ran half a mile in the wrong direction.",
    "Early boarding for families worked perfectly today. Kids settled in quickly.",
    "Seat 34B — zero legroom, broken tray table, armrest stuck. Never booking this again.",
    "The flight attendant remembered my name from a previous flight. That's loyalty.",
    "Cancelled flight, no hotel voucher, no meal voucher. Stranded for 14 hours.",
]


# ─── Helpers ─────────────────────────────────────────────────────────────────

def header():
    print(f"""
{BOLD}{CYAN}╔═══════════════════════════════════════════════════════════════╗
║         SENTIENT INTELLIGENCE PLATFORM — LIVE DEMO            ║
║         Real-Time Customer Sentiment Analysis Engine           ║
╚═══════════════════════════════════════════════════════════════╝{RESET}
""")


def sentiment_bar(pos: float, neu: float, neg: float, width: int = 40) -> str:
    p = int(pos * width)
    n = int(neu * width)
    g = max(0, width - p - n)
    return f"{GREEN}{'█' * p}{RESET}{YELLOW}{'█' * n}{RESET}{RED}{'█' * g}{RESET}"


def print_result(idx: int, text: str, result: dict):
    sentiment  = result.get("sentiment", "unknown")
    confidence = result.get("confidence", 0)
    urgency    = result.get("urgency", "low")
    impact     = result.get("business_impact_score", 0)
    attrs      = result.get("attributes", [])
    probs      = result.get("probabilities", {})
    latency    = result.get("latency_ms", 0)

    sc = SENTIMENT_COLOUR.get(sentiment, WHITE)
    uc = URGENCY_COLOUR.get(urgency, WHITE)

    short_text = (text[:72] + "…") if len(text) > 72 else text

    print(f"{DIM}┌{'─' * 65}┐{RESET}")
    print(f"{DIM}│{RESET} {BOLD}#{idx:04d}{RESET}  {DIM}{short_text}{RESET}")
    print(f"{DIM}│{RESET}")
    print(f"{DIM}│{RESET}  Sentiment  {sc}{BOLD}{sentiment.upper():10}{RESET}  Confidence {BOLD}{confidence:.1%}{RESET}")
    print(f"{DIM}│{RESET}  Urgency    {uc}{BOLD}{urgency.upper():10}{RESET}  Impact      {BOLD}{impact:.2f}/1.00{RESET}")
    if attrs:
        print(f"{DIM}│{RESET}  Attributes {CYAN}{', '.join(attrs)}{RESET}")
    print(f"{DIM}│{RESET}  {sentiment_bar(probs.get('positive', 0), probs.get('neutral', 0), probs.get('negative', 0))}")
    print(f"{DIM}│{RESET}  {GREEN}pos {probs.get('positive', 0):.0%}{RESET}  {YELLOW}neu {probs.get('neutral', 0):.0%}{RESET}  {RED}neg {probs.get('negative', 0):.0%}{RESET}   {DIM}{latency}ms{RESET}")
    print(f"{DIM}└{'─' * 65}┘{RESET}")


def print_summary(stats: dict, total: int, elapsed: float):
    pos = stats.get("positive", 0)
    neg = stats.get("negative", 0)
    neu = stats.get("neutral", 0)
    crit = stats.get("critical", 0)

    print(f"""
{BOLD}{WHITE}╔═══════════════════════════════════════════════════════════════╗
║                     SESSION SUMMARY                            ║
╚═══════════════════════════════════════════════════════════════╝{RESET}

  {BOLD}Messages processed:{RESET}  {total}
  {BOLD}Elapsed time:{RESET}        {elapsed:.1f}s  ({total/elapsed:.1f} msg/s)

  {GREEN}● Positive{RESET}   {pos:5d}  ({pos/total:.1%})
  {YELLOW}● Neutral{RESET}    {neu:5d}  ({neu/total:.1%})
  {RED}● Negative{RESET}   {neg:5d}  ({neg/total:.1%})
  {RED}{BOLD}⚠ Critical{RESET}   {crit:5d}  ({crit/total:.1%})

  Avg Business Impact:  {stats.get("avg_impact", 0):.3f}
  Avg Confidence:       {stats.get("avg_confidence", 0):.1%}
""")


# ─── Modes ───────────────────────────────────────────────────────────────────

def mode_stream(args, base_url: str):
    """Stream tweets one by one — the live demo mode."""
    tweets = load_tweets(args)
    if args.limit:
        tweets = tweets[:args.limit]
    if args.shuffle:
        random.shuffle(tweets)

    print(f"{CYAN}Streaming {len(tweets)} messages to {base_url}{RESET}\n")

    stats = {"positive": 0, "neutral": 0, "negative": 0, "critical": 0,
             "total_impact": 0.0, "total_confidence": 0.0}
    start = time.perf_counter()

    for idx, tweet in enumerate(tweets, 1):
        try:
            resp = requests.post(
                f"{base_url}/predict",
                json={"text": tweet, "source": "demo"},
                timeout=15,
            )
            resp.raise_for_status()
            result = resp.json()
        except requests.exceptions.ConnectionError:
            print(f"{RED}✗ Cannot connect to API at {base_url}{RESET}")
            print(f"  Make sure the platform is running: {BOLD}docker compose up{RESET}")
            sys.exit(1)
        except Exception as e:
            print(f"{RED}✗ Error on message #{idx}: {e}{RESET}")
            continue

        print_result(idx, tweet, result)

        s = result.get("sentiment", "neutral")
        stats[s] = stats.get(s, 0) + 1
        if result.get("urgency") == "critical":
            stats["critical"] += 1
        stats["total_impact"] += result.get("business_impact_score", 0)
        stats["total_confidence"] += result.get("confidence", 0)

        if idx % 10 == 0:
            print(f"{DIM}  ── {idx}/{len(tweets)} processed ──{RESET}\n")

        if args.delay > 0:
            time.sleep(args.delay)

    elapsed = time.perf_counter() - start
    total = len(tweets)
    stats["avg_impact"] = stats["total_impact"] / total if total else 0
    stats["avg_confidence"] = stats["total_confidence"] / total if total else 0
    print_summary(stats, total, elapsed)


def mode_batch(args, base_url: str):
    """Send messages in batches — fast ingest mode."""
    tweets = load_tweets(args)
    if args.limit:
        tweets = tweets[:args.limit]

    batch_size = args.batch
    total_processed = 0
    stats = {"positive": 0, "neutral": 0, "negative": 0, "critical": 0,
             "total_impact": 0.0, "total_confidence": 0.0}
    start = time.perf_counter()

    for i in range(0, len(tweets), batch_size):
        chunk = tweets[i:i + batch_size]
        try:
            resp = requests.post(
                f"{base_url}/predict/batch",
                json={"texts": chunk, "source": "demo-batch"},
                timeout=60,
            )
            resp.raise_for_status()
            result = resp.json()
        except Exception as e:
            print(f"{RED}✗ Batch error: {e}{RESET}")
            continue

        summary = result.get("summary", {})
        total_processed += result.get("total", 0)
        stats["positive"] += summary.get("positive", 0)
        stats["neutral"]  += summary.get("neutral", 0)
        stats["negative"] += summary.get("negative", 0)
        stats["critical"] += summary.get("critical_count", 0)
        stats["total_impact"]      += summary.get("avg_business_impact", 0) * len(chunk)
        stats["total_confidence"]  += summary.get("avg_confidence", 0) * len(chunk)

        print(f"{GREEN}✓{RESET} Batch {i//batch_size + 1}: {len(chunk)} messages  "
              f"{GREEN}pos={summary.get('positive', 0)}{RESET}  "
              f"{RED}neg={summary.get('negative', 0)}{RESET}  "
              f"{RED}critical={summary.get('critical_count', 0)}{RESET}")

    elapsed = time.perf_counter() - start
    if total_processed:
        stats["avg_impact"] = stats["total_impact"] / total_processed
        stats["avg_confidence"] = stats["total_confidence"] / total_processed
        print_summary(stats, total_processed, elapsed)


def mode_csv_upload(args, base_url: str):
    """Upload an entire CSV file to the /predict/csv endpoint."""
    file_path = Path(args.file) if args.file else None
    if not file_path or not file_path.exists():
        print(f"{RED}✗ File not found: {args.file}{RESET}")
        sys.exit(1)

    print(f"{CYAN}Uploading {file_path.name} ({file_path.stat().st_size // 1024}KB)…{RESET}")
    start = time.perf_counter()

    with open(file_path, "rb") as f:
        resp = requests.post(
            f"{base_url}/predict/csv",
            files={"file": (file_path.name, f, "text/csv")},
            timeout=300,
        )

    try:
        resp.raise_for_status()
    except Exception:
        print(f"{RED}✗ Upload failed: {resp.status_code} {resp.text[:200]}{RESET}")
        sys.exit(1)

    result = resp.json()
    elapsed = time.perf_counter() - start
    total = result.get("total", 0)
    summary = result.get("summary", {})

    print(f"\n{GREEN}✓ Upload complete — {total} rows analysed in {elapsed:.1f}s{RESET}\n")

    # Print top 10 most critical
    predictions = result.get("predictions", [])
    critical = [p for p in predictions if p.get("urgency") == "critical"]
    if critical:
        print(f"{RED}{BOLD}⚠ {len(critical)} CRITICAL messages found:{RESET}\n")
        for p in critical[:5]:
            text = p.get("text", "")[:80]
            impact = p.get("business_impact_score", 0)
            print(f"  {RED}●{RESET} [{impact:.2f}] {text}")

    print(f"\n{BOLD}Summary:{RESET}")
    print(f"  {GREEN}Positive:{RESET} {summary.get('positive', 0)} ({summary.get('positive', 0)/total:.1%})")
    print(f"  {YELLOW}Neutral: {RESET} {summary.get('neutral', 0)}")
    print(f"  {RED}Negative:{RESET} {summary.get('negative', 0)}")
    print(f"  {RED}Critical:{RESET} {summary.get('critical_count', 0)}")
    print(f"  Avg Confidence: {summary.get('avg_confidence', 0):.1%}")

    # Save results
    out_path = Path("demo_results.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n{CYAN}Full results saved to {out_path}{RESET}")


def load_tweets(args) -> list[str]:
    """Load tweets from CSV file or fall back to samples."""
    file_path = Path(args.file) if args.file else Path("Tweets.csv")

    if file_path.exists():
        print(f"{CYAN}Loading from {file_path}…{RESET}\n")
        tweets = []
        text_col = args.text_col

        with open(file_path, encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames or []

            # Auto-detect text column
            if not text_col:
                for candidate in ["text", "tweet", "message", "content", "review", "comment", "feedback"]:
                    if candidate in cols:
                        text_col = candidate
                        break
                if not text_col and cols:
                    text_col = cols[0]

            print(f"  Using column: {BOLD}{text_col}{RESET}  (available: {', '.join(cols[:5])}…)\n")

            for row in reader:
                val = row.get(text_col, "").strip()
                if val:
                    tweets.append(val)

        if tweets:
            return tweets

    print(f"{YELLOW}⚠ No CSV found — using built-in sample tweets{RESET}\n")
    return SAMPLE_TWEETS * 3  # repeat for longer demo


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Sentient Platform — Live Demo Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--api",      default="http://localhost:8000/api/v1", help="API base URL")
    parser.add_argument("--mode",     default="stream", choices=["stream", "batch", "csv-upload"],
                        help="Demo mode: stream (one-by-one), batch (fast), csv-upload (whole file)")
    parser.add_argument("--file",     default=None,    help="Path to CSV file")
    parser.add_argument("--text-col", default=None,    help="CSV column containing text (auto-detected)")
    parser.add_argument("--delay",    type=float, default=0.8, help="Seconds between messages (stream mode)")
    parser.add_argument("--limit",    type=int,   default=None, help="Max messages to process")
    parser.add_argument("--batch",    type=int,   default=25,   help="Batch size (batch mode)")
    parser.add_argument("--shuffle",  action="store_true",      help="Shuffle messages randomly")

    args = parser.parse_args()
    base_url = args.api.rstrip("/")

    header()

    # Health check
    try:
        r = requests.get(f"{base_url.replace('/api/v1', '')}/health", timeout=5)
        if r.status_code == 200:
            print(f"{GREEN}✓ API is online{RESET}\n")
        else:
            print(f"{YELLOW}⚠ API responded with {r.status_code}{RESET}\n")
    except Exception:
        print(f"{RED}✗ API not reachable at {base_url}{RESET}")
        print(f"  Run: {BOLD}docker compose up{RESET}\n")
        sys.exit(1)

    if args.mode == "stream":
        mode_stream(args, base_url)
    elif args.mode == "batch":
        mode_batch(args, base_url)
    elif args.mode == "csv-upload":
        mode_csv_upload(args, base_url)


if __name__ == "__main__":
    main()
