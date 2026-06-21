#!/usr/bin/env python3
"""
morse_transmit.py — Send a message to osmarks via HTTP request volumes.

Protocol:
  - On-Off Keying over request rate
  - T = 2 seconds (base time unit)
  - Dot  = high rate for 1T, then silence for 1T
  - Dash = high rate for 3T, then silence for 1T
  - Letter gap = silence for 3T (after the trailing 1T inter-element gap)
  - Word gap   = silence for 7T (after the trailing 1T inter-element gap)
  - Rate during ON: ~8 req/s (adjustable via RATE)
  - User-Agent: distinctive so he can grep just your stream

Usage:
  python3 morse_transmit.py "your message here"
  python3 morse_transmit.py --dry-run "SOS"
"""

import sys
import time
import threading
import argparse
import urllib.request
import urllib.error

# ── Config ────────────────────────────────────────────────────────────────────

TARGET   = "https://status.osmarks.net/"
AGENT    = "MorseContact/1.0 (+https://github.com/osmarks/osmarks.net)"
T        = 2.0        # base time unit in seconds
RATE     = 8          # requests per second during ON periods
DRY_RUN  = False

# ── Morse table ───────────────────────────────────────────────────────────────

MORSE = {
    'A': '.-',    'B': '-...',  'C': '-.-.',  'D': '-..',
    'E': '.',     'F': '..-.',  'G': '--.',   'H': '....',
    'I': '..',    'J': '.---',  'K': '-.-',   'L': '.-..',
    'M': '--',    'N': '-.',    'O': '---',   'P': '.--.',
    'Q': '--.-',  'R': '.-.',   'S': '...',   'T': '-',
    'U': '..-',   'V': '...-',  'W': '.--',   'X': '-..-',
    'Y': '-.--',  'Z': '--..',
    '0': '-----', '1': '.----', '2': '..---', '3': '...--',
    '4': '....-', '5': '.....', '6': '-....', '7': '--...',
    '8': '---..',  '9': '----.',
    '.': '.-.-.-', ',': '--..--', '?': '..--..', "'": '.----.',
    '!': '-.-.--', '/': '-..-.', '(': '-.--.',  ')': '-.--.-',
    '&': '.-...',  ':': '---...', ';': '-.-.-.',  '=': '-...-',
    '+': '.-.-.',  '-': '-....-', '_': '..--.-',  '"': '.-..-.',
    '$': '...-..-','@': '.--.-.',
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def text_to_morse(text: str) -> list[tuple[str, str]]:
    """Convert text to list of (char, morse_sequence) tuples, skipping unknowns."""
    result = []
    for ch in text.upper():
        if ch == ' ':
            result.append((' ', ''))
        elif ch in MORSE:
            result.append((ch, MORSE[ch]))
        else:
            print(f"  [skip] '{ch}' has no Morse representation")
    return result

def estimate_duration(tokens: list[tuple[str, str]]) -> float:
    """Estimate total transmission time in seconds."""
    total = 0.0
    for i, (ch, seq) in enumerate(tokens):
        if ch == ' ':
            total += 7 * T  # word gap (replaces letter gap)
            continue
        for j, sym in enumerate(seq):
            total += T if sym == '.' else 3 * T   # dot or dash ON
            total += T                              # inter-element gap
            if j < len(seq) - 1:
                pass  # gap already counted
        # letter gap: 3T total, minus the 1T inter-element already added
        if i < len(tokens) - 1 and tokens[i+1][0] != ' ':
            total += 2 * T  # extra 2T to make letter gap = 3T
    return total

def send_burst(duration: float, label: str):
    """Fire requests at RATE req/s for `duration` seconds."""
    if DRY_RUN:
        print(f"    [DRY] ON  for {duration:.1f}s  ({label})")
        time.sleep(duration)
        return

    deadline = time.monotonic() + duration
    interval = 1.0 / RATE
    sent = 0

    def fire():
        nonlocal sent
        req = urllib.request.Request(TARGET, headers={"User-Agent": AGENT})
        try:
            urllib.request.urlopen(req, timeout=5)
            sent += 1
        except Exception:
            sent += 1  # still count even on error; the request hit the logs

    while time.monotonic() < deadline:
        t = threading.Thread(target=fire, daemon=True)
        t.start()
        time.sleep(interval)

    print(f"    ON  {duration:.1f}s  (~{sent} req)  [{label}]")

def silence(duration: float, label: str = ""):
    if DRY_RUN:
        print(f"    [DRY] OFF for {duration:.1f}s  ({label})")
    else:
        print(f"    OFF {duration:.1f}s  [{label}]")
    time.sleep(duration)

# ── Transmitter ───────────────────────────────────────────────────────────────

def transmit(text: str):
    tokens = text_to_morse(text)
    est = estimate_duration(tokens)

    print(f"\n{'─'*60}")
    print(f"  Message : {text}")
    print(f"  Target  : {TARGET}")
    print(f"  Agent   : {AGENT}")
    print(f"  T       : {T}s  |  Rate: {RATE} req/s")
    print(f"  Dry run : {DRY_RUN}")
    print(f"  Est.    : {est:.0f}s  ({est/60:.1f} min)")
    print(f"{'─'*60}\n")

    if not DRY_RUN:
        input("  Press Enter to begin transmission... ")
        print()

    for i, (ch, seq) in enumerate(tokens):
        if ch == ' ':
            print(f"  [WORD GAP]")
            silence(7 * T, "word gap")
            continue

        print(f"  [{ch}]  {seq}")

        for j, sym in enumerate(seq):
            is_last_sym = (j == len(seq) - 1)

            if sym == '.':
                send_burst(T, "dot")
            else:
                send_burst(3 * T, "dash")

            # inter-element gap
            if not is_last_sym:
                silence(T, "inter-element")

        # letter gap (3T total; 1T already elapsed as post-symbol silence)
        is_last_char = (i == len(tokens) - 1)
        next_is_space = (i + 1 < len(tokens) and tokens[i+1][0] == ' ')

        if not is_last_char and not next_is_space:
            silence(2 * T, "letter gap (+2T)")  # brings total to 3T
        elif not is_last_char and next_is_space:
            silence(T, "pre-word silence")       # word gap follows

    print("\n  ✓ Transmission complete.\n")

# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    global DRY_RUN, T, RATE, TARGET

    parser = argparse.ArgumentParser(description="Transmit Morse via HTTP request volumes.")
    parser.add_argument("message", nargs="+", help="Message to transmit")
    parser.add_argument("--dry-run", action="store_true", help="Simulate without sending requests")
    parser.add_argument("--T", type=float, default=2.0, metavar="SECONDS",
                        help="Base time unit in seconds (default: 2.0)")
    parser.add_argument("--rate", type=int, default=8, metavar="REQ_PER_SEC",
                        help="Requests per second during ON periods (default: 8)")
    parser.add_argument("--target", type=str, default=TARGET,
                        help=f"Target URL (default: {TARGET})")
    args = parser.parse_args()

    DRY_RUN = args.dry_run
    T       = args.T
    RATE    = args.rate
    TARGET  = args.target

    message = " ".join(args.message)
    transmit(message)

if __name__ == "__main__":
    main()
