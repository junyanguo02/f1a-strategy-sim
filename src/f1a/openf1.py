"""OpenF1 historical laps and Jolpica race identities/results."""
from bisect import bisect_left, bisect_right
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
import json
import re
import time
from urllib.parse import urlencode, urlsplit

import pandas as pd
import requests

OPEN = "https://api.openf1.org/v1/"
JOLPI = "https://api.jolpi.ca/ergast/f1/"
_last_request = {}


def fetch(base, endpoint, *, daily=False, allow_no_results=False, **params):
    """Cache successful JSON; never cache empty responses or HTTP errors."""
    folder = Path("data/raw") / ("openf1" if base == OPEN else "jolpica")
    name = endpoint.strip("/").replace("/", "_")
    query = urlencode(sorted(params.items()))
    path = folder / f"{name}{'_' + query if query else ''}.json"
    if path.exists() and (not daily or time.time() - path.stat().st_mtime < 86400):
        return json.loads(path.read_bytes())
    url = base + endpoint + ("?" + query if query else "")
    host = urlsplit(base).netloc
    for attempt in range(4):
        # Serial requests at <= 30/min and <= 1/s per host.
        time.sleep(max(0, _last_request.get(host, 0) + 2.1 - time.monotonic()))
        _last_request[host] = time.monotonic()
        try:
            response = requests.get(url, timeout=(12, 30))
        except (requests.ConnectionError, requests.Timeout):
            if attempt == 3:
                raise
            time.sleep(2 ** (attempt + 1))
            continue
        if response.status_code == 429 and attempt < 3:
            header = response.headers.get("Retry-After")
            delay = 2 ** (attempt + 1)
            if header:
                try:
                    delay = max(delay, float(header))
                except ValueError:
                    try:
                        delay = max(delay, parsedate_to_datetime(header).timestamp() - time.time())
                    except (ValueError, TypeError, OverflowError):
                        pass
            time.sleep(delay)
            continue
        if allow_no_results and response.status_code == 404:
            try:
                detail = response.json()
            except ValueError:
                detail = None
            if detail == {"detail": "No results found."}:
                print(f"Missing auxiliary feed: HTTP 404 {url}", flush=True)
                return []
        if response.status_code != 200:
            raise RuntimeError(f"HTTP {response.status_code} {url}")
        result = response.json()
        if result:
            folder.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(response.content)
            tmp.replace(path)
        return result



def calendar(year):
    payload = fetch(JOLPI, f"{year}/races/", limit=100,
                    daily=year >= datetime.now(timezone.utc).year)
    races = payload["MRData"]["RaceTable"]["Races"]
    if not races:
        raise ValueError(f"No Jolpica schedule for {year}")
    return races


def results(year, round_number):
    payload = fetch(JOLPI, f"{year}/{round_number}/results/", limit=100,
                    daily=year >= datetime.now(timezone.utc).year)
    return payload["MRData"]["RaceTable"]["Races"][0]["Results"]


def race_session(year, round_number):
    race = next((r for r in calendar(year) if int(r["round"]) == round_number), None)
    if race is None:
        raise ValueError(f"No scheduled race for {year} R{round_number}")
    start = pd.Timestamp(f"{race['date']}T{race.get('time') or '23:59:59Z'}")
    now = pd.Timestamp.now(tz="UTC")
    if start >= now:
        raise ValueError(f"Race not yet held: {start}")
    sessions = fetch(OPEN, "sessions", year=year, session_name="Race",
                     daily=year >= now.year)
    matches = []
    for candidate in sessions:
        utc_start = pd.Timestamp(candidate["date_start"])
        offset = candidate.get("gmt_offset") or "00:00:00"
        hours, minutes, seconds = map(int, offset.lstrip("+-").split(":"))
        delta = timedelta(hours=hours, minutes=minutes, seconds=seconds)
        local_start = utc_start - delta if offset.startswith("-") else utc_start + delta
        if not candidate.get("is_cancelled") and start.date() in {utc_start.date(), local_start.date()}:
            matches.append(candidate)
    if len(matches) != 1:
        raise ValueError(f"Expected one OpenF1 Race on {start.date()}, found {len(matches)}")
    session = matches[0]
    if pd.Timestamp(session["date_end"]) > now:
        raise ValueError("Race not yet ended")
    return session


def control_changes(messages):
    """Map only global race-control indications; sector yellows are excluded."""
    changes = []
    for m in sorted(messages, key=lambda r: pd.Timestamp(r["date"])):
        msg = m["message"].upper()
        flag = m.get("flag")
        code = None
        if re.match(r"^(VIRTUAL SAFETY CAR|VSC) DEPLOYED", msg):
            code = "6"
        elif re.match(r"^(VIRTUAL SAFETY CAR|VSC) ENDING", msg):
            code = "7"
        elif msg.startswith("SAFETY CAR DEPLOYED"):
            code = "4"
        elif flag == "RED":
            code = "5"
        elif flag in {"GREEN", "CLEAR"} and m.get("scope") == "Track":
            code = "1"
        if code:
            changes.append((pd.Timestamp(m["date"]), code))
    return changes


def load_openf1(year, round_number):
    session = race_session(year, round_number)
    key = session["session_key"]
    inputs = {}
    for endpoint in ("laps", "drivers", "stints", "pit", "weather", "race_control"):
        inputs[endpoint] = fetch(
            OPEN, endpoint, session_key=key,
            allow_no_results=endpoint in {"pit", "weather", "race_control", "stints"},
        )
    return assemble(inputs, session, year, round_number)


def assemble(inputs, session, year, round_number):
    laps = sorted(inputs["laps"], key=lambda x: (x["driver_number"], x["lap_number"]))
    if not laps:
        raise ValueError("OpenF1 returned no laps")
    drivers = {d["driver_number"]: d for d in inputs["drivers"]}
    pits = {(p["driver_number"], p["lap_number"]) for p in inputs["pit"]}
    stints = {}
    for s in sorted(inputs["stints"], key=lambda s: s["stint_number"]):
        for n in range(s["lap_start"], (s["lap_end"] or s["lap_start"]) + 1):
            stints[s["driver_number"], n] = s
    weather = sorted((pd.Timestamp(w["date"]), i, w) for i, w in enumerate(inputs["weather"]))
    weather_times = [w[0] for w in weather]
    changes = control_changes(inputs["race_control"])
    change_times = [c[0] for c in changes]
    rows = []
    for i, lap in enumerate(laps):
        car, n = lap["driver_number"], lap["lap_number"]
        pair = (car, n)
        driver = drivers.get(car, {})
        stint = stints.get(pair, {})
        age = stint.get("tyre_age_at_start")
        duration = lap.get("lap_duration")
        start = pd.Timestamp(lap["date_start"])
        if pd.isna(start) and n == 1:
            start = pd.Timestamp(session["date_start"])
        end = pd.NaT
        if i+1 < len(laps) and (laps[i+1]["driver_number"], laps[i+1]["lap_number"]) == (car, n+1):
            end = pd.Timestamp(laps[i+1]["date_start"])
        if pd.isna(end) and pd.notna(start) and duration is not None:
            end = start + timedelta(seconds=duration)
        known_start = pd.notna(start)
        known_interval = known_start and pd.notna(end) and end > start
        w = {}
        if known_start:
            j = bisect_left(weather_times, start)
            if not (known_interval and j < len(weather) and weather_times[j] < end):
                j = bisect_right(weather_times, start) - 1
            if j >= 0:
                w = weather[j][2]
        codes = None
        if known_interval and inputs["race_control"]:
            lo = bisect_right(change_times, start)
            hi = bisect_left(change_times, end)
            initial = changes[lo-1][1] if lo else "1"
            codes = "".join(dict.fromkeys([initial] + [c[1] for c in changes[lo:hi]]))
        rows.append(dict(
            driver=driver.get("name_acronym"), team=driver.get("team_name"),
            driver_number=car, lap=n, lap_time_s=duration, compound=stint.get("compound"),
            tyre_life=None if age is None else age + n - stint["lap_start"] + 1,
            stint=stint.get("stint_number"), pit_in=pair in pits if inputs["pit"] else None,
            pit_out=lap.get("is_pit_out_lap"), track_status=codes,
            air_temp=w.get("air_temperature"), track_temp=w.get("track_temperature"),
            safety_car=None if codes is None else "4" in codes,
            virtual_safety_car=None if codes is None else bool(set(codes) & {"6", "7"}),
            first_lap=n == 1, year=year, round=round_number,
        ))
    frame = pd.DataFrame(rows)
    if frame.duplicated(["driver_number", "lap"]).any():
        raise ValueError("Duplicate OpenF1 driver/lap records")
    return frame.astype({"lap_time_s": "Float64", "tyre_life": "Int64", "stint": "Int64",
                         "pit_in": "boolean", "pit_out": "boolean", "safety_car": "boolean",
                         "virtual_safety_car": "boolean", "air_temp": "Float64",
                         "track_temp": "Float64", "track_status": "string"
                         }).sort_values(["driver", "lap"], kind="stable").reset_index(drop=True)
