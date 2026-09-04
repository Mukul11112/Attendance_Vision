"""
core/nvr.py
Hikvision ISAPI queries — the metadata behind the NVR's web UI.

Two things the app needs that RTSP alone cannot answer:
  • which DAYS actually hold recordings (the blue marks on the web calendar)
  • what each channel is really called ("cafeteria", "Main-door", ...)

Everything here is read-only HTTP against the NVR, authenticated with the same
camera_config.json credentials used for RTSP. Digest auth is what Hikvision
firmware expects; basic is registered as a fallback for older units.

Verified against this site's NVR (16 channels) on 31 Jul 2026.
"""
from __future__ import annotations
import calendar
import html
import logging
import os
import re
import time
import urllib.request
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional
from urllib.request import (HTTPBasicAuthHandler, HTTPDigestAuthHandler,
                            HTTPPasswordMgrWithDefaultRealm)

log = logging.getLogger("nvr")

TIMEOUT = 10
DOWNLOAD_TIMEOUT = 120
_SEARCH_ID = "{A0B1C2D3-E4F5-6789-ABCD-EF0123456789}"


def _opener(url: str, user: str, password: str):
    mgr = HTTPPasswordMgrWithDefaultRealm()
    mgr.add_password(None, url, user, password)
    return urllib.request.build_opener(HTTPDigestAuthHandler(mgr),
                                       HTTPBasicAuthHandler(mgr))


def _request(cfg: dict, path: str, body: Optional[str] = None) -> str:
    url = f"http://{cfg['ip']}{path}"
    data = body.encode() if body else None
    req = urllib.request.Request(
        url, data=data, method="POST" if body else "GET",
        headers={"Content-Type": "application/xml"} if body else {})
    with _opener(url, str(cfg["username"]), str(cfg["password"])).open(
            req, timeout=TIMEOUT) as resp:
        return resp.read().decode("utf-8", "replace")


def recorded_days(cfg: dict, year: int, month: int,
                  channel: str = "101") -> List[str]:
    """ISO dates in `year`-`month` that hold recordings, oldest first.

    This is exactly what the web UI's calendar marks blue. Raises on transport
    or auth failure so the caller can show the real reason instead of silently
    presenting an empty month.
    """
    body = ('<?xml version="1.0" encoding="utf-8"?>'
            f'<trackDailyParam><year>{year}</year>'
            f'<monthOfYear>{month}</monthOfYear></trackDailyParam>')
    xml = _request(cfg, f"/ISAPI/ContentMgmt/record/tracks/{channel}/dailyDistribution",
                   body)
    pairs = re.findall(r"<dayOfMonth>(\d+)</dayOfMonth>\s*<record>(\w+)</record>", xml)
    last = calendar.monthrange(year, month)[1]
    days = [int(d) for d, rec in pairs if rec.lower() == "true" and 1 <= int(d) <= last]
    return [f"{year:04d}-{month:02d}-{d:02d}" for d in sorted(set(days))]


@dataclass
class Segment:
    """One continuous recording the NVR holds, for one camera."""
    track: str                  # "101" = camera 1 main stream
    start: str                  # "2026-07-30T01:04:12Z"
    end: str
    uri: str                    # playbackURI, used to download it
    size: int = 0               # bytes, 0 when the NVR does not report it

    @property
    def start_clock(self) -> str:
        m = re.search(r"T(\d{2}):?(\d{2}):?(\d{2})", self.start)
        return f"{m.group(1)}:{m.group(2)}:{m.group(3)}" if m else "00:00:00"

    @property
    def minutes(self) -> float:
        def _sec(ts: str) -> float:
            m = re.search(r"T(\d{2}):?(\d{2}):?(\d{2})", ts)
            return (int(m.group(1)) * 3600 + int(m.group(2)) * 60
                    + int(m.group(3))) if m else 0.0
        d = _sec(self.end) - _sec(self.start)
        return (d if d >= 0 else d + 86400) / 60.0


def search_segments(cfg: dict, track: str, day: str,
                    max_results: int = 200) -> List[Segment]:
    """Every recorded segment for one camera on one day.

    The NVR stores a day as a handful of ~80-minute files (18 on this unit),
    not one continuous stream. Downloading them individually is what makes a
    full day practical — measured ~37 MB/s versus RTSP playback's 1.1x
    realtime.
    """
    body = (f'<?xml version="1.0" encoding="utf-8"?>'
            f'<CMSearchDescription><searchID>{_SEARCH_ID}</searchID>'
            f'<trackIDList><trackID>{track}</trackID></trackIDList>'
            f'<timeSpanList><timeSpan>'
            f'<startTime>{day}T00:00:00Z</startTime>'
            f'<endTime>{day}T23:59:59Z</endTime>'
            f'</timeSpan></timeSpanList>'
            f'<maxResults>{max_results}</maxResults>'
            f'<searchResultPostion>0</searchResultPostion>'
            f'</CMSearchDescription>')
    xml = _request(cfg, "/ISAPI/ContentMgmt/search", body)

    out: List[Segment] = []
    # Parse per match item: the response also carries the SEARCH span, so
    # scanning for bare <startTime> tags would pick up a phantom segment.
    for block in re.split(r"<searchMatchItem>", xml)[1:]:
        uri = re.search(r"<playbackURI>(.*?)</playbackURI>", block, re.S)
        span = re.search(r"<startTime>(.*?)</startTime>\s*<endTime>(.*?)</endTime>",
                         block, re.S)
        if not (uri and span):
            continue
        size = re.search(r"<size>(\d+)</size>", block)
        out.append(Segment(track=track, start=span.group(1).strip(),
                           end=span.group(2).strip(),
                           uri=html.unescape(uri.group(1).strip()),
                           size=int(size.group(1)) if size else 0))
    out.sort(key=lambda s: s.start)
    return out


def download(cfg: dict, seg: Segment, dest: str,
             cancel: Optional[Callable[[], bool]] = None,
             on_bytes: Optional[Callable[[int], None]] = None) -> str:
    """Fetch one segment to `dest`. Returns the path actually written.

    Writes to a .part file and renames on success, so an aborted download can
    never be mistaken for a complete video by the batch pipeline.
    """
    url = f"http://{cfg['ip']}/ISAPI/ContentMgmt/download"
    body = ('<?xml version="1.0" encoding="utf-8"?>'
            f'<downloadRequest><playbackURI>{html.escape(seg.uri)}'
            '</playbackURI></downloadRequest>')
    req = urllib.request.Request(url, data=body.encode(), method="POST",
                                 headers={"Content-Type": "application/xml"})
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    got = 0
    opener = _opener(url, str(cfg["username"]), str(cfg["password"]))
    with opener.open(req, timeout=DOWNLOAD_TIMEOUT) as resp, open(part, "wb") as fh:
        while True:
            if cancel and cancel():
                fh.close()
                _quiet_remove(part)
                raise InterruptedError("download cancelled")
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            got += len(chunk)
            if on_bytes:
                on_bytes(got)
    if got == 0:
        _quiet_remove(part)
        raise IOError(f"NVR returned an empty file for {seg.start}")
    _quiet_remove(dest)
    os.replace(part, dest)
    return dest


def _quiet_remove(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError as e:                        # file still held by a decoder
        log.warning("could not remove %s: %s", path, e)


def channel_names(cfg: dict) -> Dict[int, str]:
    """{channel number: display name} as configured on the NVR itself."""
    try:
        xml = _request(cfg, "/ISAPI/ContentMgmt/InputProxy/channels")
    except Exception as e:                      # names are cosmetic — never fatal
        log.warning("could not read channel names from NVR: %s", e)
        return {}
    ids = re.findall(r"<id>(\d+)</id>", xml)
    names = re.findall(r"<name>(.*?)</name>", xml)
    return {int(i): n for i, n in zip(ids, names)}
