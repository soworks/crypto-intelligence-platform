"""Production evidence capture for the open UTC session.

This workload stores pre-close and post-close observations. It does not scan,
place orders, or record a production clock. A missing classification is left
unstored; this handler does not write an absence.
"""

from __future__ import annotations

import os
import re
import shutil
from collections.abc import Callable
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from time import sleep as _sleep
from typing import Any, Protocol, cast

import boto3
import httpx
from aws_lambda_powertools import Logger
from aws_lambda_powertools.utilities.typing import LambdaContext

from cip.adapters.binance import BinanceMarketClient
from cip.adapters.market import Depth, ExchangeInfo, Kline, Ticker24h, parse_klines
from cip.domain.errors import (
    EvaluationError,
    ExchangeBannedError,
    MarketDataError,
    RateLimited,
    RecorderError,
)
from cip.domain.policy import load_policy
from cip.evaluation.capture_cycle import BookShot, CycleReport, TemporaryFailure, run_capture_cycle
from cip.evaluation.classification import SymbolClassification
from cip.evaluation.classification_bundle import ensure_bundle
from cip.evaluation.classification_producer import CatalogIncomplete, produce_classifications
from cip.evaluation.liquidity_capture import HourQuote, book_metrics, parse_daily_klines
from cip.evaluation.populate import BLOCKED_THROUGH, RegimeCapture
from cip.evaluation.prospect import universe_capture
from cip.evaluation.session import session_close
from cip.history.bars import DailyBar
from cip.recorders.sources import (
    COINGECKO_GLOBAL,
    DEFILLAMA_STABLECOINS,
    fetch_json,
    parse_btc_dominance,
    parse_stablecoin_supply,
)

logger = Logger(service="cip-session-capture")


def _pause(seconds: float) -> None:
    _sleep(seconds)


def _catalog_delay(attempt: int, header: str | None) -> float | None:
    """Seconds to wait after this 429. None means this attempt is the last one."""
    if attempt >= len(_CATALOG_FALLBACK_SECONDS):
        return None
    if header is not None:
        try:
            parsed = float(header)
        except ValueError:
            parsed = None
        if parsed is not None and parsed >= 0 and parsed != float("inf"):
            return parsed
    return _CATALOG_FALLBACK_SECONDS[attempt]


def _budget(context: object) -> _Budget:
    getter = getattr(context, "get_remaining_time_in_millis", None)
    if not callable(getter):
        return lambda: 180_000

    def read() -> int:
        value = getter()
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
        return 180_000

    return read


_SEALED = date(2026, 10, 6)
_CATALOG_RESERVE_SECONDS = 15.0
_CATALOG_PACE_SECONDS = 2.0
_CATALOG_FALLBACK_SECONDS = (30.0, 45.0, 60.0)
_Budget = Callable[[], int]
_TRADING = re.compile(r"^[A-Z0-9]{1,20}$")
_DEPTH_LIMIT = 100
_PEG_LIMIT = 30
_DAILY_LIMIT = 40
_OPEN_KINDS = frozenset(
    {"universe", "ticker", "peg", "regime", "book", "classification", "listing"}
)
_BAR_KINDS = frozenset({"book", "daily_bars", "hour_bars"})


class _Spot(Protocol):
    def exchange_info(self) -> ExchangeInfo: ...

    def ticker_24hr(self) -> tuple[Ticker24h, ...]: ...

    def klines(self, symbol: str, *, interval: str, limit: int) -> tuple[Kline, ...]: ...

    def depth(self, symbol: str, *, limit: int) -> Depth: ...


class _Closeable(Protocol):
    def close(self) -> None: ...


_Reader = Callable[[str, dict[str, str] | None], object]


class EvidenceSource:
    """Public market data for one capture cycle. No order endpoint is called."""

    def __init__(
        self,
        spot: _Spot,
        now: datetime,
        depth_band: Decimal,
        base_url: str,
        reader: _Reader,
        remaining: _Budget | None = None,
    ) -> None:
        _utc(now)
        if not base_url.startswith("https://"):
            raise MarketDataError("market data base URL must use https")
        self._spot = spot
        self.now = now
        self._band = depth_band
        self._base = base_url.rstrip("/")
        self._reader = reader
        self.calls: list[str] = []
        self.catalog_trace: list[dict[str, object]] = []
        self._remaining = remaining or (lambda: 180_000)
        self._info: ExchangeInfo | None = None
        self._rows: dict[str, SymbolClassification] | None = None
        self._pages: dict[str, object] = {}
        self._root: Path | None = None
        self._catalog_unavailable = False

    def bind_store(self, root: Path) -> None:
        """Remember the session store so a finished source survives this process."""
        self._root = root

    def fetch(self, session: date, kind: str, symbol: str | None) -> object:
        """One observation. A temporary failure is raised with nothing stored."""
        if kind in _OPEN_KINDS and self.now >= session_close(session):
            raise TemporaryFailure("capture is after the close")
        try:
            return self._read(session, kind, symbol)
        except TemporaryFailure:
            raise
        except CatalogIncomplete as error:
            raise TemporaryFailure("classification catalog is incomplete") from error
        except ExchangeBannedError:
            raise
        except RecorderError as error:
            if str(error) == "status 418":
                raise ExchangeBannedError("Binance returned 418; the scan must stop") from error
            raise TemporaryFailure(str(error)) from error
        except (MarketDataError, httpx.HTTPError) as error:
            raise TemporaryFailure(str(error)) from error

    def _read(self, session: date, kind: str, symbol: str | None) -> object:
        if kind == "universe":
            self.calls.append("GET /api/v3/exchangeInfo")
            self._info = self._spot.exchange_info()
            return universe_capture(session, self._info, self.now, self.now)
        if kind == "classification":
            return self._classification(session, symbol)
        if kind == "ticker":
            self.calls.append("GET /api/v3/ticker/24hr")
            rows = [
                {"quote_volume": format(item.quote_volume, "f"), "symbol": item.symbol}
                for item in self._spot.ticker_24hr()
            ]
            return (rows, None, self.now)
        if kind == "peg":
            if symbol != "USDCUSDT":
                raise EvaluationError("capture symbol is unusable")
            return self._peg(session)
        if kind == "regime":
            return self._regime(session, symbol)
        if kind == "listing":
            return self._listing(session, _trading_symbol(symbol))
        if kind in _BAR_KINDS:
            return self._named(session, kind, _trading_symbol(symbol))
        raise TemporaryFailure("capture kind is not collected")

    def _classification(
        self, session: date, symbol: str | None
    ) -> tuple[SymbolClassification, datetime]:
        if symbol is None or _TRADING.fullmatch(symbol) is None:
            raise EvaluationError("capture symbol is unusable")
        if self._catalog_unavailable:
            raise TemporaryFailure("classification catalog is incomplete")
        if self._rows is None:
            pairs = [
                (item.symbol, item.base_asset)
                for item in self._exchange_info().symbols
                if item.quote_asset == "USDT" and item.base_asset != ""
            ]
            try:
                produced = self._produce(session, pairs)
            except CatalogIncomplete:
                self._catalog_unavailable = True
                raise
            if produced is None:
                self._catalog_unavailable = True
                raise TemporaryFailure("classification catalog is incomplete")
            self._rows = {item.symbol: item for item in produced}
        found = self._rows.get(symbol)
        if found is None:
            raise TemporaryFailure("classification base is unresolved")
        return found, self.now

    def _produce(
        self, session: date, pairs: list[tuple[str, str]]
    ) -> tuple[SymbolClassification, ...] | None:
        if self._root is None:
            return produce_classifications(pairs, self._get)
        return ensure_bundle(self._root, session, self.now, pairs, self._get)

    def _exchange_info(self) -> ExchangeInfo:
        if self._info is None:
            self.calls.append("GET /api/v3/exchangeInfo")
            self._info = self._spot.exchange_info()
        return self._info

    def _get(self, url: str) -> object:
        cached = self._pages.get(url)
        if cached is not None:
            return cached
        if "coingecko.com" in url and not self._wait(url, 0, _CATALOG_PACE_SECONDS, None, "pace"):
            return self._abandon(url, 0, None, RecorderError("status 429"))
        return self._attempt(url, 0)

    def _attempt(self, url: str, attempt: int) -> object:
        try:
            self.calls.append(f"GET {url}")
            payload = self._reader(url, None)
        except RecorderError as error:
            if str(error) != "status 429":
                self._trace(url, attempt, "error", None, None, "failed")
                raise
            header = error.retry_after if isinstance(error, RateLimited) else None
            delay = _catalog_delay(attempt, header)
            if delay is None or not self._wait(url, attempt, delay, header, "retry"):
                return self._abandon(url, attempt, header, error)
            return self._attempt(url, attempt + 1)
        self._pages[url] = payload
        self._trace(url, attempt, "200", None, None, "succeeded")
        return payload

    def _trace(
        self,
        url: str,
        attempt: int,
        status: str,
        header: str | None,
        delay: float | None,
        outcome: str,
    ) -> None:
        event: dict[str, object] = {
            "attempt": attempt + 1,
            "delay_seconds": delay,
            "outcome": outcome,
            "page": url,
            "remaining_ms": self._remaining(),
            "retry_after": header,
            "status": status,
        }
        self.catalog_trace.append(event)
        logger.info("classification catalog request", extra=event)

    def _wait(
        self,
        url: str,
        attempt: int,
        delay: float,
        header: str | None,
        kind: str,
    ) -> bool:
        remaining = self._remaining()
        if remaining / 1000 < delay + _CATALOG_RESERVE_SECONDS:
            self._trace(url, attempt, "429", header, delay, "budget")
            return False
        self._trace(url, attempt, "429" if kind == "retry" else "pace", header, delay, kind)
        _pause(delay)
        return True

    def _abandon(self, url: str, attempt: int, header: str | None, error: RecorderError) -> object:
        self._catalog_unavailable = True
        self._trace(url, attempt, "429", header, None, "abandoned")
        raise error

    def _peg(self, session: date) -> tuple[tuple[Decimal, ...], datetime, datetime]:
        self.calls.append("GET /api/v3/klines USDCUSDT 1h")
        candles = self._spot.klines("USDCUSDT", interval="1h", limit=_PEG_LIMIT)
        close_ms = _ms(session_close(session))
        now_ms = _ms(self.now)
        closed = [
            candle
            for candle in candles
            if candle.close_time < close_ms and candle.close_time < now_ms
        ]
        if not closed:
            raise TemporaryFailure("peg hour is still open")
        latest = max(candle.close_time for candle in closed)
        source_timestamp = datetime.fromtimestamp(latest / 1000, UTC)
        return (tuple(candle.close for candle in closed), source_timestamp, self.now)

    def _regime(self, session: date, series: str | None) -> RegimeCapture:
        if series == "btc_dominance":
            self.calls.append("GET https://api.coingecko.com/api/v3/global")
            payload = self._reader(COINGECKO_GLOBAL, None)
            observation = parse_btc_dominance(payload, observed_at=self.now)
        elif series == "stablecoin_supply":
            self.calls.append("GET https://stablecoins.llama.fi/stablecoincharts/all")
            payload = self._reader(DEFILLAMA_STABLECOINS, None)
            observation = parse_stablecoin_supply(payload, observed_at=self.now)
        else:
            raise EvaluationError("regime series is not a readiness input")
        source_timestamp = observation.source_timestamp
        if source_timestamp is not None and source_timestamp >= session_close(session):
            raise TemporaryFailure("capture is after the close")
        return RegimeCapture(observation, self.now)

    def _named(self, session: date, kind: str, symbol: str) -> object:
        if kind == "book":
            self.calls.append(f"GET /api/v3/depth {symbol}")
            book = self._spot.depth(symbol, limit=_DEPTH_LIMIT)
            reading = book_metrics(book, band=self._band, observed_at=self.now)
            if reading is None:
                raise TemporaryFailure("book is empty")
            return BookShot(
                reading.spread_bps,
                None,
                self.now,
                tuple((level.price, level.quantity) for level in book.bids),
                tuple((level.price, level.quantity) for level in book.asks),
            )
        if kind == "daily_bars":
            return self._daily(session, symbol)
        return self._hours(session, symbol)

    def _listing(self, session: date, symbol: str) -> object:
        """The earliest daily open at or before this session. Later candles are excluded."""
        close = session_close(session)
        self.calls.append(f"GET /api/v3/klines {symbol} listing")
        return self._reader(
            f"{self._base}/api/v3/klines",
            {
                "endTime": str(_ms(close) - 1),
                "interval": "1d",
                "limit": "1",
                "startTime": "0",
                "symbol": symbol,
            },
        )

    def _daily(self, session: date, symbol: str) -> tuple[DailyBar, ...]:
        close = session_close(session)
        self.calls.append(f"GET /api/v3/klines {symbol} 1d")
        payload = self._reader(
            f"{self._base}/api/v3/klines",
            {
                "endTime": str(_ms(close) - 1),
                "interval": "1d",
                "limit": str(_DAILY_LIMIT),
                "startTime": str(_ms(close - timedelta(days=_DAILY_LIMIT))),
                "symbol": symbol,
            },
        )
        bars = tuple(
            bar for bar in parse_daily_klines(payload, symbol=symbol) if bar.open_date <= session
        )
        if not any(bar.open_date == session for bar in bars):
            raise TemporaryFailure("session bar is missing")
        return bars

    def _hours(self, session: date, symbol: str) -> tuple[HourQuote, ...]:
        opened = datetime.combine(session, time.min, tzinfo=UTC)
        close = session_close(session)
        self.calls.append(f"GET /api/v3/klines {symbol} 1h")
        payload = self._reader(
            f"{self._base}/api/v3/klines",
            {
                "endTime": str(_ms(close) - 1),
                "interval": "1h",
                "limit": "24",
                "startTime": str(_ms(opened)),
                "symbol": symbol,
            },
        )
        return tuple(
            HourQuote(candle.open_time, candle.close_time, candle.quote_volume)
            for candle in _klines(payload)
        )


def capture(
    event: dict[str, Any],
    _context: LambdaContext,
    *,
    now: datetime | None = None,
    source: EvidenceSource | None = None,
    client: Any | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """Capture the session the clock names. A payload date is ignored."""
    moment = _clock() if now is None else now
    _utc(moment)
    _note_supplied(event)
    store = _storage_root() if root is None else root
    _reset(store)
    s3 = _s3() if client is None else client
    bucket = os.environ["DATA_BUCKET"]
    owned: tuple[_Closeable, ...] = ()
    evidence = source
    if evidence is None:
        evidence, owned = _open_source(moment, _budget(_context))
    report: CycleReport | None = None
    error: BaseException | None = None
    written: tuple[str, ...] = ()
    unchanged: tuple[str, ...] = ()
    try:
        original = _download(s3, bucket, store, moment)
        try:
            report = run_capture_cycle(store, moment, evidence, capture_symbols())
        except BaseException as caught:
            error = caught
        written, unchanged = _upload(s3, bucket, store, original)
    finally:
        for item in owned:
            item.close()
    if error is not None:
        logger.exception("session capture failed")
        raise error
    finished = cast(CycleReport, report)
    ready = None if finished.readiness is None else finished.readiness.ready
    blocks = () if finished.readiness is None else finished.readiness.blocks
    body = {
        "blocks": list(blocks),
        "calls": [{"kind": kind, "symbol": symbol} for kind, symbol in finished.calls],
        "finalized": finished.finalized,
        "open_session": finished.cycle.open_session.isoformat(),
        "post_close": None
        if finished.cycle.post_close is None
        else finished.cycle.post_close.isoformat(),
        "pre_close": None
        if finished.cycle.pre_close is None
        else finished.cycle.pre_close.isoformat(),
        "provider_calls": list(evidence.calls),
        "ready": ready,
        "retries": list(finished.retries),
        "skipped": list(finished.skipped),
        "unchanged": list(unchanged),
        "written": list(written),
    }
    logger.info("session capture stored", extra=body)
    return body


def capture_symbols() -> tuple[str, ...]:
    """Symbols that receive a book. Session-level evidence does not need one."""
    raw = os.environ.get("CAPTURE_SYMBOLS", "BTCUSDT")
    symbols = tuple(part.strip() for part in raw.split(",") if part.strip())
    for symbol in symbols:
        _trading_symbol(symbol)
    return symbols


def evidence_prefixes(now: datetime) -> tuple[str, ...]:
    """Object prefixes this clock may read. The sealed session is not among them."""
    _utc(now)
    opened = now.astimezone(UTC).date()
    days: list[date] = []
    if opened != _SEALED and opened > BLOCKED_THROUGH:
        days.append(opened)
    previous = opened - timedelta(days=1)
    if previous != _SEALED and previous > BLOCKED_THROUGH:
        days.append(previous)
    prefixes: list[str] = []
    for day in days:
        iso = day.isoformat()
        prefixes.append(f"captures/session={iso}/")
        prefixes.append(f"sessions/date={iso}/")
    return tuple(prefixes)


def _open_source(
    now: datetime, remaining: _Budget | None = None
) -> tuple[EvidenceSource, tuple[_Closeable, ...]]:
    policy = load_policy(Path(os.environ["POLICY_PATH"]))
    base = policy.policy.venue.market_data_base_url
    public = httpx.Client(timeout=10.0, follow_redirects=False)
    spot = BinanceMarketClient(base)

    def read(url: str, params: dict[str, str] | None) -> object:
        return fetch_json(public, url, params)

    source = EvidenceSource(spot, now, policy.policy.recorders.depth_band, base, read, remaining)
    return source, (public, spot)


def _upload(
    client: Any, bucket: str, root: Path, original: dict[str, bytes]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    pending: list[tuple[str, bytes]] = []
    unchanged: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        key = path.relative_to(root).as_posix()
        if not key.startswith(("captures/", "sessions/")):
            continue
        if _sealed(key):
            raise EvaluationError("sealed session stays sealed")
        body = path.read_bytes()
        previous = original.get(key)
        if previous == body:
            unchanged.append(key)
            continue
        if previous is not None:
            raise EvaluationError("conflicting observation")
        pending.append((key, body))
    written: list[str] = []
    for key, body in pending:
        client.put_object(Bucket=bucket, Key=key, Body=body)
        written.append(key)
    return tuple(written), tuple(unchanged)


def _download(client: Any, bucket: str, root: Path, now: datetime) -> dict[str, bytes]:
    stored: dict[str, bytes] = {}
    keys = [key for prefix in evidence_prefixes(now) for key in _list(client, bucket, prefix)]
    keys.extend(key for key in _history_book_keys(client, bucket, now) if key not in keys)
    for key in keys:
        if _sealed(key):
            continue
        body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
        if not isinstance(body, bytes):
            raise EvaluationError("capture is unusable")
        path = root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        stored[key] = body
    return stored


def _history_book_keys(client: Any, bucket: str, now: datetime) -> tuple[str, ...]:
    """Book objects on or before the open session. Later sessions are not read."""
    _utc(now)
    opened = now.astimezone(UTC).date()
    keys: list[str] = []
    for day in _listed_sessions(client, bucket):
        if day > opened or day <= BLOCKED_THROUGH or day == _SEALED:
            continue
        prefix = f"captures/session={day.isoformat()}/book/"
        keys.extend(key for key in _list(client, bucket, prefix) if key.startswith(prefix))
    return tuple(keys)


def _listed_sessions(client: Any, bucket: str) -> tuple[date, ...]:
    found: list[date] = []
    token: str | None = None
    while True:
        page = client.list_objects_v2(**_page(bucket, "captures/", token, delimiter="/"))
        for item in page.get("CommonPrefixes") or []:
            prefix = str(item.get("Prefix", ""))
            marker = "captures/session="
            if not prefix.startswith(marker) or not prefix.endswith("/"):
                continue
            try:
                found.append(date.fromisoformat(prefix[len(marker) : -1]))
            except ValueError:
                continue
        if not page.get("IsTruncated"):
            return tuple(found)
        token = str(page["NextContinuationToken"])


def _list(client: Any, bucket: str, prefix: str) -> tuple[str, ...]:
    keys: list[str] = []
    token: str | None = None
    while True:
        page = client.list_objects_v2(**_page(bucket, prefix, token))
        keys.extend(str(item["Key"]) for item in page.get("Contents") or [])
        if not page.get("IsTruncated"):
            return tuple(keys)
        token = str(page["NextContinuationToken"])


def _page(
    bucket: str, prefix: str, token: str | None, *, delimiter: str | None = None
) -> dict[str, str]:
    params = {"Bucket": bucket, "Prefix": prefix}
    if token is not None:
        params["ContinuationToken"] = token
    if delimiter is not None:
        params["Delimiter"] = delimiter
    return params


def _s3() -> Any:
    return boto3.client("s3")


def _storage_root() -> Path:
    raw = os.environ.get("CAPTURE_ROOT")
    return Path(raw) if raw else Path("/tmp/cip-session-capture")  # noqa: S108


def _reset(root: Path) -> None:
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)


def _note_supplied(event: dict[str, Any]) -> None:
    supplied = event.get("session", event.get("date"))
    if supplied is not None:
        logger.info("supplied session ignored", extra={"supplied_session": str(supplied)})


def _clock() -> datetime:
    return datetime.now(UTC)


def _utc(moment: datetime) -> None:
    if moment.tzinfo is None or moment.utcoffset() != timedelta(0):
        raise EvaluationError("as_of must be timezone-aware UTC")


def _ms(moment: datetime) -> int:
    return int(moment.timestamp() * 1000)


def _trading_symbol(symbol: str | None) -> str:
    if symbol is None or _TRADING.fullmatch(symbol) is None:
        raise EvaluationError("capture symbol is unusable")
    return symbol


def _sealed(key: str) -> bool:
    return "session=2026-10-06" in key or "date=2026-10-06" in key


def _klines(payload: object) -> tuple[Kline, ...]:
    try:
        return parse_klines(payload)
    except MarketDataError as error:
        raise TemporaryFailure(str(error)) from error
