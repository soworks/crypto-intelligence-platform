"""Fetch the catalogs the classification contract already knows how to read.

Stablecoin and wrapped exclusions stay on the policy lists. This module does
not invent a flag the parsers leave missing, and it does not call a scan.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from cip.domain.errors import EvaluationError
from cip.evaluation.classification import (
    CapReading,
    SymbolClassification,
    classify_universe,
    map_binance_tickers,
    map_cmc_ids,
    parse_asset_catalog,
    parse_coin_config,
    parse_coingecko_markets,
)

_ASSET = "https://www.binance.com/bapi/asset/v2/public/asset/asset/get-all-asset"
_COINS = "https://www.binance.com/bapi/capital/v1/public/capital/getNetworkCoinAll"
_TICKERS = "https://api.coingecko.com/api/v3/exchanges/binance/tickers?page={page}"
_EUR = (
    "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd"
    "&category=eur-stablecoin&per_page=250&page={page}"
)
_MARKETS = "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&per_page=250&ids="
_MARKETING = "https://www.binance.com/bapi/composite/v1/public/marketing/symbol/list"
_Reader = Callable[[str], object]
_PAGE_SPEC = {
    "tickers": (_TICKERS, "tickers", 100),
    "eur": (_EUR, None, 250),
}


class CatalogIncomplete(Exception):
    """A page ended before it could be stored. Earlier pages of this session stay."""


def produce_classifications(
    pairs: Sequence[tuple[str, str]],
    get: _Reader,
) -> tuple[SymbolClassification, ...]:
    """Join one complete retrieval onto the captured pairs. An incomplete page stores nothing."""
    assets = parse_asset_catalog(get(_ASSET))
    coins = parse_coin_config(get(_COINS))
    mapping = map_binance_tickers(_pages(get, _TICKERS, "tickers", 100))
    eur_ids = frozenset(
        row["id"]
        for row in _pages(get, _EUR, None, 250)
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    )
    wanted = sorted({mapping.ids[symbol] for symbol, _base in pairs if symbol in mapping.ids})
    marketing = get(_MARKETING)
    rows = marketing.get("data") if isinstance(marketing, dict) else None
    if not isinstance(rows, list):
        raise EvaluationError("capture is unusable")
    return classify_universe(
        pairs,
        assets=assets,
        coins=coins,
        mapping=mapping,
        eur_ids=eur_ids,
        caps=_caps(get, wanted),
        cmc=map_cmc_ids(rows),
        cmc_caps={},
    )


def _pages(get: _Reader, url: str, key: str | None, page_size: int) -> list[object]:
    rows: list[object] = []
    for page in range(1, 41):
        payload = get(url.format(page=page))
        batch = payload.get(key) if key is not None and isinstance(payload, dict) else payload
        if not isinstance(batch, list):
            raise CatalogIncomplete("classification catalog is incomplete")
        rows.extend(batch)
        if len(batch) < page_size:
            return rows
    raise CatalogIncomplete("classification catalog is incomplete")


def fetch_page(name: str, get: _Reader, page: int) -> tuple[list[object], int]:
    """One catalog page. A body that is not a list is not a page."""
    spec = _PAGE_SPEC.get(name)
    if spec is None or page < 1:
        raise EvaluationError("classification source is unknown")
    template, key, size = spec
    payload = get(template.format(page=page))
    batch = payload.get(key) if key is not None and isinstance(payload, dict) else payload
    if not isinstance(batch, list) or len(batch) > size:
        raise CatalogIncomplete("classification catalog is incomplete")
    return batch, size


def fetch_component(
    name: str,
    get: _Reader,
    pairs: Sequence[tuple[str, str]],
    prior: dict[str, object],
) -> object:
    """One finished non-paginated source. Ticker and euro pages are stored individually."""
    if name == "assets":
        return get(_ASSET)
    if name == "coins":
        return get(_COINS)
    if name == "marketing":
        payload = get(_MARKETING)
        rows = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            raise EvaluationError("capture is unusable")
        return payload
    if name == "markets":
        tickers = prior.get("tickers")
        if not isinstance(tickers, list):
            raise CatalogIncomplete("classification catalog is incomplete")
        mapping = map_binance_tickers(tickers)
        wanted = sorted({mapping.ids[symbol] for symbol, _base in pairs if symbol in mapping.ids})
        found: list[object] = []
        for start in range(0, len(wanted), 200):
            chunk = wanted[start : start + 200]
            payload = get(_MARKETS + ",".join(chunk))
            if not isinstance(payload, list):
                raise CatalogIncomplete("classification catalog is incomplete")
            found.extend(payload)
        returned = {
            row.get("id")
            for row in found
            if isinstance(row, dict) and isinstance(row.get("id"), str)
        }
        if any(asset_id not in returned for asset_id in wanted):
            raise CatalogIncomplete("classification catalog is incomplete")
        return found
    raise EvaluationError("classification source is unknown")


def classify_stored(
    pairs: Sequence[tuple[str, str]], sources: dict[str, object]
) -> tuple[SymbolClassification, ...]:
    """Join a complete stored bundle. This does not fetch."""
    tickers = sources.get("tickers")
    eur_rows = sources.get("eur")
    marketing = sources.get("marketing")
    markets = sources.get("markets")
    if not isinstance(tickers, list) or not isinstance(eur_rows, list):
        raise EvaluationError("classification source is incomplete")
    if not isinstance(markets, list):
        raise EvaluationError("classification source is incomplete")
    rows = marketing.get("data") if isinstance(marketing, dict) else None
    if not isinstance(rows, list):
        raise EvaluationError("capture is unusable")
    mapping = map_binance_tickers(tickers)
    eur_ids = frozenset(
        row["id"] for row in eur_rows if isinstance(row, dict) and isinstance(row.get("id"), str)
    )
    return classify_universe(
        pairs,
        assets=parse_asset_catalog(sources.get("assets")),
        coins=parse_coin_config(sources.get("coins")),
        mapping=mapping,
        eur_ids=eur_ids,
        caps=parse_coingecko_markets(markets),
        cmc=map_cmc_ids(rows),
        cmc_caps={},
    )


def _caps(get: _Reader, asset_ids: Sequence[str]) -> dict[str, CapReading]:
    found: dict[str, CapReading] = {}
    for start in range(0, len(asset_ids), 200):
        chunk = asset_ids[start : start + 200]
        found.update(parse_coingecko_markets(get(_MARKETS + ",".join(chunk))))
    return found
