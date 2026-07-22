"""Kraken broker implementation, backed by the low-level krakenex SDK
(krakenex only wraps signing/nonce — response shapes match Kraken's REST
API docs directly, not a typed client).

No verified sandbox/paper mode — every linked Kraken account always
connects to the real live API (accounts.py forces is_paper=False for this
broker). Spot trading only; `ticker` must be a Kraken pair like "XBTUSD".

Position market values aren't computed: Kraken's asset-code system (XXBT,
XETH, ZUSD, ...) doesn't map onto trading pairs in a small, reliably-correct
way without a full pair-lookup table, so get_positions() reports quantity
only rather than guess at a price. Account-level equity in
get_account_snapshot is exact — Kraken computes it server-side (TradeBalance
"eb" field), no guessing involved there.

Bracket orders (take_profit_price/stop_loss_price) aren't implemented.
"""

from __future__ import annotations

import krakenex

from backtester.brokers.base import AccountSnapshot, BrokerAccount, EquityPoint, OrderResult, OrderSide, Position

FIAT_ASSETS = {"ZUSD", "ZEUR", "ZGBP", "ZCAD", "ZJPY", "ZCHF", "ZAUD"}


class KrakenError(RuntimeError):
    pass


class KrakenBroker(BrokerAccount):
    def __init__(self, nickname: str, api_key: str, api_secret: str):
        self.nickname = nickname
        self.is_paper = False  # no verified sandbox for this broker
        self._client = krakenex.API(key=api_key, secret=api_secret)

    def _private(self, method: str, data: dict | None = None) -> dict:
        response = self._client.query_private(method, data)
        if response.get("error"):
            raise KrakenError("; ".join(response["error"]))
        return response.get("result", {})

    def get_account_snapshot(self) -> AccountSnapshot:
        balance = self._private("Balance")
        trade_balance = self._private("TradeBalance")
        cash = float(balance.get("ZUSD", 0.0))
        equity = float(trade_balance.get("eb", cash))
        return AccountSnapshot(account_id=self.nickname, equity=equity, cash=cash, buying_power=cash, is_paper=False)

    def get_positions(self) -> list[Position]:
        balance = self._private("Balance")
        positions = []
        for asset, qty_str in balance.items():
            qty = float(qty_str)
            if asset in FIAT_ASSETS or qty == 0:
                continue
            positions.append(
                Position(
                    ticker=asset,
                    qty=qty,
                    side="long",
                    avg_entry_price=0.0,
                    current_price=None,
                    market_value=0.0,
                    unrealized_pl=0.0,
                )
            )
        return positions

    def get_equity_history(self, period: str = "1M", timeframe: str = "1D") -> list[EquityPoint]:
        raise NotImplementedError(
            "Kraken's API doesn't expose a historical portfolio-equity time series through "
            "this client — only current balances (see get_account_snapshot)."
        )

    def submit_market_order(
        self,
        ticker: str,
        side: OrderSide,
        qty: float,
        take_profit_price: float | None = None,
        stop_loss_price: float | None = None,
    ) -> OrderResult:
        if take_profit_price is not None or stop_loss_price is not None:
            return OrderResult(
                account_nickname=self.nickname,
                success=False,
                error="Bracket orders (take-profit/stop-loss) aren't implemented for Kraken yet.",
            )
        try:
            result = self._private(
                "AddOrder",
                {
                    "pair": ticker,
                    "type": "buy" if side is OrderSide.BUY else "sell",
                    "ordertype": "market",
                    "volume": str(qty),
                },
            )
            txids = result.get("txid", [])
            return OrderResult(
                account_nickname=self.nickname,
                success=True,
                broker_order_id=txids[0] if txids else None,
            )
        except Exception as e:  # noqa: BLE001
            return OrderResult(account_nickname=self.nickname, success=False, error=str(e))
