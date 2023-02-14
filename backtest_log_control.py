import csv
from utils.constants import FIELD_NAMES
from flumine.controls.loggingcontrols import LoggingControl
from flumine.order.ordertype import OrderTypes


class BacktestLoggingControl(LoggingControl):
    NAME = "BACKTEST_LOGGING_CONTROL"

    def __init__(self, logger, *args, **kwargs):
        super(BacktestLoggingControl, self).__init__(*args, **kwargs)
        self._setup()
        self.logger = logger

    def _setup(self):
        with open("orders.txt", "w") as m:
            csv_writer = csv.DictWriter(m, delimiter=",", fieldnames=FIELD_NAMES)
            csv_writer.writeheader()

    def _process_cleared_orders_meta(self, event):
        orders = event.event
        with open("orders.txt", "a") as m:
            for order in orders:
                if order.order_type.ORDER_TYPE == OrderTypes.LIMIT:
                    size = order.order_type.size
                else:
                    size = order.order_type.liability
                if order.order_type.ORDER_TYPE == OrderTypes.MARKET_ON_CLOSE:
                    price = None
                else:
                    price = order.order_type.price
                try:
                    order_data = {
                        "bet_id": order.bet_id,
                        "strategy_name": order.trade.strategy,
                        "market_id": order.market_id,
                        "selection_id": order.selection_id,
                        "trade_id": order.trade.id,
                        "date_time_placed": order.responses.date_time_placed,
                        "price": price,
                        "price_matched": order.average_price_matched,
                        "size": size,
                        "size_matched": order.size_matched,
                        "profit": order.simulated.profit,
                        "side": order.side,
                        "elapsed_seconds_executable": order.elapsed_seconds_executable,
                        "order_status": order.status.value,
                        "market_note": order.trade.market_notes,
                        "trade_notes": order.trade.notes_str,
                        "order_notes": order.notes_str,
                    }
                    csv_writer = csv.DictWriter(
                        m, delimiter=",", fieldnames=FIELD_NAMES
                    )
                    csv_writer.writerow(order_data)
                except Exception as e:
                    self.logger.error(
                        "_process_cleared_orders_meta: %s" % e,
                        extra={"order": order, "error": e},
                    )

        self.logger.info("Orders updated", extra={"order_count": len(orders)})

    def _process_cleared_markets(self, event):
        cleared_markets = event.event
        for cleared_market in cleared_markets.orders:
            self.logger.info(
                "Cleared market",
                extra={
                    "market_id": cleared_market.market_id,
                    "bet_count": cleared_market.bet_count,
                    "profit": cleared_market.profit,
                    "commission": cleared_market.commission,
                },
            )
