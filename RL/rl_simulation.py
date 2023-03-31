import logging
from collections import defaultdict

from RL.betfair_env import FlumineEnv

from flumine.utils import SimulatedDateTime
from flumine.baseflumine import BaseFlumine
from flumine.clients import BaseClient
from flumine.events import events
from flumine import utils
from flumine.exceptions import RunError
from flumine.order.order import OrderTypes

logger = logging.getLogger(__name__)


class FlumineRLSimulation(BaseFlumine):
    """
    Single threaded implementation of flumine
    for simulating strategies with betfair
    historic (or self recorded) streaming data.
    """

    SIMULATED = True

    def __init__(self, step_event, step_complete_event, client: BaseClient = None):
        super(FlumineRLSimulation, self).__init__(client)
        self.simulated_datetime = SimulatedDateTime()
        self.handler_queue = []
        self.__set_event_streams()
        self.observation = None
        self.reward = 0
        self.info = None
        self.done = False

    def __set_event_streams(self):
        event_streams = defaultdict(list)  # eventId: [<Stream>, ..]
        for stream in self.streams:
            event_id = stream.event_id if stream.event_processing else None
            event_streams[event_id].append(stream)
        self.event_streams = event_streams

    def run_to_time(self, end_time: float) -> None:
        """
        Advance the simulation up to the specified end time.

        :param end_time: The timestamp up to which to run the simulation.
        """
        if not self.clients.simulated:
            raise RunError(
                "Incorrect client provided, only a Simulated client can be used when simulating"
            )

        with self:
            with self.simulated_datetime.real_time():
                # get all the streams
                event_streams = defaultdict(list)
                for stream in self.streams:
                    event_id = stream.event_id if stream.event_processing else None
                    event_streams[event_id].append(stream)

                # process events up to end time
                while self.simulated_datetime.timestamp() < end_time:
                    for event_id, streams in event_streams.items():
                        if event_id and len(streams) > 1:
                            # start processing historical event
                            logger.info(
                                "Starting historical event '{0}'".format(event_id),
                                extra={
                                    "event_id": event_id,
                                    "markets": [s.market_filter for s in streams],
                                },
                            )
                            self.simulated_datetime.reset_real_datetime()
                            # create cycles
                            cycles = []  # [[epoch, [MarketBook], gen], ..]
                            for stream in streams:
                                stream_gen = stream.create_generator()()
                                market_book = next(stream_gen)
                                publish_time_epoch = market_book[0].publish_time_epoch
                                cycles.append([publish_time_epoch, market_book, stream_gen])
                            # process cycles
                            while cycles:
                                # order by epoch
                                cycles.sort(key=lambda x: x[0])
                                # get current
                                _, market_book, stream_gen = cycles.pop(0)
                                # process current
                                self._process_market_books(
                                    events.MarketBookEvent(market_book)
                                )
                                # gen next
                                try:
                                    market_book = next(stream_gen)
                                except StopIteration:
                                    continue
                                publish_time_epoch = market_book[0].publish_time_epoch
                                # add back
                                cycles.append([publish_time_epoch, market_book, stream_gen])
                            self.handler_queue.clear()
                            logger.info("Completed historical event '{0}'".format(event_id))
                        else:
                            for stream in streams:
                                # start processing historical market
                                logger.info(
                                    "Starting historical market '{0}'".format(
                                        stream.market_filter
                                    ),
                                    extra={"market": stream.market_filter},
                                )
                                self.simulated_datetime.reset_real_datetime()
                                stream_gen = stream.create_generator()()
                                for event in stream.advance_time(
                                    self.simulated_datetime.timestamp(), end_time
                                ):
                                    self._process_market_books(
                                        events.MarketBookEvent(event)
                                    )
                                self.handler_queue.clear()
                                logger.info(
                                    "Completed historical market '{0}'".format(
                                        stream.market_filter
                                    )
                                )

                    # process any remaining orders
                    if self.handler_queue:
                        self._check_pending_packages(None)

                    # move time forward
                    self.simulated_datetime(end_time)

                # process end of simulation
                self._process_end_flumine()
                logger.info("Simulation complete")

    def _process_market_books(self, event: events.MarketBookEvent) -> None:
        # todo DRY!

        for market_book in event.event:
            market_id = market_book.market_id
            self.simulated_datetime(market_book.publish_time)

            # check if there are orders to process (limited to current market only)
            if self.handler_queue:
                self._check_pending_packages(market_id)

            if market_book.status == "CLOSED":
                self._process_close_market(event=events.CloseMarketEvent(market_book))
                continue

            # get market
            market = self.markets.markets.get(market_id)
            if market is None:
                market = self._add_market(market_id, market_book)
                self.log_control(events.MarketEvent(market))
            elif market.closed:
                self.markets.add_market(market_id, market)

            # process market
            market(market_book)

            # process middleware
            for middleware in self._market_middleware:
                utils.call_middleware_error_handling(middleware, market)

            # process current orders
            if market.blotter.active:
                self._process_simulated_orders(market)

            for strategy in self.strategies:
                if utils.call_strategy_error_handling(
                    strategy.check_market, market, market_book
                ):
                    # this is an action (STEP)
                    utils.call_strategy_error_handling(
                        strategy.process_market_book, market, market_book
                    )

    def process_order_package(self, order_package) -> None:
        # place in pending list (wait for latency+delay)
        self.handler_queue.append(order_package)

    def _process_simulated_orders(self, market) -> None:
        """Remove order from blotter live
        orders if complete and process
        orders through strategies
        """
        blotter = market.blotter
        for order in blotter.live_orders:
            if order.complete:
                blotter.complete_order(order)
            else:
                if order.order_type.ORDER_TYPE == OrderTypes.LIMIT:
                    if order.size_remaining == 0:
                        order.execution_complete()
                        blotter.complete_order(order)
                elif order.order_type.ORDER_TYPE in [
                    OrderTypes.LIMIT_ON_CLOSE,
                    OrderTypes.MARKET_ON_CLOSE,
                ]:
                    if order.current_order.status == "EXECUTION_COMPLETE":
                        order.execution_complete()
                        blotter.complete_order(order)
        for strategy in self.strategies:
            strategy_orders = blotter.strategy_orders(strategy)
            if strategy_orders:
                # Get the reward here
                utils.call_process_orders_error_handling(
                    strategy, market, strategy_orders
                )

    def _check_pending_packages(self, market_id: str) -> None:
        processed = []
        for order_package in self.handler_queue:
            if (
                order_package.market_id == market_id
                and order_package.elapsed_seconds > order_package.simulated_delay
            ):
                order_package.client.execution.handler(order_package)
                processed.append(order_package)
        for p in processed:
            self.handler_queue.remove(p)

    def __repr__(self) -> str:
        return "<FlumineSimulation>"

    def __str__(self) -> str:
        return "<FlumineSimulation>"
