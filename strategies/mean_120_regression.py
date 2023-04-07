from typing import Dict, Tuple, Union
import numpy as np
import pandas as pd

from flumine import BaseStrategy
from flumine.order.trade import Trade
from flumine.order.order import (
    LimitOrder,
    MarketOnCloseOrder,
    OrderStatus,
    BaseOrder,
)
from flumine.markets.market import Market
from betfairlightweight.resources import MarketBook, RunnerBook
from sklearn.preprocessing import StandardScaler
from utils.constants import KELLY_PERCENT, TIME_BEFORE_START
from utils.utils import (
    calculate_gambled,
    calculate_kelly_stake,
    calculate_margin,
    calculate_odds,
    normalized_transform,
    preprocess_test_analysis,
)


class Mean120Regression(BaseStrategy):
    def __init__(
        self,
        scaler: StandardScaler,
        ticks_df: pd.DataFrame,
        test_analysis_df: pd.DataFrame,
        balance,
        model,
        clm,
        green_enabled=False,
        *args,
        **kwargs,
    ):
        self.scaler = scaler
        self.ticks_df = ticks_df
        self.model = model
        self.clm = clm
        self._set_and_preprocess_test_analysis(test_analysis_df)
        self.metrics = {
            "profit": 0,
            "q_correct": 0,
            "q_incorrect": 0,
            "matched_correct": 0,
            "matched_incorrect": 0,
            "m_c_margin": 0,
            "m_i_margin": 0,
            "green_margin": 0,
            "amount_gambled": 0,
            "lay_matched_correct": 0,
            "lay_matched_incorrect": 0,
            "back_matched_correct": 0,
            "back_matched_incorrect": 0,
            "q_margin": 0,
        }
        self.green_enabled = green_enabled
        self.regression = True
        self.balance = balance
        self.back_bet_tracker = {}
        self.matched_back_bet_tracker = {}
        self.lay_bet_tracker = {}
        self.matched_lay_bet_tracker = {}
        self.seconds_to_start = None
        self.market_open = True
        # self.stake = 50
        self.first_nonrunners = True
        self.runner_number = None

        super_kwargs = kwargs.copy()
        super_kwargs.pop("scaler", None)
        super_kwargs.pop("ticks_df", None)
        super_kwargs.pop("test_analysis_df", None)
        super_kwargs.pop("model", None)
        super_kwargs.pop("clm", None)
        super_kwargs.pop("balance", None)
        super_kwargs.pop("green_enabled", None)
        super().__init__(*args, **super_kwargs)

    def start(self) -> None:
        pass

    def _set_and_preprocess_test_analysis(self, test_analysis_df):
        test_analysis_df, test_analysis_df_y = preprocess_test_analysis(
            test_analysis_df
        )
        self.test_analysis_df = test_analysis_df
        self.test_analysis_df_y = test_analysis_df_y

    def _set_market_id_bet_trackers(self, market_id: float):
        if market_id not in self.back_bet_tracker.keys():
            self.back_bet_tracker[market_id] = {}
            self.matched_back_bet_tracker[market_id] = {}
        if market_id not in self.lay_bet_tracker.keys():
            self.lay_bet_tracker[market_id] = {}
            self.matched_lay_bet_tracker[market_id] = {}

    def _should_skip_runner(self, market_id: float, runner: RunnerBook):
        """Determines whether or not to skip processing a runner based on various conditions.

        Args:
            market_id (float): The ID of the market.
            runner (RunnerBook): The runner to process.

        Returns:
            bool: True if the runner should be skipped, False otherwise.
        """
        if (
            not runner.status == "ACTIVE"
            or not self.test_analysis_df["selection_ids"]
            .isin([runner.selection_id])
            .any()
            or not self.test_analysis_df["market_id"].isin([market_id]).any()
            or runner.selection_id in self.back_bet_tracker[market_id].keys()
            or runner.selection_id in self.lay_bet_tracker[market_id].keys()
        ):
            return True
        return False

    def process_fundamentals(self, market_book: MarketBook):
        """
        Extracts the necessary information from the market book to be used by the strategy.

        Args:
            market_book (MarketBook): The current market book for the market being monitored.

        Returns:
            bool: Always returns True.

        Sets:
            self.seconds_to_start (float): The number of seconds until the market start time.
        """
        seconds_to_start = (
            market_book.market_definition.market_time - market_book.publish_time
        ).total_seconds()
        self.seconds_to_start = seconds_to_start

        return True

    def check_market_book(self, market: Market, market_book: MarketBook) -> bool:
        """
        Checks whether the market book should be processed based on certain conditions
        (process_market_book only runs if this returns True).


        Args:
            market (Market): The market object.
            market_book (MarketBook): The market book object.

        Returns:
            bool: True if the market book should be processed, False otherwise.
        """
        _ = self.process_fundamentals(market_book)

        if self.first_nonrunners:
            self.runner_number = market_book.number_of_active_runners
            self.first_nonrunners = False
        else:
            if market_book.number_of_active_runners != self.runner_number:
                return False  # this will stop any more action happening in this market.

        if market_book.status == "CLOSED" or self.seconds_to_start < TIME_BEFORE_START:
            return False
        else:
            return True

    def process_market_book(self, market: Market, market_book: MarketBook) -> None:
        """
        Processes the market book and sends bets based on the current state of the market.

        Args:
            market (Market): The current market.
            market_book (MarketBook): The current market book.
        Returns:
            None
        """
        market_id = float(market_book.market_id)
        self._set_market_id_bet_trackers(market_id)

        if not (
            self.seconds_to_start > 100
            and self.seconds_to_start < 120
            # and market_book.inplay  # maybe keep this
        ):
            return

        for runner in market_book.runners:
            if self._should_skip_runner(market_id, runner):
                continue
            (
                runner_predicted_bsp,
                mean_120,
                std_120,
                vol_120,
            ) = self._get_model_prediction_and_features(runner, market_id)

            if not mean_120 or not runner_predicted_bsp or not std_120 or not vol_120:
                continue

            (
                back_price_adjusted,
                back_confidence_price,
                back_bsp_value,
            ) = self._get_adjusted_prices2(
                market_id=market_id, runner=runner, mean_120=mean_120, side="BACK"
            )

            (
                lay_price_adjusted,
                lay_confidence_price,
                lay_bsp_value,
            ) = self._get_adjusted_prices2(
                market_id=market_id, runner=runner, mean_120=mean_120, side="LAY"
            )

            # (
            #     back_price_adjusted,
            #     back_confidence_price,
            #     back_bsp_value,
            # ) = self._get_adjusted_prices(
            #     market_id=market_id,
            #     runner=runner,
            #     mean_120=mean_120,
            #     std_120=std_120,
            #     vol_120=vol_120,
            #     side="BACK",
            # )

            # (
            #     lay_price_adjusted,
            #     lay_confidence_price,
            #     lay_bsp_value,
            # ) = self._get_adjusted_prices(
            #     market_id=market_id,
            #     runner=runner,
            #     mean_120=mean_120,
            #     std_120=std_120,
            #     vol_120=vol_120,
            #     side="LAY",
            # )

            # print("PRICES ADJUSTED", back_price_adjusted, lay_price_adjusted)
            # print("CONFIDENCE PRICES", back_confidence_price, lay_confidence_price)
            # print("BSPS", back_bsp_value, lay_bsp_value)
            # print("MEAN120", mean_120)

            # here is action back
            if (
                runner_predicted_bsp < back_confidence_price
                and mean_120 <= 50
                and mean_120 > 1.1
                and runner.selection_id not in self.back_bet_tracker[market_id].keys()
            ):
                print(
                    "back predicted",
                    runner_predicted_bsp,
                    "back confidence",
                    back_confidence_price,
                )
                self._create_order(
                    market_id,
                    runner,
                    back_confidence_price,
                    mean_120,
                    back_price_adjusted,
                    back_bsp_value,
                    market,
                    side="BACK",
                )

            # here is action lay
            if (
                runner_predicted_bsp > lay_confidence_price
                # and lay_price_adjusted <= self.stake
                and lay_price_adjusted > 1.1
                and runner.selection_id not in self.lay_bet_tracker[market_id].keys()
            ):
                print(
                    "lay predicted",
                    runner_predicted_bsp,
                    "lay confidence",
                    lay_confidence_price,
                )

                self._create_order(
                    market_id,
                    runner,
                    lay_confidence_price,
                    mean_120,
                    lay_price_adjusted,
                    lay_bsp_value,
                    market,
                    side="LAY",
                )

    def process_orders(self, market: Market, orders: list) -> None:
        """Processes orders that have been placed in the market.

        Args:
            market (Market): The market object associated with the orders.
            orders (list): A list of Order objects that have been placed in the market.

        Returns:
            None
        """
        sides = ["BACK", "LAY"]

        for side in sides:
            tracker = self._get_tracker(side)
            matched_tracker = self._get_matched_tracker(side)

            if not len(tracker.keys()) > 0:
                print(f"No bets have been made for side {side} and market {market}")
                continue

            for market_id in tracker.keys():
                for selection_id in tracker[market_id].keys():
                    if len(tracker[market_id][selection_id]) == 0:
                        print(
                            f"No bets have been made for side {side} and runner id {selection_id} in market id {market_id}"
                        )
                        continue

                    # Now we process orders we have made
                    self._process_order_for_selection(
                        market,
                        side,
                        tracker,
                        matched_tracker,
                        market_id,
                        selection_id,
                    )

    def _process_order_for_selection(
        self,
        market: Market,
        side: str,
        tracker: dict,
        matched_tracker: dict,
        market_id: str,
        selection_id: str,
    ) -> None:
        side_mc_key = f"{side.lower()}_matched_correct"
        side_mi_key = f"{side.lower()}_matched_incorrect"

        order = tracker[market_id][selection_id][0]
        price = tracker[market_id][selection_id][-1]
        bsp_value = tracker[market_id][selection_id][1]

        margin = calculate_margin(side, order.size_matched, price, bsp_value)
        gambled = calculate_gambled(side, order.size_matched, price)
        # print("order", order.size_matched, "margin", margin, "gambled", gambled)

        # Order has been fully matched
        if order.status == OrderStatus.EXECUTION_COMPLETE:
            if matched_tracker[market_id][selection_id]:
                return
            matched_tracker[market_id][selection_id] = True
            self._process_matched_order(
                market,
                side,
                tracker,
                market_id,
                selection_id,
                order,
                side_mc_key,
                side_mi_key,
                price,
                bsp_value,
                margin,
                gambled,
            )

        # Order that has a remaining unmatched portion
        elif (
            order.status == OrderStatus.EXECUTABLE
            and order.size_matched > 0
            and self.seconds_to_start < TIME_BEFORE_START
            # and not matched_tracker[market_id][selection_id]
        ):
            self.metrics["amount_gambled"] += gambled
            self._update_metrics(
                side, price, bsp_value, margin, side_mc_key, side_mi_key
            )
            market.cancel_order(order)
            matched_tracker[market_id][selection_id] = True
            print(f"Order cancelled: {order}")

    def _process_matched_order(
        self,
        market: Market,
        side: str,
        tracker: dict,
        market_id: str,
        selection_id: str,
        order: BaseOrder,
        side_mc_key: str,
        side_mi_key: str,
        price: float,
        bsp_value: float,
        margin: float,
        gambled: float,
    ) -> None:
        """
        Process a matched order, updating relevant trackers and metrics.

        Args:
            market (Market): The market the order was placed on.
            side (str): The side of the bet, either 'BACK' or 'LAY'.
            tracker (dict): The tracker containing the orders for the side and market.
            market_id (str): The ID of the market.
            selection_id (str): The ID of the selection.
            order (BaseOrder): The order that was matched.
            side_mc_key (str): The key to the matched correct side metric.
            side_mi_key (str): The key to the matched incorrect side metric.
            price (float): The price of the bet.
            bsp_value (float): The Betfair Starting Price value.
            margin (float): The calculated margin.
            gambled (float): The amount gambled.

        Returns:
            None
        """

        if (order.size_matched >= 10.00 and side == "BACK") or (
            order.size_matched > 1 and side == "LAY"
        ):
            print(
                f"Order matched for side: {side}, market_id: {market_id}, selection_id: {selection_id}, order: {order}, price: {price}, margin: {margin}, bsp_value: {bsp_value}, gambled: {gambled}"
            )

            # TODO ensure this is correct
            if self.green_enabled:
                self._green_up(market, side, order, tracker, market_id, selection_id)
                self.metrics["green_margin"] += margin

            self._update_metrics(
                side, price, bsp_value, margin, side_mc_key, side_mi_key
            )
            self.metrics["amount_gambled"] += gambled

    def _green_up(
        self,
        market: Market,
        side: str,
        order: BaseOrder,
        tracker: dict,
        market_id: str,
        selection_id: str,
    ) -> None:
        """
        Greens up a matched bet by placing an opposing bet.

        Args:
            market (Market): The market object.
            side (str): The side of the bet ("BACK" or "LAY").
            order (BaseOrder): The matched order.
            tracker (dict): The tracker containing the orders for the side and market.
            market_id (str): The ID of the market.
            selection_id (str): The ID of the selection.

        Returns:
            None
        """
        selection_id_ = tracker[market_id][selection_id][3]
        handicap_ = tracker[market_id][selection_id][4]
        market_id_ = tracker[market_id][selection_id][2]
        trade = Trade(
            market_id=market_id_,
            selection_id=selection_id_,
            handicap=handicap_,
            strategy=self,
        )
        opposite_side = "LAY" if side == "BACK" else "BACK"
        order = trade.create_order(
            side=opposite_side,
            order_type=MarketOnCloseOrder(liability=order.size_matched),
        )
        market.place_order(order)
        print("Greened", order)

    def _update_metrics(
        self,
        side: str,
        price: float,
        bsp_value: float,
        margin: float,
        matched_correct_key: str,
        matched_incorrect_key: str,
    ) -> None:
        """
        Updates the metrics based on the side, price, BSP value and margin.

        Args:
            side (str): The side of the bet ("BACK" or "LAY").
            price (float): The price of the bet.
            bsp_value (float): The BSP value.
            margin (float): The margin of the bet.
            matched_correct_key (str): The key for the matched correct metric.
            matched_incorrect_key (str): The key for the matched incorrect metric.

        Returns:
            None
        """
        if (price > bsp_value and side == "BACK") or (
            price < bsp_value and side == "LAY"
        ):
            print("matched_correct", margin)
            self.metrics["matched_correct"] += 1
            self.metrics[matched_correct_key] += 1
            self.metrics["m_c_margin"] += margin
        else:
            print("incorrect", margin)
            self.metrics["matched_incorrect"] += 1
            self.metrics[matched_incorrect_key] += 1
            self.metrics["m_i_margin"] += margin

        self.balance = self.balance + margin
        print("Balance is now", self.balance)

    def _should_skip_runner(self, market_id: str, runner: RunnerBook) -> bool:
        """
        Determine whether a runner should be skipped based on its status and whether it
        has been previously processed or bet on.

        Args:
            market_id (str): The ID of the market the runner is in.
            runner (RunnerBook): The runner to check.

        Returns:
            bool: True if the runner should be skipped, False otherwise.
        """
        if (
            not runner.status == "ACTIVE"
            # or runner.selection_id in self.prediction_status_tracker[market_id]
            or not self.test_analysis_df["selection_ids"]
            .isin([runner.selection_id])
            .any()
            or not self.test_analysis_df["market_id"].isin([market_id]).any()
            # ONLY ONE BET/PREDICTION FOR EACH RUNNER IN EACH MARKET MAX
            or runner.selection_id in self.back_bet_tracker[market_id].keys()
            or runner.selection_id in self.lay_bet_tracker[market_id].keys()
        ):
            return True
        return False

    def _create_order(
        self,
        market_id: float,
        runner: RunnerBook,
        confidence_price,
        mean_120,
        price_adjusted,
        bsp_value: float,
        market: Market,
        side: str,
    ):
        """Creates and places an order on the specified market with the specified parameters.

        Args:
            market_id (float): The ID of the market to place the order on.
            runner (RunnerBook): The runner on which to place the order.
            price_adjusted (numpy.float64): The adjusted price for the order.
            bsp_value (numpy.float64): The Betfair Starting Price value for the runner.
            market (Market): The Betfair market on which to place the order.
            side (str): The side of the market to place the order on, either 'BACK' or 'LAY'.

        Returns:
            None.

        """
        tracker = self._get_tracker(side)
        matched_tracker = self._get_matched_tracker(side)
        tracker[market_id].setdefault(runner.selection_id, {})
        matched_tracker[market_id].setdefault(runner.selection_id, {})
        trade = Trade(
            market_id=str(market_id),
            selection_id=runner.selection_id,
            handicap=runner.handicap,
            strategy=self,
        )

        is_price_above_bsp = price_adjusted > bsp_value
        self.metrics["q_correct" if is_price_above_bsp else "q_incorrect"] += 1

        odds = calculate_odds(bsp_value, confidence_price, mean_120)
        # current_odds = (
        #     runner.ex.available_to_back[0].price
        #     if side == "BACK"
        #     else runner.ex.available_to_lay[0].price
        # )
        # print("myodds", odds, "betfair odds", current_odds)
        stake = calculate_kelly_stake(balance=self.balance, odds=odds)
        order = trade.create_order(
            side=side,
            order_type=LimitOrder(
                price=price_adjusted,
                size=stake,
                persistence_type="LAPSE",
            ),
        )

        print(
            f"{side} order created at {self.seconds_to_start}: \n\tmarket id {market_id} \n\tmarket {market} \n\trunner {runner} \n\tprice adjusted {price_adjusted} \n\tbsp_value {bsp_value} \n\tOrder size (kelly_stake): {stake} \n\tOdds: {odds}"
        )

        # Get trackers based on order side
        tracker = self._get_tracker(side)
        matched_tracker = self._get_matched_tracker(side)

        tracker[market_id][runner.selection_id] = [
            order,
            bsp_value,
            str(market_id),
            runner.selection_id,
            runner.handicap,
            runner.sp.actual_sp,
            price_adjusted,
        ]

        matched_tracker[market_id][runner.selection_id] = False
        market.place_order(order)
        margin = calculate_margin(side, stake, price_adjusted, bsp_value)
        self.metrics["q_margin"] += margin

    def _get_adjusted_prices2(
        self,
        mean_120: np.float64,
        runner: RunnerBook,
        market_id: float,
        side: str,
    ) -> Tuple[np.float64, np.float64, np.float64]:
        """
        Calculate adjusted prices for back and lay bets along with the BSP value.

        :param test_analysis_df_y: A DataFrame containing the test analysis data
        :param mean_120: The mean_120 value for the current runner
        :param runner: The current RunnerBook object
        :param market_id: The market ID for the current market
        :return: A tuple containing the adjusted price, confidence price, and BSP value
        """
        number = self.ticks_df.iloc[self.ticks_df["tick"].sub(mean_120).abs().idxmin()][
            "number"
        ]
        number_adjust = number
        confidence_number = number + 7 if side == "LAY" else number - 5
        confidence_price = self.ticks_df.iloc[
            self.ticks_df["number"].sub(confidence_number).abs().idxmin()
        ]["tick"]
        price_adjusted = self.ticks_df.iloc[
            self.ticks_df["number"].sub(number_adjust).abs().idxmin()
        ]["tick"]
        bsp_row = self.test_analysis_df_y.loc[
            (self.test_analysis_df_y["selection_ids"] == runner.selection_id)
            & (self.test_analysis_df_y["market_id"] == market_id)
        ]
        bsp_value = bsp_row["bsps"].values[0]

        return price_adjusted, confidence_price, bsp_value

    def _get_adjusted_prices(
        self,
        mean_120: np.float64,
        std_120: np.float64,
        vol_120: np.float64,
        runner: RunnerBook,
        market_id: float,
        side: str,
    ) -> Tuple[np.float64, np.float64, np.float64]:
        """
        Calculate adjusted prices for back and lay bets along with the BSP value.

        :param test_analysis_df_y: A DataFrame containing the test analysis data
        :param mean_120: The mean_120 value for the current runner
        :param runner: The current RunnerBook object
        :param market_id: The market ID for the current market
        :return: A tuple containing the adjusted price, confidence price, and BSP value
        """
        number = self.ticks_df.iloc[self.ticks_df["tick"].sub(mean_120).abs().idxmin()][
            "number"
        ]
        number_adjust = number
        confidence_number = (
            number + (std_120 / vol_120)
            if side == "LAY"
            else number - (std_120 / vol_120)
        )

        confidence_price = self.ticks_df.iloc[
            self.ticks_df["number"].sub(confidence_number).abs().idxmin()
        ]["tick"]
        price_adjusted = self.ticks_df.iloc[
            self.ticks_df["number"].sub(number_adjust).abs().idxmin()
        ]["tick"]
        bsp_row = self.test_analysis_df_y.loc[
            (self.test_analysis_df_y["selection_ids"] == runner.selection_id)
            & (self.test_analysis_df_y["market_id"] == market_id)
        ]
        bsp_value = bsp_row["bsps"].values[0]

        return price_adjusted, confidence_price, bsp_value

    def _get_tracker(self, side: str) -> Dict[str, Dict[str, list]]:
        """
        Get the back or lay bet tracker based on the side.

        Args:
            side (str): The side to retrieve the tracker for.

        Returns:
            dict: The corresponding back or lay bet tracker.
        """
        return self.back_bet_tracker if side == "BACK" else self.lay_bet_tracker

    def _get_matched_tracker(self, side: str) -> Dict[str, Dict[str, bool]]:
        """
        Get the matched back or lay bet tracker based on the side.

        Args:
            side (str): The side to retrieve the matched tracker for.

        Returns:
            dict: The corresponding matched back or lay bet tracker.
        """
        return (
            self.matched_back_bet_tracker
            if side == "BACK"
            else self.matched_lay_bet_tracker
        )

    def _get_model_prediction_and_features(
        self, runner: RunnerBook, market_id: float
    ) -> Tuple[np.float64, np.float64]:
        predict_row = self.test_analysis_df.loc[
            (self.test_analysis_df["selection_ids"] == runner.selection_id)
            & (self.test_analysis_df["market_id"] == market_id)
        ]
        mean_120 = predict_row["mean_120"].values[0]
        std_120 = predict_row["std_120"].values[0]
        vol_120 = predict_row["volume_120"].values[0]
        predict_row = normalized_transform(predict_row, self.ticks_df)
        predict_row = predict_row.drop(["bsps_temp", "bsps"], axis=1)
        predict_row = pd.DataFrame(self.scaler.transform(predict_row), columns=self.clm)
        runner_predicted_bsp = self.model.predict(predict_row)

        return runner_predicted_bsp, mean_120, std_120, vol_120
