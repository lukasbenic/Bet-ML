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

from utils.constants import TIME_BEFORE_START

from toms_utils import normalized_transform
from utils.utils import preprocess_test_analysis


class Mean120RegressionStrategy(BaseStrategy):
    def __init__(
        self,
        scaler: StandardScaler,
        ticks_df: pd.DataFrame,
        test_analysis_df: pd.DataFrame,
        model,
        clm,
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
        self.regression = True
        self.back_bet_tracker = {}
        self.matched_back_bet_tracker = {}
        self.lay_bet_tracker = {}
        self.matched_lay_bet_tracker = {}
        self.seconds_to_start = None
        self.market_open = True
        self.stake = 50
        self.first_nonrunners = True
        self.runner_number = None

        super_kwargs = kwargs.copy()
        super_kwargs.pop("scaler", None)
        super_kwargs.pop("ticks_df", None)
        super_kwargs.pop("test_analysis_df", None)
        super_kwargs.pop("model", None)
        super_kwargs.pop("clm", None)
        super().__init__(*args, **super_kwargs)

    def start(self) -> None:
        pass

    def _set_and_preprocess_test_analysis(self, test_analysis_df):
        test_analysis_df, test_analysis_df_y = preprocess_test_analysis(
            test_analysis_df
        )
        self.test_analysis_df = test_analysis_df
        self.test_analysis_df_y = test_analysis_df_y

    def reset(self) -> None:
        self.metrics = dict.fromkeys(self.metrics, 0)
        self.regression = True
        self.back_bet_tracker = {}
        self.matched_back_bet_tracker = {}
        self.lay_bet_tracker = {}
        self.matched_lay_bet_tracker = {}
        self.seconds_to_start = None
        self.market_open = True
        self.stake = 50
        self.first_nonrunners = True
        self.runner_number = None
        print("strategy params reset")

    def set_market_filter(self, market_filter: Union[dict, list]) -> None:
        self.market_filter = market_filter
        print("market filter set")

    def check_market_book(self, market: Market, market_book: MarketBook) -> bool:
        # process_market_book only executed if this returns True
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

    def process_fundamentals(self, market_book: MarketBook):
        # We already have everything we need in an excel file.
        # runner_count = 0
        seconds_to_start = (
            market_book.market_definition.market_time - market_book.publish_time
        ).total_seconds()
        self.seconds_to_start = seconds_to_start

        return True

    def process_runners(self):
        # This doesn't need to do anything.
        pass

    def _set_market_id_bet_trackers(self, market_id):
        if market_id not in self.back_bet_tracker.keys():
            self.back_bet_tracker[market_id] = {}
            self.matched_back_bet_tracker[market_id] = {}
        if market_id not in self.lay_bet_tracker.keys():
            self.lay_bet_tracker[market_id] = {}
            self.matched_lay_bet_tracker[market_id] = {}

    def process_market_book(self, market: Market, market_book: MarketBook) -> None:
        _ = self.process_fundamentals(market_book)
        self.market_open = market_book.status
        market_id = float(market_book.market_id)
        self._set_market_id_bet_trackers(market_id)

        if not self._should_process_market_book(market_id, market, market_book):
            # print(f"\Skipping processing market ({market_id})... ")
            return

        # print(f"\rNumber of runners: {len(market_book.runners)}")

        for runner in market_book.runners:
            if market_id not in self.back_bet_tracker.keys():
                self.back_bet_tracker[market_id] = {}
                self.matched_back_bet_tracker[market_id] = {}

            if self._should_skip_runner(market_id, runner):
                continue
            (
                runner_predicted_bsp,
                mean_120,
            ) = self._get_model_prediction_and_mean_120(runner, market_id)

            if not mean_120 or not runner_predicted_bsp:
                continue

            (
                back_price_adjusted,
                back_confidence_price,
                back_bsp_value,
            ) = self._get_adjusted_prices(
                market_id=market_id, runner=runner, mean_120=mean_120, side="BACK"
            )

            (
                lay_price_adjusted,
                lay_confidence_price,
                lay_bsp_value,
            ) = self._get_adjusted_prices(
                market_id=market_id, runner=runner, mean_120=mean_120, side="LAY"
            )

            # here is action back
            if (
                runner_predicted_bsp < back_confidence_price
                and mean_120 <= 50
                and mean_120 > 1.1
                and runner.selection_id not in self.back_bet_tracker[market_id].keys()
            ):
                print(
                    "predicted",
                    runner_predicted_bsp,
                    "confidence",
                    back_confidence_price,
                )
                self._send_bet(
                    market_id,
                    runner,
                    back_price_adjusted,
                    back_bsp_value,
                    market,
                    side="BACK",
                )

            # here is action lay
            if (
                runner_predicted_bsp > lay_confidence_price
                and lay_price_adjusted <= self.stake
                and lay_price_adjusted > 1.1
                and runner.selection_id not in self.lay_bet_tracker[market_id].keys()
            ):
                print(
                    "predicted",
                    runner_predicted_bsp,
                    "confidence",
                    back_confidence_price,
                )

                self._send_bet(
                    market_id,
                    runner,
                    lay_price_adjusted,
                    lay_bsp_value,
                    market,
                    side="LAY",
                )

    def process_orders(self, market: Market, orders: list) -> None:
        sides = ["BACK", "LAY"]

        for side in sides:
            tracker = self._get_tracker(side)
            matched_tracker = self._get_matched_tracker(side)

            if len(tracker.keys()) <= 0:
                continue

            for market_id in tracker.keys():
                for selection_id in tracker[market_id].keys():
                    self._process_order_for_selection(
                        market,
                        side,
                        tracker,
                        matched_tracker,
                        market_id,
                        selection_id,
                    )

    def _get_tracker(self, side: str) -> dict:
        return self.back_bet_tracker if side == "BACK" else self.lay_bet_tracker

    def _get_matched_tracker(self, side: str) -> dict:
        return (
            self.matched_back_bet_tracker
            if side == "BACK"
            else self.matched_lay_bet_tracker
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
        if not tracker[market_id][selection_id]:
            return  # Continue ??

        order = tracker[market_id][selection_id][0]
        price = tracker[market_id][selection_id][-1]
        bsp_value = tracker[market_id][selection_id][1]

        side_mc_key = f"{side.lower()}_matched_correct"
        side_mi_key = f"{side.lower()}_matched_incorrect"
        margin = self._calculate_margin(side, order.size_matched, price, bsp_value)
        gambled = self._calculate_gambled(side, order.size_matched, price)

        if (
            order.status == OrderStatus.EXECUTION_COMPLETE
            and not matched_tracker[market_id][selection_id]
        ):
            matched_tracker[market_id][selection_id] = True
            self._process_matched_order(
                market,
                side,
                tracker,
                market_id,
                selection_id,
                order,
                price,
                side_mc_key,
                side_mi_key,
                margin,
                bsp_value,
                gambled,
            )

        elif order.status == OrderStatus.EXECUTABLE and order.size_matched != 0:
            if self.seconds_to_start < TIME_BEFORE_START:
                self.metrics["amount_gambled"] += gambled
                self._update_metrics(
                    side, price, bsp_value, margin, side_mc_key, side_mi_key
                )
                market.cancel_order(order)
                tracker[market_id][selection_id] = True
                print("order canceled")

    def _update_matched_tracker(
        self, matched_tracker: dict, market_id: str, selection_id: str
    ) -> None:
        if not matched_tracker[market_id][selection_id]:
            matched_tracker[market_id][selection_id] = True

    def _should_process_market_book(self, market_id, market, market_book):
        # if market_id not in self.prediction_status_tracker:
        #     self.prediction_status_tracker[market_id] = {}

        if (
            self.seconds_to_start > 100
            and self.seconds_to_start < 120
            and self.check_market_book(market, market_book)
        ):
            return True
        return False

    def _process_matched_order(
        self,
        market: Market,
        side: str,
        tracker: dict,
        market_id: str,
        selection_id: str,
        order: BaseOrder,
        price: float,
        side_mc_key: str,
        side_mi_key: str,
        margin,
        bsp_value,
        gambled,
    ) -> None:
        print(
            f"Processing matched order for side: {side}, market_id: {market_id}, selection_id: {selection_id}, order: {order}, price: {price}, margin: {margin}, bsp_value: {bsp_value}, gambled: {gambled}"
        )
        selection_id_ = tracker[market_id][selection_id][3]
        handicap_ = tracker[market_id][selection_id][4]
        market_id_ = tracker[market_id][selection_id][2]

        if (order.size_matched >= 10.00 and side == "BACK") or (
            order.size_matched > 1 and side == "LAY"
        ):
            self._green_up(market, side, order, market_id_, selection_id_, handicap_)
            self._update_metrics(
                side, price, bsp_value, margin, side_mc_key, side_mi_key
            )
            self.metrics["green_margin"] += margin

        elif order.size_matched != 0:
            tracker[market_id][selection_id] = True
            self.metrics["amount_gambled"] += gambled
            self._update_metrics(
                side, price, bsp_value, margin, side_mc_key, side_mi_key
            )

    def _calculate_gambled(self, side, size_matched, price):
        if side == "BACK":
            return size_matched
        else:
            return size_matched * (price - 1)

    def _calculate_margin(
        self, side: str, size: float, price: float, bsp_value: float
    ) -> float:
        if side == "BACK":
            return size * (price - bsp_value) / price
        else:
            return size * (bsp_value - price) / price

    def _update_metrics(
        self,
        side: str,
        price: float,
        bsp_value: float,
        margin: float,
        matched_correct_key,
        matched_incorrect_key,
    ) -> None:
        if (price > bsp_value and side == "BACK") or (
            price < bsp_value and side == "LAY"
        ):
            self.metrics["matched_correct"] += 1
            self.metrics[matched_correct_key] += 1
            self.metrics["m_c_margin"] += margin
        else:
            self.metrics["matched_incorrect"] += 1
            self.metrics[matched_incorrect_key] += 1
            self.metrics["m_i_margin"] += margin

    def _green_up(
        self,
        market: Market,
        side: str,
        order: BaseOrder,
        market_id: str,
        selection_id: str,
        handicap,
    ) -> None:
        trade = Trade(
            market_id=market_id,
            selection_id=selection_id,
            handicap=handicap,
            strategy=self,
        )
        opposite_side = "LAY" if side == "BACK" else "BACK"
        order = trade.create_order(
            side=opposite_side,
            order_type=MarketOnCloseOrder(liability=order.size_matched),
        )
        market.place_order(order)
        print("greened", order)

    def _should_skip_runner(self, market_id, runner):
        if (
            not runner.status == "ACTIVE"
            # or runner.selection_id in self.prediction_status_tracker[market_id]
            or not self.test_analysis_df["selection_ids"]
            .isin([runner.selection_id])
            .any()
            or not self.test_analysis_df["market_id"].isin([market_id]).any()
            # ONLY ONE BET/PREDICTION FOR EACH RUNNENR IN EACH MARKET MAX
            or runner.selection_id in self.back_bet_tracker[market_id].keys()
            or runner.selection_id in self.lay_bet_tracker[market_id].keys()
        ):
            return True
        return False

    def _send_bet(self, market_id, runner, price_adjusted, bsp_value, market, side):
        tracker = self._get_tracker(side)
        matched_tracker = self._get_matched_tracker(side)
        tracker[market_id].setdefault(runner.selection_id, {})
        matched_tracker[market_id].setdefault(runner.selection_id, {})
        self._send_order(
            market_id,
            runner,
            price_adjusted,
            bsp_value,
            market,
            side=side,
        )
        print(f"{side} order sent.")

    def _send_order(
        self,
        market_id: float,
        runner: RunnerBook,
        price_adjusted: np.float64,
        bsp_value: np.float64,
        market: Market,
        side: str,
    ):
        trade = Trade(
            market_id=str(market_id),
            selection_id=runner.selection_id,
            handicap=runner.handicap,
            strategy=self,
        )

        if price_adjusted > bsp_value:
            self.metrics["q_correct"] += 1
        else:
            self.metrics["q_incorrect"] += 1

        size = (
            self.stake
            if side == "BACK"
            else round(self.stake / (price_adjusted - 1), 2)
        )
        order = trade.create_order(
            side=side,
            order_type=LimitOrder(
                price=price_adjusted,
                size=size,
                persistence_type="LAPSE",
            ),
        )

        print(
            f"{side} order created at {self.seconds_to_start}: \n\tmarket id {market_id} \n\tmarket {market} \n\trunner {runner} \n\tprice adjusted {price_adjusted} \n\tbsp_value {bsp_value} \n\tOrder size: {size}"
        )
        tracker = self._get_tracker(side)
        tracker[market_id][runner.selection_id] = [
            order,
            bsp_value,
            str(market_id),
            runner.selection_id,
            runner.handicap,
            runner.sp.actual_sp,
            price_adjusted,
        ]

        matched_tracker = self._get_matched_tracker(side)
        matched_tracker[market_id][runner.selection_id] = False
        market.place_order(order)
        margin = self._calculate_margin(side, size, price_adjusted, bsp_value)
        self.metrics["q_margin"] += margin

    def _get_model_prediction_and_mean_120(
        self, runner: RunnerBook, market_id: float
    ) -> Tuple[np.float64, np.float64]:
        predict_row = self.test_analysis_df.loc[
            (self.test_analysis_df["selection_ids"] == runner.selection_id)
            & (self.test_analysis_df["market_id"] == market_id)
        ]
        mean_120 = predict_row["mean_120"].values[0]
        predict_row = normalized_transform(predict_row, self.ticks_df)
        predict_row = predict_row.drop(["bsps_temp", "bsps"], axis=1)
        predict_row = pd.DataFrame(self.scaler.transform(predict_row), columns=self.clm)
        runner_predicted_bsp = self.model.predict(predict_row)

        return runner_predicted_bsp, mean_120

    def _get_adjusted_prices2(
        self,
        mean_120,
        predicted_price: np.float64,
        runner: RunnerBook,
        market_id: float,
    ) -> Tuple[np.float64, np.float64, np.float64]:
        """
        Calculate adjusted prices for back and lay bets along with the BSP value.

        :param test_analysis_df_y: A DataFrame containing the test analysis data
        :param predicted_price: The predicted price for the current runner
        :param runner: The current RunnerBook object
        :param market_id: The market ID for the current market
        :return: A tuple containing the adjusted price, confidence price, and BSP value
        """

        price_ratio = predicted_price / mean_120

        adjusted_ticks = self.ticks_df.copy()
        adjusted_ticks["adjusted_tick"] = adjusted_ticks["tick"] * price_ratio

        price_adjusted = adjusted_ticks.iloc[
            adjusted_ticks["adjusted_tick"].sub(predicted_price).abs().idxmin()
        ]["tick"]

        confidence_adjusted_tick = price_adjusted * (1 - self.confidence_factor)
        confidence_price = adjusted_ticks.iloc[
            adjusted_ticks["tick"].sub(confidence_adjusted_tick).abs().idxmin()
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
        confidence_number = number + 4 if side == "LAY" else number - 4
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
