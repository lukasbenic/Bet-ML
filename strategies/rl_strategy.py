from typing import Tuple, Union
import gymnasium
import numpy as np
import pandas as pd

from flumine import BaseStrategy
from flumine.order.trade import Trade
from flumine.order.order import LimitOrder, MarketOnCloseOrder, OrderStatus
from flumine.markets.market import Market
from betfairlightweight.resources import MarketBook, RunnerBook
from sklearn.preprocessing import StandardScaler
from stable_baselines3 import PPO

from utils.constants import TIME_BEFORE_START

from toms_utils import normalized_transform
from utils.utils import preprocess_test_analysis


class RLStrategy(BaseStrategy):
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
        self.test_analysis_df = test_analysis_df

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

        super().__init__(
            *args,
            **{
                k: v
                for k, v in kwargs.items()
                if k
                not in ["scaler", "ticks_df", "test_analysis_df", "model", "env", "clm"]
            },
        )

    # back and lay in here

    def start(self) -> None:
        self.back_bet_tracker = {}
        self.matched_back_bet_tracker = {}
        self.lay_bet_tracker = {}
        self.matched_lay_bet_tracker = {}
        self.order_dict_back = {}
        self.order_dict_lay = {}
        self.LP_traded = {}
        self.seconds_to_start = None
        self.market_open = True
        self.stake = 50  # @WHAT IS THIS
        self.first_nonrunners = True
        self.runner_number = None

    def check_market_book(self, market: Market, market_book: MarketBook) -> bool:
        # process_market_book only executed if this returns True
        _ = self.process_fundamentals(market_book)
        if self.first_nonrunners:
            self.runner_number = market_book.number_of_active_runners
            self.first_nonrunners = False
        else:
            if market_book.number_of_active_runners != self.runner_number:
                return False  # this will stop any more action happening in this market.

        if (market_book.status == "CLOSED") or (
            self.seconds_to_start < TIME_BEFORE_START
        ):
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

    def __send_order(
        self,
        market_id: float,
        runner: RunnerBook,
        price_adjusted: np.float64,
        bsp_value: np.float64,
        market: Market,
        action: str,
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
            if action == "BACK"
            else round(self.stake / (price_adjusted - 1), 2)
        )
        order = trade.create_order(
            side=action,
            order_type=LimitOrder(
                price=price_adjusted,
                size=size,
                persistence_type="LAPSE",
            ),
        )

        print(
            f"{side} bet order created at {self.seconds_to_start} seconds to start: \n\t{side} price adjusted: {price_adjusted}\n\tOrder size: {size}"
        )
        tracker = self.back_bet_tracker if side == "BACK" else self.lay_bet_tracker
        tracker[market_id][runner.selection_id] = [
            order,
            bsp_value,
            str(market_id),
            runner.selection_id,
            runner.handicap,
            runner.sp.actual_sp,
            price_adjusted,
        ]

        matched_tracker = (
            self.matched_back_bet_tracker
            if side == "BACK"
            else self.matched_lay_bet_tracker
        )

        matched_tracker[market_id][runner.selection_id] = False

        market.place_order(order)

        self.metrics["q_margin"] += size * (price_adjusted - bsp_value) / price_adjusted

    def process_market_book(
        self, market: Market, market_book: MarketBook, action
    ) -> None:
        # process marketBook object
        # Take each incoming message and combine to a df
        cont = self.process_fundamentals(market_book)

        self.market_open = market_book.status
        # We want to limit our betting to before the start of the race.

        market_id = float(market_book.market_id)
        if (
            self.seconds_to_start < 120
            and self.seconds_to_start > 100
            and self.check_market_book(market, market_book)
        ):
            for runner in market_book.runners:
                if market_id not in self.back_bet_tracker.keys():
                    self.back_bet_tracker.setdefault(market_id, {})
                    self.matched_back_bet_tracker.setdefault(market_id, {})
                if market_id not in self.lay_bet_tracker.keys():
                    self.lay_bet_tracker.setdefault(market_id, {})
                    self.matched_lay_bet_tracker.setdefault(market_id, {})

                runner_in_back_tracker = (
                    runner.selection_id in self.back_bet_tracker[market_id].keys()
                )
                runner_in_lay_tracker = (
                    runner.selection_id in self.lay_bet_tracker[market_id].keys()
                )
                if not runner_in_back_tracker or not runner_in_lay_tracker:
                    # self.back_bet_tracker[market_id][runner.selection_id] = {}
                    # self.matched_back_bet_tracker[market_id][runner.selection_id] = {}

                    if not runner.status == "ACTIVE":
                        continue
                    # print("RUNNER is active")

                    try:
                        (
                            test_analysis_df,
                            test_analysis_df_y,
                        ) = preprocess_test_analysis(self.test_analysis_df)
                        # if bsps not available skip
                        (
                            runner_predicted_bsp,
                            mean_120,
                            predict_row,
                        ) = self.__get_model_prediction_and_mean_120(
                            test_analysis_df, runner, market_id
                        )

                        predict_row["runner_predicted_bsp"] = runner_predicted_bsp
                        self.observation = predict_row

                    except Exception as e:
                        if isinstance(e, IndexError):
                            runner_predicted_bsp = mean_120 = False
                        else:
                            error_message = f"An error occurred during preprocessing test_analysis_df and/or getting model prediction: {e.__class__.__name__}"
                            print(f"{error_message} - {e}")
                            runner_predicted_bsp = mean_120 = False

                    if not mean_120:
                        continue
                        # in the back_price / number put in where you want to base from: prevoisly runner.last_price_traded
                        # get_price(runner.ex.available_to_back, 1)
                    try:
                        (
                            price_adjusted,
                            confidence_price,
                            bsp_value,
                        ) = self.__get_back_lay(
                            test_analysis_df_y, mean_120, runner, market_id
                        )

                        # SEND BACK BET ORDER
                        if not runner_in_back_tracker and action == "BACK":
                            self.back_bet_tracker[market_id].setdefault(
                                runner.selection_id, {}
                            )
                            self.matched_back_bet_tracker[market_id].setdefault(
                                runner.selection_id, {}
                            )

                            self.__send_order(  # IN RL only send order if optimal action
                                market_id,
                                runner,
                                price_adjusted,
                                bsp_value,
                                market,
                                action=action,
                            )
                        # SEND LAY BET ORDER
                        if not runner_in_lay_tracker and action == "LAY":
                            self.lay_bet_tracker[market_id].setdefault(
                                runner.selection_id, {}
                            )
                            self.matched_lay_bet_tracker[market_id].setdefault(
                                runner.selection_id, {}
                            )
                            self.__send_order(
                                market_id,
                                runner,
                                price_adjusted,
                                bsp_value,
                                market,
                                side=action,
                            )

                    except Exception as e:
                        error_message = f"An error occurred during order process: {e.__class__.__name__} - {e}"
                        print(error_message)

    def process_orders(self, market: Market, orders: list) -> None:
        # this is where we get the reward
        sides = ["BACK", "LAY"]
        try:
            for side in sides:
                tracker = (
                    self.back_bet_tracker if side == "BACK" else self.lay_bet_tracker
                )
                matched_tracker = (
                    self.matched_back_bet_tracker
                    if side == "BACK"
                    else self.matched_lay_bet_tracker
                )
                if len(tracker.keys()) == 0:
                    return

                for market_id in tracker.keys():
                    for selection_id in tracker[market_id].keys():
                        if len(tracker[market_id][selection_id]) == 0:
                            continue

                        order = tracker[market_id][selection_id][0]

                        if not order.status == OrderStatus.EXECUTION_COMPLETE:
                            continue

                        if not matched_tracker[market_id][selection_id]:
                            matched_tracker[market_id][selection_id] = True

                            # lay price adjusted or back price
                            price = tracker[market_id][selection_id][-1]

                            # NOTE an order got matched at   -seconds_to_start - FIX?
                            print(
                                f"{side} matched at {self.seconds_to_start} seconds to start: \n\tOrder size: {order.size_matched}"
                            )
                            if side == "LAY":
                                print(
                                    f"\tSupposed layed size: {round(self.stake / (price - 1), 2)}"
                                )

                            if (
                                (order.size_matched >= 10.00 and side == "BACK")
                                or order.size_matched > 1
                                and side == "LAY"
                            ):  # because lowest amount
                                bsp_value = tracker[market_id][selection_id][1]
                                margin = (
                                    (order.size_matched * (price - bsp_value) / price)
                                    if side == "BACK"
                                    else (
                                        order.size_matched * (bsp_value - price) / price
                                    )
                                )
                                if (price > bsp_value and side == "BACK") or (
                                    price < bsp_value and side == "LAY"
                                ):
                                    side_matched_correct = (
                                        self.metrics["back_matched_correct"]
                                        if side == "BACK"
                                        else self.metrics["lay_matched_correct"]
                                    )
                                    self.metrics["matched_correct"] += 1
                                    side_matched_correct += 1
                                    self.metrics["m_c_margin"] += margin

                                else:
                                    side_matched_incorrect = (
                                        self.metrics["back_matched_incorrect"]
                                        if side == "BACK"
                                        else self.metrics["lay_matched_incorrect"]
                                    )
                                    self.metrics["matched_incorrect"] += 1
                                    side_matched_incorrect += 1
                                    self.metrics["m_i_margin"] += margin

                                self.metrics["green_margin"] += margin

                                market_id_ = tracker[market_id][selection_id][2]
                                selection_id_ = tracker[market_id][selection_id][3]
                                handicap_ = tracker[market_id][selection_id][4]

                                trade = Trade(
                                    market_id=market_id_,
                                    selection_id=selection_id_,
                                    handicap=handicap_,
                                    strategy=self,
                                )
                                order = trade.create_order(
                                    side=side,
                                    order_type=MarketOnCloseOrder(
                                        liability=order.size_matched
                                    ),
                                )
                                market.place_order(order)

                                # Frees this up to be done again once we have greened.
                                # !! unhash below and the horse del_list if you want to do more bets.
                                # del_list_back.append(selection_id)
                                # self.matched_back_bet_tracker[market_id][selection_id] = False

                            elif order.size_matched != 0:
                                bsp_value = tracker[market_id][selection_id][1]
                                backed_price = tracker[market_id][selection_id][-1]
                                self.metrics["amount_gambled"] += order.size_matched
                                self.matched_tracker[market_id][selection_id] = True

                                if (price > bsp_value and side == "BACK") or (
                                    price < bsp_value and side == "LAY"
                                ):
                                    self.metrics["matched_correct"] += 1
                                    self.metrics["back_matched_correct"] += 1
                                    self.metrics["m_c_margin"] += margin
                                else:
                                    self.metrics["matched_incorrect"] += 1
                                    self.metrics["back_matched_incorrect"] += 1
                                    self.metrics["m_i_margin"] += margin

                            elif (order.status == OrderStatus.EXECUTABLE) & (
                                order.size_matched != 0
                            ):
                                bsp_value = tracker[market_id][selection_id][1]
                                price = tracker[market_id][selection_id][-1]
                                if self.seconds_to_start < TIME_BEFORE_START:
                                    self.metrics["amount_gambled"] += (
                                        order.size_matched
                                        if side == "LAY"
                                        else order.size_matched * (price - 1)
                                    )
                                    if (price > bsp_value and side == "BACK") or (
                                        price < bsp_value and side == "LAY"
                                    ):
                                        self.metrics["matched_correct"] += 1
                                        self.metrics["back_matched_correct"] += 1
                                        self.metrics["m_c_margin"] += margin

                                    else:
                                        self.metrics["matched_incorrect"] += 1
                                        self.metrics["back_matched_incorrect"] += 1
                                        self.metrics["m_i_margin"] += margin

                                    market.cancel_order(order)
                                    matched_tracker[market_id][selection_id] = True
            reward = self.metrics
            return rewards
        except Exception as e:
            print(f"Exception during process orders function execution: {e}")
            # for horse in del_list_back:
            # del self.back_bet_tracker[market_id][horse]s
