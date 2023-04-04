class MetricTracker:
    def __init__(self, balance=10000.00):
        self.profit = 0.0
        self.q_correct = 0
        self.q_incorrect = 0
        self.matched_correct = 0
        self.matched_incorrect = 0
        self.m_c_margin = 0.0
        self.m_i_margin = 0.0
        self.green_margin = 0.0
        self.amount_gambled = 0.0
        self.lay_matched_correct = 0
        self.lay_matched_incorrect = 0
        self.back_matched_correct = 0
        self.back_matched_incorrect = 0
        self.q_margin = 0.0
        self.balance = balance

    def update_metrics(
        self,
        side: str,
        price: float,
        bsp_value: float,
        margin: float,
        green_up: bool = False,
    ) -> None:
        """
        Updates the metrics based on the side, price, BSP value and margin.

        Args:
            side (str): The side of the bet ("BACK" or "LAY").
            price (float): The price of the bet.
            bsp_value (float): The BSP value.
            margin (float): The margin of the bet.
            green_up (bool): Whether or not the bet was greened up.

        Returns:
            None
        """
        if (price > bsp_value and side == "BACK") or (
            price < bsp_value and side == "LAY"
        ):
            self.matched_correct += 1
            if side == "BACK":
                self.back_matched_correct += 1
            else:
                self.lay_matched_correct += 1
            self.m_c_margin += margin
            if green_up:
                self.green_margin += margin
            self.profit += margin
        else:
            self.matched_incorrect += 1
            if side == "BACK":
                self.back_matched_incorrect += 1
            else:
                self.lay_matched_incorrect += 1
            self.m_i_margin += margin
            if green_up:
                self.green_margin -= margin
            self.profit -= margin

        self.amount_gambled += margin
        self.q_margin += margin

    def __repr__(self):
        return f"MetricTracker(balance={self.balance}, profit={self.profit}, matched_correct={self.matched_correct}, matched_incorrect={self.matched_incorrect}, amount_gambled={self.amount_gambled})"
