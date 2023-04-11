from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import (
    BayesianRidge,
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

from ensemble_regressor import EnsembleRegressor
from xgboost import XGBRegressor


FIELD_NAMES = [
    "bet_id",
    "strategy_name",
    "market_id",
    "selection_id",
    "trade_id",
    "date_time_placed",
    "price",
    "price_matched",
    "size",
    "size_matched",
    "profit",
    "side",
    "elapsed_seconds_executable",
    "order_status",
    "market_note",
    "trade_notes",
    "order_notes",
]

# Hyperparam for sorting out the end
TIME_BEFORE_START = 5

KELLY_PERCENT = 0.05

ANALYSIS_FILES_TRAIN = [
    "jan20_analysis_direct_nr0_100_50_many_wom.csv",
    "feb20_analysis_direct_nr0_100_50_many_wom.csv",
    "mar20_analysis_direct_nr0_100_50_many_wom.csv",
    "may22_analysis_direct_nr0_100_50_many_wom.csv",
    "jun22_analysis_direct_nr0_100_50_many_wom.csv",
]

ANALYSIS_FILES_TEST = ["jul22_analysis_direct_nr0_100_50_many_wom.csv"]

STRATEGY_COL_NAMES = [
    "SecondsToStart",
    "MarketId",
    "SelectionId",
    "MarketTotalMatched",
    "SelectionTotalMatched",
    "LastPriceTraded",
    "volume_last_price",
    "available_to_back_1_price",
    "available_to_back_1_size",
    "volume_traded_at_Bprice1",
    "available_to_back_2_price",
    "available_to_back_2_size",
    "volume_traded_at_Bprice2",
    "available_to_back_3_price",
    "available_to_back_3_size",
    "volume_traded_at_Bprice3",
    "reasonable_back_WoM",
    "available_to_lay_1_price",
    "available_to_lay_1_size",
    "volume_traded_at_Lprice1",
    "available_to_lay_2_price",
    "available_to_lay_2_size",
    "volume_traded_at_Lprice2",
    "available_to_lay_3_price",
    "available_to_lay_3_size",
    "volume_traded_at_Lprice3",
    "reasonable_lay_WoM",
]

regression_models = {
    "LinearRegression": LinearRegression,
    "Ridge": Ridge,
    "Lasso": Lasso,
    "ElasticNet": ElasticNet,
    "DecisionTreeRegressor": DecisionTreeRegressor,
    "RandomForestRegressor": RandomForestRegressor,
    "GradientBoostingRegressor": GradientBoostingRegressor,
    "SVR": SVR,
    "KNeighborsRegressor": KNeighborsRegressor,
    "BayesianRidge": BayesianRidge,
    "Ensemble": EnsembleRegressor,
    "GaussianProcessRegressor": GaussianProcessRegressor,
    "XGBRegressor": XGBRegressor,
}
