#  BETFAIR ML

## RUN Mean120Regression Strategy Instructions

1. In main.py
    1. Make sure --strategy_name is set to Mean120Regression
    2. Make sure to use one of the models found in the models folder (without .pkl extension) for —model_name
    3. For a quick test run change --races
2. Run main.py

NOTE that if you want to adjust confidence in predictions go to Mean120Regression.py and change the following line in _get_adjusted_prices(…),  confidence_number = number + 12 if side == "LAY" else number - 4

## RUN RLStrategy Strategy Instructions

1. In main.py
    1. Make sure --strategy_name is set to RLStrategy
    2. Set —model_name to PPO_BayesianRidge or RPPO_Bayesian Ridge depending on what you want to load.
    3. To set the RL model, go to the load_model func in utils/rl_model_utils and set the correct path.
        1. For PPO available models are found in RL/PPO
        2. For RPPO available models are found in RL/RPPO
    4. For a quick test run change --races
2. Run main.py

## Train RL agents

1. In plhr_env.py or plhr_env2.py (plhr_env2.py attemtps to simulate simplistic order matching)
    1. Use train_model2 and set the desired params.
    2. Note that saved_model_path uses the extension that identifies a model e.g. 2_-2_+2 and this is used to load and retrain a saved model.