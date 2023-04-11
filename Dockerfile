FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

RUN pip install pandas flumine numpy scikit-learn seaborn gymnasium joblib betfairlightweight Office365-REST-Python-Client matplotlib pyro-ppl optuna stable_baselines3 tensorboard xgboost
 
# Set working directory and default command
COPY . /app
WORKDIR /app
CMD ["python main.py", "python pre_live_horse_race_env.py"]
