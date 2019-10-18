# Trading Environment
# 	InstrumentExchange - provides observations to the environment and executes the agent's trades
# 	FeaturePipeline - optionally transforms the exchange output into a more meaninful set of features before it is passed to the agent
# 	ActionStrategy - converts the agent's actions into executable trades
# 	RewardStrategy - calculates the reward for each time step based on the agent's performance




# Instrument Exchanges
from tensortrade.environments import TradingEvnironment

environment = TradingEvnironment(exchange=exchange,
								action_strategy=action_strategy,
								reward_strategy=reward_strategy,
								feature_pipeline=feature_pipeline)

import ccxt

from tensortrade.exchanges.live import CCXTExchange

coinbase = ccxt.coinbasepro()
exchange = CCXTExchange(exchange=coinbase, base_instrument='USD')

# RobinhoodExchange and InteractiveBrokersExchange are works in progress and allow you to trade stocks and ETF
# FBMExchnage is a simulated exchange which is simply an implementation of SimulatedExchange

from tensortrade.exchanges.simulated import FBMExchange

exchange = FBMExchange(base_instrument='BTC', timeframe='1h')

import pandas as pd

from tensortrade.exchanges.simulated import SimulatedExchange

df = pd.read_csv('./data/btc_ohclv_1h.csv')

exchange = SimulatedExchange(data_frame=df, base_instrument='USD')




# Feature Pipelines

from tensortrade.features import FeaturePipeline
from tensortrade.features.scalers import MinMaxNormalizer
from tensortrade.features.stationarity import FractionalDifference
from tensortrade.features.indicators import SimpleMovingAverage

price_columns = ['open','high','low','close']

normalize_price = MinMaxNormalizer(price_columns)
moving_averages = SimpleMovingAverage(price_columns)
difference_all = FractionalDifference(difference_order=0.6)

feature_pipeline = FeaturePipeline(steps=[normalize_price, moving_averages,difference_all])

exchange.feature_pipeline = feature_pipeline





# Action Strategies

from tensortrade.actions import DiscreteActionStrategy

action_strategy = DiscreteActionStrategy(n_actions=20,instrument_symbol='BTC')



# Reward Strategies

from tensortrade.rewards import SimpleProfitStrategy

reward_strategy = SimpleProfitStrategy()




# Stable Baselines - our learning agent but tensortrade can be used with Tensorforce, Rays' RLLib, OpenAI's Baselines, Intel's Coach or anything from TensorFlow

from stable_baselines.common.policies import MlpLnLstmPolicy
from stable_baselines import PPO2

model = PPO2
policy = MlpLnLstmPolicy
params = {'learning_rate': 1e-5}

agent = model(policy, environment, model_kwargs=params)





# Trading Strategy

from tensortrade.strategies import TensorforceTradingStrategy, StableBaselinesTradingStrategy

a_strategy = TensorforceTradingStrategy(environment=environment,
										agent_spec=agent_spec,
										network_spec=network_spec)

b_strategy = StableBaselinesTradingStrategy(environment=environment,
											model=PPO2,
											policy=MlpLnLstmPolicy)
































