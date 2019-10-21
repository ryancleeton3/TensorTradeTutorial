from tensortrade.exchanges.simulated import FBMExchange
from tensortrade.features.scalers import MinMaxNormalizer
from tensortrade.features.stationarity import FractionalDifference
from tensortrade.features import FeaturePipeline
from tensortrade.rewards import SimpleProfitStrategy
from tensortrade.actions import DiscreteActionStrategy
from tensortrade.environments import TradingEnvironment

# creating an environment 

normalize_price = MinMaxNormalizer(['open', 'high', 'low', 'close'])
difference = FractionalDifference(difference_order=0.6)
feature_pipeline = FeaturePipeline(steps=[normalize_price,difference])

exchange = FBMExchange(timeframe='1h',
						base_instrument='BTC',
						feature_pipeline=feature_pipeline)

reward_strategy = SimpleProfitStrategy()

action_strategy = DiscreteActionStrategy(n_actions=20,
										instrument_symbol='ETH/BTC')

environment = TradingEnvironment(exchange=exchange,
								action_strategy=action_strategy,
								reward_strategy=reward_strategy,
								feature_pipeline=feature_pipeline)


# defining the agent

from stable_baselines.common.policies import MlpLnLstmPolicy
from stable_baselines import PPO2

model = PPO2
policy = MlpLnLstmPolicy
params = {'learning_rate':1e-5,'nminibatches':1}

# training a strategy

from tensortrade.strategies import StableBaselinesTradingStrategy

strategy = StableBaselinesTradingStrategy(environment=environment,
										model=model,
										policy=policy,
										model_kwargs=params)

performance = strategy.run(steps=10)