import gymnasium as gym
import Environments.findTargetEnv
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.evaluation import evaluate_policy

# Press the green button in the gutter to run the script.
def demo_Lunar_Lander_random_action():
    env = gym.make("LunarLander-v2", render_mode = "human")
    observation, info = env.reset()

    for _ in range(1000):
        action = env.action_space.sample() # random action
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()


def learn_lunar_lander_PPO():
    env = gym.make("LunarLander-v2", render_mode = "rgb_array")
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=1e6, progress_bar=True)
    model.save("LunarLanderModel_1e6")
    #evaluate
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

def demo_trainaed_model(model_path:str):
    env = gym.make("LunarLander-v2", render_mode="rgb_array")
    model = PPO.load(model_path, env)
    vec_env = model.get_env()
    obs = vec_env.reset()

    for _ in range(10000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = vec_env.step(action)
        vec_env.render("human")


if __name__ == '__main__':
    # create env
    env = gym.make("FindTargetEnv-v0", render_mode="rgb_array", size=5)
    # create model
    ALGO = PPO
    model = ALGO("MlpPolicy", env=env, verbose=1)
    # train model
    model.learn(total_timesteps=5e5, progress_bar=True)
    # save model
    model_path = "FindTarget_5e4"
    model.save(model_path)
    # demo model

    env = gym.make("FindTargetEnv-v0", render_mode="human", size=5)
    model = ALGO.load(model_path, env)
    vec_env = model.get_env()
    obs = vec_env.reset()

    for _ in range(10000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = vec_env.step(action)

        vec_env.render("human")

        if done:
            obs = vec_env.reset()

    vec_env.close()
    #pass

