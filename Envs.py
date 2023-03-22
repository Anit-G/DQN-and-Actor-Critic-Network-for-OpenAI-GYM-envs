import gym


def cartpole():
    env = gym.make('CartPole-v1')
    env.seed(0)
    state_shape = env.observation_space.shape[0]
    no_of_actions = env.action_space.n

    print(f"State Shape: {state_shape}")
    print(f"Action Shape: {no_of_actions}")
    print(f"Enviornment Action Sample: {env.action_shape.sample()}")

    state = env.reset()
    action = env.action_shape.sample()
    print(f"-----------\nInitial State Value:\n{state}\n-----------")
    print(f"-----------\nInitial Action Value:\n{action}\n-----------")

    return env, state_shape, no_of_actions,

