import gym


def cartpole():

    env = gym.make('CartPole-v1')
    env.reset(seed=0)
    state_shape = env.observation_space.shape[0]
    no_of_actions = env.action_space.n
    print('--------- CartPole-v1 Env ----------')
    print(f"State Shape: {state_shape}")
    print(f"Action Shape: {no_of_actions}")
    print(f"Enviornment Action Sample: {env.action_space.sample()}")
    
    state = env.reset(seed=0)
    action = env.action_space.sample()
    print(f"-----------\nInitial State Value:\n{state}\n-----------")
    print(f"-----------\nInitial Action Value:\n{action}\n-----------")
    print(f"Enviornment step: ", env.step(action))

    return env, state_shape, no_of_actions

def acrobot():

    env = gym.make('Acrobot-v1')
    env.reset(seed=0)
    state_shape = env.observation_space.shape[0]
    no_of_actions = env.action_space.n
    print('--------- Acrobot-v1 Env ----------')
    print(f"State Shape: {state_shape}")
    print(f"Action Shape: {no_of_actions}")
    print(f"Enviornment Action Sample: {env.action_space.sample()}")
    
    state = env.reset(seed=0)
    action = env.action_space.sample()
    print(f"-----------\nInitial State Value:\n{state}\n-----------")
    print(f"-----------\nInitial Action Value:\n{action}\n-----------")
    print(f"Enviornment step: ", env.step(action))

    return env, state_shape, no_of_actions


def mountaincar():

    env = gym.make('MountainCar-v0')
    env.reset(seed=0)
    state_shape = env.observation_space.shape[0]
    no_of_actions = env.action_space.n
    print('--------- MountainCar-v0 Env ----------')
    print(f"State Shape: {state_shape}")
    print(f"Action Shape: {no_of_actions}")
    print(f"Enviornment Action Sample: {env.action_space.sample()}")
    
    state = env.reset(seed=0)
    action = env.action_space.sample()
    print(f"-----------\nInitial State Value:\n{state}\n-----------")
    print(f"-----------\nInitial Action Value:\n{action}\n-----------")
    print(f"Enviornment step: ", env.step(action))

    return env, state_shape, no_of_actions


