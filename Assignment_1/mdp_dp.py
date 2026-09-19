### MDP Value Iteration and Policy Iteration
### Reference: https://web.stanford.edu/class/cs234/assignment1/index.html 
import numpy as np
import gymnasium as gym

np.set_printoptions(precision=3)

"""
For policy_evaluation, policy_improvement, policy_iteration and value_iteration,
the parameters P, nS, nA, gamma are defined as follows:

	P: nested dictionary
		From gym.core.Environment
		For each pair of states in [1, nS] and actions in [1, nA], P[state][action] is a
		tuple of the form (probability, nextstate, reward, terminal) where
			- probability: float
				the probability of transitioning from "state" to "nextstate" with "action"
			- nextstate: int
				denotes the state we transition to (in range [0, nS - 1])
			- reward: int
				either 0 or 1, the reward for transitioning from "state" to
				"nextstate" with "action"
			- terminal: bool
			  True when "nextstate" is a terminal state (hole or goal), False otherwise
	nS: int
		number of states in the environment
	nA: int
		number of actions in the environment
	gamma: float
		Discount factor. Number in range [0, 1)
"""

def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-8):
    """Evaluate the value function from a given policy.

    Parameters:
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    policy: np.array[nS,nA]
        The policy to evaluate. Maps states to actions.
    tol: float
        Terminate policy evaluation when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns:
    -------
    value_function: np.ndarray[nS]
        The value function of the given policy, where value_function[s] is
        the value of state s
    """
    
    value_function = np.zeros(nS)
    while True:
        delta = 0
        for s in range(nS): # looping through states
            v = value_function[s]
            sum1 = 0
            for a in range(nA): # looping through the actions for each state
                action_probability = policy[s, a] # value action
                sum2 = 0
                for state_action in P[s][a]: # represents the outer summation that mulitples inner summation by the probability of taking action given state
                    prob = state_action[0]
                    next_state = state_action[1]
                    reward = state_action[2]
                    terminal = state_action[3]
                    if not terminal:
                        sum2 += prob * (reward + gamma * value_function[next_state])
                    else:
                        sum2 += prob * reward
                sum1 += action_probability * sum2
            value_function[s] = sum1
            dif = abs(v - value_function[s])
            delta = max(delta, dif)
        if delta < tol:
            break
    return value_function 


def policy_improvement(P, nS, nA, value_from_policy, gamma=0.9):
    """Given the value function from policy improve the policy.

    Parameters:
    -----------
    P, nS, nA, gamma:
        defined at beginning of file
    value_from_policy: np.ndarray
        The value calculated from the policy
    Returns:
    --------
    new_policy: np.ndarray[nS,nA]
        A 2D array of floats. Each float is the probability of the action
        to take in that state according to the environment dynamics and the 
        given value function.
    """

    new_policy = np.ones([nS, nA]) / nA # policy as a uniform distribution

    for s in range(nS):
        state_action_values = [] # represents a list of values for the current state that holds said states' state action values
        for a in range(nA):
            sum2 = 0
            for state_action in P[s][a]:
                prob = state_action[0]
                next_state = state_action[1]
                reward = state_action[2]
                terminal = state_action[3]
                sum2 += prob * (reward + gamma * value_from_policy[next_state])
            state_action_values.append(sum2)

        max_action = state_action_values.index(max(state_action_values))
        new_action = np.zeros(nA)
        new_action[max_action] = 1
        new_policy[s] = new_action

    return new_policy


def policy_iteration(P, nS, nA, policy, gamma=0.9, tol=1e-8):
    """Runs policy iteration.

    You should call the policy_evaluation() and policy_improvement() methods to
    implement this method.

    Parameters
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    policy: policy to be updated
    tol: float
        tol parameter used in policy_evaluation()
    Returns:
    ----------
    new_policy: np.ndarray[nS,nA]
    V: np.ndarray[nS]
    """
    policy_stable = False

    old_policy = policy.copy()
    new_policy = None

    value_function = policy_evaluation(P, nS, nA, old_policy, gamma, tol = tol) # initialize value function to from current policy

    while not policy_stable:

        new_policy = policy_improvement(P, nS, nA, value_function, gamma)
        if np.array_equal(old_policy, new_policy):
            policy_stable = True
        else:
            value_function = policy_evaluation(P, nS, nA, new_policy,gamma, tol = tol)
        old_policy = new_policy.copy()


    return new_policy, value_function


def value_iteration(P, nS, nA, V, gamma=0.9, tol=1e-8):
    """
    Learn value function and policy by using value iteration method for a given
    gamma and environment.

    Parameters:
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    V: value to be updated
    tol: float
        Terminate value iteration when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns:
    ----------
    policy_new: np.ndarray[nS,nA]
    V_new: np.ndarray[nS]
    """
    V_new = V.copy()
    policy_new = np.zeros([nS, nA])

    while True:
        delta = 0
        for s in range(nS):
            v = V_new[s]
            action_sums = []
            for a in range(nA):
                sum2 = 0
                for state_action in P[s][a]:
                    prob = state_action[0]
                    next_state = state_action[1]
                    reward = state_action[2]
                    terminal = state_action[3]
                    if not terminal:
                        sum2 += prob * (reward + gamma * V_new[next_state])
                    else:
                        sum2 += prob * reward
                action_sums.append(sum2)
            max_value = max(action_sums)
            V_new[s] = max_value
            delta = max(delta, abs(v - V_new[s]))
        if delta < tol:
            break

    policy_new = policy_improvement(P,nS,nA,V_new,gamma)

    return policy_new, V_new

def render_single(env, policy, render = False, n_episodes=100):
    """
    Given a game envrionemnt of gym package, play multiple episodes of the game.
    An episode is over when the returned value for "done" = True.
    At each step, pick an action and collect the reward and new state from the game.

    Parameters:
    ----------
    env: gym.core.Environment
      Environment to play on. Must have nS, nA, and P as attributes.
    policy: np.array of shape [env.nS, env.nA]
      The action to take at a given state
    render: whether or not to render the game(it's slower to render the game)
    n_episodes: the number of episodes to play in the game. 
    Returns:
    ------
    total_rewards: the total number of rewards achieved in the game.
    """
    nA = env.action_space.n


    total_rewards = 0

    for episode_num in range(n_episodes):
        obs, info = env.reset() # initialize the episode

        current_states_prob = policy[obs]

        done = False
        episode_reward = 0
        step_count = 0

        while not done:
            if render:
                env.render() # render the game
            action = np.random.choice(nA, p=current_states_prob)
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            step_count += 1
            current_states_prob = policy[obs]
            done = terminated or truncated


        print(f"Episode {episode_num+ 1}: {step_count} steps, reward = {episode_reward}")

        total_rewards += episode_reward

    env.close()
    return total_rewards



