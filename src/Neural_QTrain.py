import gym
import tensorflow as tf
import numpy as np
import random

# General Parameters
# -- DO NOT MODIFY --
ENV_NAME = 'CartPole-v0'
EPISODE = 200000  # Episode limitation
STEP = 200  # Step limitation in an episode
TEST = 10  # The number of tests to run every TEST_FREQUENCY episodes
TEST_FREQUENCY = 100  # Num episodes to run before visualizing test accuracy

# TODO: HyperParameters
GAMMA = 0.99 # discount factor
INITIAL_EPSILON = 1 # starting value of epsilon
FINAL_EPSILON = 0.01 # final value of epsilon
EPSILON_DECAY_STEPS = 100 # decay period
BATCH_SIZE = 20


# Create environment
# -- DO NOT MODIFY --
env = gym.make(ENV_NAME)
epsilon = INITIAL_EPSILON
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.n

# Placeholders
# -- DO NOT MODIFY --
state_in = tf.placeholder("float", [None, STATE_DIM])
action_in = tf.placeholder("float", [None, ACTION_DIM])
target_in = tf.placeholder("float", [None])

# TODO: Define Network Graph
def n_learning(state_in, state_dim, action_dim):
        W1 = tf.get_variable("W1",shape=[state_dim,64])
        b1 = tf.get_variable("b1",shape=[1,64],initializer=tf.constant_initializer(0.0))
        W2 = tf.get_variable("W2", shape=[100, 100],)
        b2 = tf.get_variable("b2", shape=[1, 100], initializer = tf.constant_initializer(0.0))
        W3 = tf.get_variable("W3", shape=[64, action_dim])
        b3 = tf.get_variable("b3", shape=[1, action_dim], initializer = tf.constant_initializer(0.0))
        logits_layer1 = tf.matmul(state_in, W1) + b1
        output_layer1 = tf.tanh(logits_layer1)
        logits_layer3 = tf.matmul(output_layer1, W3) + b3
        output_layer3 = logits_layer3
        return output_layer3
'''
    # Layer2
	logits_layer2 = tf.matmul(output_layer1, W2) + b2
	output_layer2 = tf.tanh(logits_layer2)
    # Layer3
'''

# TODO: Network outputs
q_values =n_learning(state_in,STATE_DIM,ACTION_DIM)
q_action =tf.reduce_sum(tf.multiply(q_values,action_in),reduction_indices=1)

# TODO: Loss/Optimizer Definition
loss =tf.reduce_sum(tf.squared_difference(target_in, q_action))
optimizer =tf.train.AdamOptimizer().minimize(loss)

# Start session - Tensorflow housekeeping
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())


# -- DO NOT MODIFY ---
def explore(state, epsilon):
    """
    Exploration function: given a state and an epsilon value,
    and assuming the network has already been defined, decide which action to
    take using e-greedy exploration based on the current q-value estimates.
    """
    Q_estimates = q_values.eval(feed_dict={
        state_in: [state]
    })
    if random.random() <= epsilon:
        action = random.randint(0, ACTION_DIM - 1)
    else:
        action = np.argmax(Q_estimates)
    one_hot_action = np.zeros(ACTION_DIM)
    one_hot_action[action] = 1
    return one_hot_action

replay_buffer=[]
# Main learning loop
for episode in range(EPISODE):

    # initialize task
    state = env.reset()

    # Update epsilon once per episode
    epsilon -= epsilon / EPSILON_DECAY_STEPS

    # Move through env according to e-greedy policy
    for step in range(STEP):
        action = explore(state, epsilon)
        next_state, reward, done, _ = env.step(np.argmax(action))

        nextstate_q_values = q_values.eval(feed_dict={
            state_in: [next_state]
        })

        # TODO: Calculate the target q-value.
        # hint1: Bellman
        # hint2: consider if the episode has terminated
        replay_buffer.append((state, action, reward, next_state, done))
        if len(replay_buffer) > BATCH_SIZE:
        	batch = random.sample(replay_buffer, BATCH_SIZE)
        	state_batch = [data[0] for data in batch]
        	action_batch = [data[1] for data in batch]
        	reward_batch = [data[2] for data in batch]
        	next_state_batch = [data[3] for data in batch]
        	target_batch = []
        	q_value_batch = q_values.eval(feed_dict={state_in: next_state_batch})
        	for x in range(0,BATCH_SIZE):
        		is_done = batch[x][4]
        		if is_done:
        			target_batch.append(reward_batch[x])
        		else:
        			target = reward_batch[x] + GAMMA * np.max(q_value_batch[x])
        			target_batch.append(target)


	        # Do one training step
	        session.run([optimizer], feed_dict={
	            target_in: target_batch,
	            action_in: action_batch,
	            state_in: state_batch
	        })

        # Update
        state = next_state
        if done:
            break

    # Test and view sample runs - can disable render to save time
    # -- DO NOT MODIFY --
    if (episode % TEST_FREQUENCY == 0 and episode != 0):
        total_reward = 0
        for i in range(TEST):
            state = env.reset()
            for j in range(STEP):
                #env.render()
                action = np.argmax(q_values.eval(feed_dict={
                    state_in: [state]
                }))
                state, reward, done, _ = env.step(action)
                total_reward += reward
                if done:
                    break
        ave_reward = total_reward / TEST
        print('episode:', episode, 'epsilon:', epsilon, 'Evaluation '
                                                        'Average Reward:', ave_reward)

env.close()
