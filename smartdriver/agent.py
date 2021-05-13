from collections import deque
from smartdriver.constants import *
import tensorflow as tf
import numpy as np
from tensorflow import keras
import random

epsilon = 1 # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start
max_epsilon = 1 # You can't explore more than 100% of the time
min_epsilon = 0.01 # At a minimum, we'll always explore 1% of the time
decay = 0.01

class Agent():
    def __init__(self, player_sprite, state_shape, action_shape):
        self.player_sprite = player_sprite
        self.model = self.init_model(state_shape, action_shape)
        self.target_model = self.init_model(state_shape, action_shape)
        self.replay_memory = deque(maxlen=50_000)
        self.finished = False
        self.total_training_rewards = 0
        self.episode = 1
        self.train_iteration = 0
        self.steps_to_update_target_model = 0
        self.epsilon = 1
        self.actions = ["R","L", ""]

    def do_training_step(self):
        self.steps_to_update_target_model += 1

        model = self.model
        replay_memory = self.replay_memory
        state = self.player_sprite.get_current_state()
        total_training_rewards = self.total_training_rewards
        steps_to_update_target_model = self.steps_to_update_target_model
        target_model = self.target_model

        random_number = np.random.rand()
        # 2. Explore using the Epsilon Greedy Exploration Strategy
        if random_number <= self.epsilon:
            # Explore
            action = random.choice(self.actions)
            is_prediction = False
        else:
            # Exploit best known action
            # model dims are (batch, env.state_space.n)
            is_prediction = True

            encoded_reshaped = np.array((self.player_sprite.get_current_state(),))
            predicted = model.predict(encoded_reshaped).flatten()

            # odločimo se, kaj bo naslednja poteza
            action = self.actions[np.argmax(predicted)]
            #print("predicted:", list(map(lambda x : round(x,2), predicted)))
            #print("action:", action)
            
        # glede na izbrano potezo se premaknemo
        is_not_finished = self.player_sprite.next_move_and_update(action)
        self.finished = not is_not_finished

        # za novo stanje izračunamo nagrado
        if not self.finished:
            new_state = self.player_sprite.get_current_state()
            reward = self.get_reward(new_state)

            #print(f"state: {state} action: {action} reward: {reward}, new_state: {new_state} is_prediction {is_prediction}")
            #print(f"action: {action}, reward: {round(reward,4)}, is_prediction: {is_prediction}")

            # v spomin dodamo (stanje, premik, nagrada, novo_stanje, končal)
            replay_memory.append([state, action, reward, new_state, self.finished])

            # 3. Update the Main Network using the Bellman Equation
            if steps_to_update_target_model % 4 == 0 or self.finished:
                #print("jo")
                self.train_model(replay_memory, self.finished)

        else:
            # player je prišel do konca kroga
            print('Total training rewards: {} after n steps = {}.'.format(total_training_rewards, self.episode))

            if steps_to_update_target_model >= 100:
                print('Copying main network weights to the target network weights')
                target_model.set_weights(model.get_weights())
                steps_to_update_target_model = 0

            # naredimo reset 
            self.player_sprite.next_checkpoint = 1
            self.player_sprite.track.next_checkpoint = 1
            self.player_sprite.center_x = self.player_sprite.track.checkpoints[0][0]
            self.player_sprite.center_y = self.player_sprite.track.checkpoints[0][1]
            self.player_sprite.accelerating = False
            self.player_sprite.braking = False
            self.player_sprite.speed = 0
            self.player_sprite.angle = 0
            self.player_sprite.change_angle = 0

            
        self.model = model
        self.replay_memory = replay_memory
        self.state = new_state
        self.total_training_rewards += reward
        self.steps_to_update_target_model = steps_to_update_target_model
        self.target_model = target_model

        self.epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * self.episode)
        #print("epsilon:", self.epsilon)
        self.episode += 1 
        

        print("Train iteration: ", self.train_iteration)                    
        self.train_iteration += 1

    def get_action_ind(self, action):
        if action == "R":
            return 0
        elif action == "L":
            return 1
        else:
            return 2

    def encode_state(self, state, n_dims):
        return state

    def train_model(self, replay_memory, done):
        learning_rate = 0.3 # Learning rate
        discount_factor = 0.9

        MIN_REPLAY_SIZE = 100
        if len(replay_memory) < MIN_REPLAY_SIZE:
            return None

        #print(replay_memory)
        batch_size = 64
        mini_batch = random.sample(replay_memory, batch_size)

        # stanja iz trenutnega batcha
        current_states = np.array([self.encode_state(transition[0], 2) for transition in mini_batch])

        # napovedi za stanja i-te iteracije
        current_qs_list = self.model.predict(current_states)
        
        # stanja 1 korak naprej
        new_current_states = np.array([self.encode_state(transition[3], 2) for transition in mini_batch])

        # napovedi za stanja (i+1)-te iteracije
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        Y = []
        for index, (state, action, reward, new_state, done) in enumerate(mini_batch):
            if not done:
                max_future_q = reward + discount_factor * np.max(future_qs_list[index])
            else:
                max_future_q = reward

            current_qs = current_qs_list[index]
            ind = self.get_action_ind(action)
            current_qs[ind] = (1 - learning_rate) * current_qs[ind] + learning_rate * max_future_q

            X.append(self.encode_state(state, 2))
            Y.append(current_qs)
        self.model.fit(np.array(X), np.array(Y), batch_size=batch_size, verbose=0, shuffle=True)



    def init_model(self, state_shape, action_shape):
        """ The agent maps X-states to Y-actions
        e.g. The neural network output is [.1, .7, .1, .3]
        The highest value 0.7 is the Q-Value.
        The index of the highest action (0.7) is action #1.
        """
        learning_rate = 0.001
        init = tf.keras.initializers.HeUniform()
        model = keras.Sequential()
        model.add(keras.Input(shape=state_shape))
        model.add(keras.layers.Dense(24, input_dim=state_shape, activation='relu', kernel_initializer=init))
        model.add(keras.layers.Dense(12, activation='relu', kernel_initializer=init))
        model.add(keras.layers.Dense(action_shape, activation='linear', kernel_initializer=init))
        model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
        return model


    # 300, 400
    def get_reward(self, state):
        distance, angle = state
        reward_distance = np.exp(-(distance - TOL_CHECKPOINT))
        reward_angle = (180 - abs(angle))/180
        reward = reward_distance + reward_angle
        #print("angle, reward_angle:", round(angle,2),",", round(reward_angle,2))
        print("state:", list(map(lambda x : round(x,2), state)))
        print("stRew:", list(map(lambda x : round(x,2), [np.exp(-(distance - TOL_CHECKPOINT)), (180 - abs(angle))/180])))
        #print("reward:",round(reward,4))
        return reward