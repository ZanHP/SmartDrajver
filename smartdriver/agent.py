from collections import deque
from smartdriver.constants import *
import tensorflow as tf
import numpy as np
from tensorflow import keras
import random

epsilon = 1 # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start
max_epsilon = 1 # You can't explore more than 100% of the time
min_epsilon = 0.01 # At a minimum, we'll always explore 1% of the time


class Agent():
    def __init__(self, player_sprite, state_shape, weights=None):
        self.actions = np.array(["L", "R", ""])#, "U", "D"])
        self.actions_count = np.array([0,0,0,0])
        self.player_sprite = player_sprite
        self.state_shape = state_shape

        self.model = self.init_model(self.state_shape, len(self.actions))
        
        self.target_model = self.init_model(state_shape, len(self.actions))
        if weights:
            self.model.set_weights(weights)
        self.update_target_model()

        self.min_replay_size = 1000
        self.max_replay_size = 10*self.min_replay_size
        self.replay_memory = deque(maxlen=self.max_replay_size)
        self.finished = False
        self.total_training_rewards = 0
        self.episode = 1
        self.train_iteration = 1
        self.steps_to_update_target_model = 0
        self.epsilon = 1
        self.decay = 0.0004

        self.learning_rate = 0.3 # Learning rate
        self.discount_factor = 0.9

        self.batch_size = 256

        self.test()

    def test(self):
        test_states = [[50,45], [50,20], [100,130], [100,90], [100,45], [500,90], [200,45], [100,-130], [50,-45], [50,-20], [100,-90], [100,-45], [500,-90], [200,-45]]
        test_predict = np.array([self.model.predict(np.array((state,))).flatten() for state in test_states])
        test_actions = np.array([self.actions[np.argmax(predicted)] for predicted in test_predict])
        #test_predict = self.model.predict(test_states)
        print("\nTEST")
        print(np.round(self.player_sprite.get_current_state(),decimals=2))
        print(round(self.get_reward(self.player_sprite.get_current_state(),0),2))
        #print(np.round(test_predict,decimals=5))
        print(test_predict)
        print(test_actions)
        print()
        print(self.actions_count)
        print()

    def do_predicted_move(self):
        encoded_reshaped = np.array((self.player_sprite.get_current_state(),))
        predicted = self.target_model.predict(encoded_reshaped).flatten()
        action = self.actions[np.argmax(predicted)]
        self.player_sprite.next_move_and_update(action)

    def do_training_step(self):
        self.steps_to_update_target_model += 1

        #model = self.model
        #replay_memory = self.replay_memory
        state = self.player_sprite.get_current_state()
        checkpoint = self.player_sprite.next_checkpoint
        #total_training_rewards = self.total_training_rewards
        #steps_to_update_target_model = self.steps_to_update_target_model
        #target_model = self.target_model

        ####
        if self.train_iteration % 100 == 0:
            self.test()
        ####

        random_number = np.random.random()
        # 2. Explore using the Epsilon Greedy Exploration Strategy
        if random_number <= self.epsilon:
            # Explore
            if random_number <= ALPHA and self.epsilon > ALPHA/2:
                action = self.player_sprite.angle_heuristic()
            else:
                action = np.random.choice(self.actions)
            #action = "U"
        else:
            # Exploit best known action
            encoded_reshaped = np.array((self.player_sprite.get_current_state(),))
            predicted = self.model.predict(encoded_reshaped).flatten()

            # odločimo se, kaj bo naslednja poteza
            action = self.actions[np.argmax(predicted)]
            #print("predicted:", list(map(lambda x : round(x,2), predicted)))
            #print("action:", action)

        self.actions_count[self.get_action_ind(action)] += 1
        
        # glede na izbrano potezo se premaknemo
        is_not_finished = self.player_sprite.next_move_and_update(action)
        self.finished = not is_not_finished

        # za novo stanje izračunamo nagrado
        if not self.finished:
            new_state = self.player_sprite.get_current_state()
            new_checkpoint = self.player_sprite.next_checkpoint
            reward = self.get_reward(new_state, new_checkpoint-checkpoint)

            #print(f"state: {state} action: {action} reward: {reward}, new_state: {new_state} is_prediction {is_prediction}")
            #print(f"action: {action}, reward: {round(reward,4)}, is_prediction: {is_prediction}")

            # v spomin dodamo (stanje, premik, nagrada, novo_stanje, končal)
            self.replay_memory.append([state, action, reward, new_state, self.finished])

            # 3. Update the Main Network using the Bellman Equation
            if self.steps_to_update_target_model % 4 == 0 or self.finished:
                #old_weights = np.array(self.model.get_weights())
                self.train_main_model()#replay_memory, self.finished)
                #new_weights = np.array(self.model.get_weights())
                #print(old_weights-new_weights)
            self.state = new_state
            self.total_training_rewards += reward

        else:
            # player je prišel do konca kroga
            print('Total training rewards: {} after n steps = {}.'.format(self.total_training_rewards, self.episode))

            if self.steps_to_update_target_model >= 100:
                self.update_target_model()
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

            
        if len(self.replay_memory) > self.max_replay_size:
            self.replay_memory = random.sample(self.replay_memory, self.min_replay_size)        
        
        
        self.epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-self.decay * self.episode)
        if self.train_iteration % 50 == 0:
            print("epsilon:", round(self.epsilon,2))
        self.episode += 1 
        

        #print("Train iteration: ", self.train_iteration)                    
        self.train_iteration += 1
    
    def update_target_model(self):
        print('Copying main network weights to the target network weights')
        self.target_model.set_weights(self.model.get_weights())

    def get_action_ind(self, action):
        #print("get_action:", action,np.where(self.actions == action))
        return np.where(self.actions == action)

    def encode_state(self, state, n_dims):
        return state

    def train_model2(self, replay_memory, done):
        #learning_rate = 0.3 # Learning rate
        #discount_factor = 0.9

        #MIN_REPLAY_SIZE = 100
        if len(replay_memory) < self.min_replay_size:
            return None

        #print(replay_memory)
        batch_size = 64
        #mini_batch_inds = np.random.randint(len(replay_memory), size=batch_size)
        #mini_batch = np.array(replay_memory)[mini_batch_inds]
        mini_batch = random.sample(replay_memory, batch_size)

        # stanja iz trenutnega batcha
        current_states = np.array([self.encode_state(transition[0], 2) for transition in mini_batch])

        # napovedi Q za stanja i-te iteracije (z glavnim modelom)
        current_qs_list = self.model.predict(current_states)
        
        # stanja 1 korak naprej
        new_current_states = np.array([self.encode_state(transition[3], 2) for transition in mini_batch])

        # napovedi Q za stanja (i+1)-te iteracije (s target modelom)
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        Y = []
        for index, (state, action, reward, new_state, done) in enumerate(mini_batch):
            # max_future_q predstavlja target
            if not done:
                max_future_q = reward + self.discount_factor * np.amax(future_qs_list[index])
            else:
                max_future_q = reward

            current_qs = current_qs_list[index]
            action_ind = self.get_action_ind(action)
            current_qs[action_ind] = (1 - self.learning_rate) * current_qs[action_ind] + self.learning_rate * max_future_q

            X.append(self.encode_state(state, 2))
            Y.append(current_qs)
        self.model.fit(np.array(X), np.array(Y), batch_size=batch_size, verbose=0, shuffle=True)


    def train_main_model(self):
        # vir: https://theaisummer.com/Taking_Deep_Q_Networks_a_step_further/
        if len(self.replay_memory) < self.min_replay_size:
            return

        batch_size = min(self.batch_size, len(self.replay_memory))
        mini_batch = random.sample(self.replay_memory, batch_size)

        current_states = np.zeros((batch_size, self.state_shape[0]))
        next_states = np.zeros((batch_size, self.state_shape[0]))
        action, reward, done = [], [], []
        
        for i in range(batch_size):
            current_states[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            next_states[i] = mini_batch[i][3]
            done.append(mini_batch[i][4])

        current_Qs_main = self.model.predict(current_states)
        next_Qs_main = self.model.predict(next_states) #DQN
        next_Qs_target = self.target_model.predict(next_states) #Target model

        for i in range(self.batch_size):
            if done[i]:
                current_Qs_main[i][action[i]] = reward[i]
            else:
                # selection of action is from model
                # update is from target model
                action_ind = np.argmax(next_Qs_main[i])
                #print("action[i]",action[i])
                #print("a:",i, action[i], action_ind, self.get_action_ind(action[i]))
                current_Qs_main[i][self.get_action_ind(action[i])] = reward[i] + self.discount_factor * (next_Qs_target[i][action_ind])
        self.model.fit(current_states, current_Qs_main, batch_size=self.batch_size, epochs=30, verbose=0)


    def init_model(self, state_shape, action_shape):
        """ The agent maps X-states to Y-actions
        e.g. The neural network output is [.1, .7, .1, .3]
        The highest value 0.7 is the Q-Value.
        The index of the highest action (0.7) is action #1.
        """
        learning_rate = 0.001
        init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=RANDOM_SEED)
        inputs = keras.layers.Input(shape=self.state_shape)

        # Convolutions on the frames on the screen
        #layer1 = keras.layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
        #layer2 = keras.layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
        #layer3 = keras.layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

        #layer4 = keras.layers.Flatten()(layer3)

        layer1 = keras.layers.Dense(12, activation="sigmoid", kernel_initializer=init)(inputs)
        layer2 = keras.layers.Dense(6, activation="sigmoid", kernel_initializer=init)(layer1)
        action = keras.layers.Dense(len(self.actions), activation="softmax")(layer2)
        model = keras.Model(inputs=inputs, outputs=action)
        model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
        return model
        '''
        learning_rate = 0.001
        init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=RANDOM_SEED)
        model = keras.Sequential()
        model.add(keras.Input(shape=state_shape))
        model.add(keras.layers.Dense(8, input_dim=state_shape, activation='relu', kernel_initializer=init))
        #model.add(keras.layers.Dense(24, input_dim=state_shape, activation='relu', kernel_initializer=init))
        #model.add(keras.layers.Dense(12, activation='relu', kernel_initializer=init))
        model.add(keras.layers.Dense(action_shape, activation='softmax', kernel_initializer=init))
        model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
        #model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate))
        return model
        '''


    # 300, 400
    def get_reward(self, state, checkpoint_dif):
        distance, angle = state
        reward_distance = np.exp(-(distance - TOL_CHECKPOINT)/100)
        angle = 0 if abs(angle) < ANGLE_SPEED/2 else angle
        reward_angle = 10*np.exp(-5*(1 - (180 - abs(angle))/180))
        reward_angle_x_distance = 2*((reward_distance + reward_angle)/2)**4
        reward_checkpoint = TOL_CHECKPOINT*checkpoint_dif
        reward = reward_distance + reward_angle + reward_angle_x_distance + reward_checkpoint
        #reward = reward_distance + (180 - abs(angle))/180
        #reward = 1
        #print("angle, reward_angle:", round(angle,2),",", round(reward_angle,2))
        #print("state:", list(map(lambda x : round(x,2), state)), ", ch_dif:",checkpoint_dif, end=", ")
        #print("stRew:", list(map(lambda x : round(x,2), [reward_distance, reward_angle, reward_angle_x_distance, reward_checkpoint])))
        #print("reward:",round(reward,4))
        return reward_angle