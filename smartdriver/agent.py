from collections import deque
from smartdriver.constants import *
import tensorflow as tf
import numpy as np
from tensorflow import keras
import random
import matplotlib.pyplot as plt

max_epsilon = 1 # You can't explore more than 100% of the time
min_epsilon = 0.01 # At a minimum, we'll always explore 1% of the time


class Agent():
    def __init__(self, player_sprite, state_shape, weights=None):
        self.actions = np.array(["L", "R"])#, ""])#, "U", "D"])
        #self.actions = np.array(["L","Ll", "R","Rl"])#, "U", "D"])
        self.predicted_actions = np.array([0,0,0,0,0,0])
        self.heuristic_actions = np.array([0,0,0,0,0,0])
        self.random_actions = np.array([0,0,0,0,0,0])
        self.player_sprite = player_sprite
        self.state_shape = state_shape

        self.min_replay_size = 400
        self.max_replay_size = 50*self.min_replay_size
        self.replay_memory = deque(maxlen=self.max_replay_size)
        self.finished = False
        self.total_training_rewards = 0
        
        self.train_iteration = 1
        self.target_update_period = self.min_replay_size // 2
        self.main_update_period = self.min_replay_size // 50

        self.epsilon = 1
        self.decay = 0.4
        self.episode = 1

        self.learning_rate = 0.05 # Learning rate
        self.discount_factor = 0.99


        self.max_reward = 0
        self.max_state = None

        self.batch_size = self.min_replay_size // 10


        #lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                            #initial_learning_rate=1e-2,
                            #decay_steps=10000,
                            #decay_rate=0.9)
        self.optimizer = keras.optimizers.Adam(learning_rate=0.0001)
        self.model = self.init_model()
        self.target_model = self.init_model()
        self.test()
        if weights:
            self.model.set_weights(weights)
        else:
            self.initial_train()
            self.test()
        self.update_target_model()

        

    def initial_train(self):
        train_states = []
        Qs = []
        n = 200
        max_r = 2
        Ds, As = [], []
        for angle in np.linspace(-179,179,n):
            for distance in np.linspace(TOL_CHECKPOINT+1, 500, n):
                state = [distance,angle]
                Ds.append(distance)
                As.append(angle)
                #r = self.get_reward(state,0)
                r = 5
                #if r > max_r:
                #    max_r = r
                if len(Qs) % 1000 == 0:
                    print(state, r)

                train_states.append(np.array(state))
                if angle < -ANGLE_SPEED/2:
                    #Qs.append(np.array([0,0,4,0]))
                    Qs.append(np.array([0,r]))
                elif angle > ANGLE_SPEED/2:
                    Qs.append(np.array([r,0]))
                else:
                    #Qs.append(np.array([0,2,0,2]))
                    Qs.append(np.array([r/2,r/2]))
                '''
                for angle_n in np.linspace(-179,179,n):
                    for distance_n in np.linspace(TOL_CHECKPOINT+1, 500, n):                            
                        for speed in np.linspace(0,MAX_SPEED,n):
                            for checkpoint_dif in [0,1]:
                                state = [distance,angle,distance_n,angle_n,speed]
                                state = state[:2]
                                r = self.get_reward(state,checkpoint_dif)
                                if len(Qs) % 1000 == 0:
                                    print(state, r)
                                    max_r = r

                                train_states.append(np.array(state))
                                if angle < -ANGLE_SPEED/2:
                                    #Qs.append(np.array([0,0,4,0]))
                                    Qs.append(np.array([0,r]))
                                elif angle > ANGLE_SPEED/2:
                                    Qs.append(np.array([r,0]))
                                else:
                                    #Qs.append(np.array([0,2,0,2]))
                                    Qs.append(np.array([r/2,r/2]))
                '''
        train_states = np.array(train_states)
        plt.plot(Ds,As,'.')
        plt.show()
        Qs = np.array(Qs)
        #print(train_states)
        #print(Qs)
        self.model.fit(train_states, Qs, epochs=8, verbose=1)
        self.update_target_model()

    def test(self):
        test_states = [[50,45], [50,20], [100,130], [100,90], [100,45], [500,90], [200,45], [500,10], [500,2], [100,-130], [50,-45], [50,-20], [100,-90], [100,-45], [500,-90], [200,-45], [500,-10], [500,-2]]
        #test_states = [self.player_sprite.round_state(state) for state in test_states]
        #test_states = [[x[0],x[1],MAX_DISTANCE, 180, MAX_SPEED] for x in test_states]
        test_predict = np.array([self.model.predict((state,)).flatten() for state in test_states])
        test_predict_target = np.array([self.target_model.predict(np.array((state,))).flatten() for state in test_states])
        test_actions = np.array([self.actions[np.argmax(predicted)] for predicted in test_predict])
        test_actions_target = np.array([self.actions[np.argmax(predicted)] for predicted in test_predict_target])
        #test_predict = self.model.predict(test_states)
        print("\nTEST")
        print(np.round(self.player_sprite.get_current_state(),decimals=2))
        print(round(self.get_reward(self.player_sprite.get_current_state(),0),2))
        print(np.round(test_predict,decimals=2))
        
        print("model:")
        print(len(test_states))
        print(test_actions[:len(test_states)//2])
        print(test_actions[len(test_states)//2:])

        print("\ntarget_model:")
        print(np.round(test_predict_target,decimals=2))
        print(test_actions_target[:len(test_states)//2])
        print(test_actions_target[len(test_states)//2:])
        print()
        print("P:",self.predicted_actions)
        print("R:",self.random_actions)
        print("H:",self.heuristic_actions)
        print()

    def do_predicted_move(self):
        encoded_reshaped = np.array((self.player_sprite.get_current_state(),))
        predicted = self.target_model.predict(encoded_reshaped).flatten()
        action = self.actions[np.argmax(predicted)]
        self.player_sprite.next_move_and_update(action)

    def do_training_step(self):

        #model = self.model
        #replay_memory = self.replay_memory
        state = self.player_sprite.get_current_state()
        checkpoint = self.player_sprite.next_checkpoint
        #total_training_rewards = self.total_training_rewards
        #target_model = self.target_model

        ####
        if self.train_iteration % 100 == 0:
            self.test()
        ####

        random_number = np.random.random()
        random_choice = False
        heuristic_choice = False
        predicted_choice = False
        # 2. Explore using the Epsilon Greedy Exploration Strategy
        if random_number <= self.epsilon or self.train_iteration < self.min_replay_size:
            # Explore
            #if (random_number <= ALPHA and self.epsilon > ALPHA/2):
            #    action = self.player_sprite.angle_heuristic()
            #    heuristic_choice = True if action else False
            #if not heuristic_choice:
            action = np.random.choice(self.actions)
            random_choice = True
            #print("RR:", round(self.player_sprite.get_angle_dif(),1), action)
        else:
            # Exploit best known action
            encoded_reshaped = np.array((self.player_sprite.get_current_state(),))
            predicted = self.model.predict(encoded_reshaped).flatten()

            # odločimo se, kaj bo naslednja poteza
            
            if abs(predicted[0] - predicted[1]) / np.max(predicted) > 0.02:
                action = self.actions[np.argmax(predicted)]
                if state[1] > 0 and action == "R":
                    print("predict:", predicted, state, action)
                elif state[1] < 0 and action == "L":
                    print("predict:", predicted, state, action)
                predicted_choice = True
            else:
                action = np.random.choice(self.actions)
                random_choice = True
            #print("--:", round(self.player_sprite.get_angle_dif(),1), action)
            #print("predicted:", list(map(lambda x : round(x,2), predicted)))
            #print("action:", action)
        
        

        if predicted_choice:
            self.predicted_actions[self.get_action_ind(action)] += 1
        if random_choice:
            self.random_actions[self.get_action_ind(action)] += 1
        if heuristic_choice:
            self.heuristic_actions[self.get_action_ind(action)] += 1
        
        # glede na izbrano potezo se premaknemo
        is_not_finished = self.player_sprite.next_move_and_update(action)
        self.finished = not is_not_finished

        # za novo stanje izračunamo nagrado
        if not self.finished:

            #if np.random.random() < 0.1:
            #    new_state = self.player_sprite.get_current_state()
            #    new_checkpoint = self.player_sprite.next_checkpoint
            #    print(state, self.get_reward(state, new_checkpoint-checkpoint))

            new_state = self.player_sprite.get_current_state()
            new_checkpoint = self.player_sprite.next_checkpoint
            reward = self.get_reward(new_state, new_checkpoint-checkpoint)

            #print(f"state: {state} action: {action} reward: {reward}, new_state: {new_state} is_prediction {is_prediction}")
            #print(f"action: {action}, reward: {round(reward,4)}, is_prediction: {is_prediction}")

            if reward > 0:
                #print(state, self.get_reward(state, new_checkpoint-checkpoint))
                # v spomin dodamo (stanje, premik, nagrada, novo_stanje, končal)
                self.replay_memory.append([state, action, reward, new_state, self.finished])

            # 3. Update the Main Network using the Bellman Equation
            if self.train_iteration % self.main_update_period == 0 or self.finished:
                #old_weights = np.array(self.model.get_weights())
                self.train_main_model()#replay_memory, self.finished)
                #new_weights = np.array(self.model.get_weights())
                #print(old_weights-new_weights)
            self.state = new_state
            self.total_training_rewards += reward

        else:
            # player je prišel do konca kroga
            print('Total training rewards: {} after n steps = {}.'.format(self.total_training_rewards, self.episode))
            self.episode += 1

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

        if self.train_iteration % self.target_update_period == 0:
            self.update_target_model()
            
        if len(self.replay_memory) > self.max_replay_size:
            self.replay_memory = random.sample(self.replay_memory, self.min_replay_size)        
        
        
        self.epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-self.decay * self.episode)
        if self.train_iteration % 100 == 0:
            print("epsilon:", round(self.epsilon,2))
        
         
        

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

        batch_size = min(self.batch_size, len(self.replay_memory)//3)
        mini_batch = random.sample(self.replay_memory, batch_size)

        current_states = np.zeros((batch_size, self.state_shape[0]))
        next_states = np.zeros((batch_size, self.state_shape[0]))
        action, reward, next_reward, done = [], [], [], []
        
        for i in range(batch_size):
            current_states[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            next_states[i] = mini_batch[i][3]
            next_reward.append(self.get_reward(next_states[i],0))
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
                #print(action[i], self.get_action_ind(action[i]))
                action_made_ind = self.get_action_ind(action[i])[0][0]
                #print("a:",i, action[i], action_ind, action_made_ind)

                # Q trenutnega stanja za akcijo, ki smo jo naredili (Q(s,a)), popravimo na 
                # (reward trenutnega stanja) + gamma * max_{a}(Q_target(s',a))
                current_Qs_main[i][action_made_ind] = reward[i] + self.discount_factor * (next_Qs_target[i][action_ind])
                #t = 1000*(next_reward[i] - reward[i])
                #current_Qs_main[i] = [-t for _ in range(len(self.actions))]
                #current_Qs_main[i][action_made_ind] = t
        self.model.fit(current_states, current_Qs_main, batch_size=self.batch_size, epochs=2, verbose=0)
        #self.do_epochs(current_states, current_Qs_main, epochs=3)

    def loss(self, y, y_predicted):
        
        print(y,y_predicted)
        return (y - y_predicted)**2

    def do_epochs(self, X, Y, epochs=1):
        for x, y in zip(X,Y):
            with tf.GradientTape() as tape:
                # Forward pass.
                logits = self.model(np.array(x,))#.predict(np.array((x,))).flatten()
                # Loss value for this batch.
                loss_value = self.loss(y, logits)

            # Get gradients of loss wrt the weights.
            gradients = tape.gradient(loss_value, self.model.trainable_weights)

            # Update the weights of the model.
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))



    def init_model(self):
        """ The agent maps X-states to Y-actions
        e.g. The neural network output is [.1, .7, .1, .3]
        The highest value 0.7 is the Q-Value.
        The index of the highest action (0.7) is action #1.
        """
        #learning_rate = 0.0005
        model = keras.Sequential()
        init = tf.keras.initializers.RandomNormal(mean=0, stddev=0.05, seed=RANDOM_SEED)
        #inputs = keras.layers.Input(shape=self.state_shape)

        # Convolutions on the frames on the screen
        #layer1 = keras.layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
        #layer2 = keras.layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
        #layer3 = keras.layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

        #layer4 = keras.layers.Flatten()(layer3)
        layer0 = keras.layers.Dense(self.state_shape[0], activation="linear", kernel_initializer=init)
        layer1 = keras.layers.Dense(self.state_shape[0]*8, activation="relu", kernel_initializer=init)
        layer2 = keras.layers.Dense(self.state_shape[0]*8, activation="relu", kernel_initializer=init)
        layer3 = keras.layers.Dense(self.state_shape[0]*8, activation="relu", kernel_initializer=init)
        action = keras.layers.Dense(len(self.actions), activation="linear")

        model.add(layer0)
        model.add(layer1)
        model.add(layer2)
        model.add(layer3)
        model.add(action)
        #layer1 = keras.layers.Dense(12, activation="relu", kernel_initializer=init)(inputs)
        #layer2 = keras.layers.Dense(8, activation="relu", kernel_initializer=init)(layer1)
        #layer3 = keras.layers.Dense(20, activation="linear", kernel_initializer=init)(layer2)
        #action = keras.layers.Dense(len(self.actions), activation="linear")(layer2)
        #model = keras.Model(inputs=inputs, outputs=action)
        model.compile(loss=keras.losses.Huber(), optimizer=self.optimizer, metrics=[keras.losses.Huber()])
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

    def get_reward_distance(self, distance):
        if distance == MAX_DISTANCE:
            return 0
        else:
            return np.exp(-(distance - TOL_CHECKPOINT)/100)

    def get_reward_angle(self, angle):
        if angle == 180:
            return 0
        else:
            angle = 0 if abs(angle) < ANGLE_SPEED else angle
            return np.exp(-5*(1 - (180 - abs(angle))/180))

    # 300, 400
    def get_reward(self, state, checkpoint_dif):
        #distance, angle, distance_n, angle_n, speed = state#, wrong_way = state
        distance, angle = state
        reward_distance = self.get_reward_distance(distance)
        reward_angle = self.get_reward_angle(angle)

        '''
        if distance_n < 5*TOL_CHECKPOINT:
            reward_distance_n = self.get_reward_distance(distance_n)
            reward_angle_n = self.get_reward_angle(angle_n)
        else:
            reward_distance_n, reward_angle_n = 0, 0
        '''

        #penalty_wrong_way = -2 if wrong_way else 0
        reward_checkpoint = TOL_CHECKPOINT*checkpoint_dif
        reward = reward_angle + reward_distance + reward_checkpoint# + reward_angle_n/2 + reward_distance_n/2 #+ penalty_wrong_way#+ reward_checkpoint
        #reward = reward_distance + (180 - abs(angle))/180
        #reward = 1
        #print("angle, reward_angle:", round(angle,2),",", round(reward_angle,2))
        #print("state:", list(map(lambda x : round(x,2), state)), ", ch_dif:",checkpoint_dif, end=", ")
        #print("stRew:", list(map(lambda x : round(x,2), [reward_distance, reward_angle, reward_angle_x_distance, reward_checkpoint])))
        #print("reward:",round(reward,4))
        if reward > self.max_reward:
            self.max_reward = reward
            self.max_state = state
            print(state,":",round(reward,2))
        reward_distance = 1 if distance < 3*TOL_CHECKPOINT else 0
        return reward_checkpoint + reward_distance