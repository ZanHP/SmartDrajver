from collections import deque
import os
import pickle
import time

import arcade
import numpy as np
from arcade import Point, Sprite, SpriteList, check_for_collision_with_list
from PIL import Image
from pyglet.libs.x11.xlib import Screen
from pyglet.window.key import F
from scipy import interpolate
from shapely.geometry import LineString, Polygon  # type: ignore

from smartdriver.constants import *
from smartdriver.player import Player
from smartdriver.track import Track
import random
import tensorflow as tf
import numpy as np
from tensorflow import keras


epsilon = 1 # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start
max_epsilon = 1 # You can't explore more than 100% of the time
min_epsilon = 0.01 # At a minimum, we'll always explore 1% of the time
decay = 0.01

observation_shape = 2
action_shape = 3

actions = ["D","A", ""]

train_episodes = 300
test_episodes = 100


def get_action_ind(action):
    if action == "D":
        return 0
    elif action == "A":
        return 1
    else:
        return 2

def encode_observation(observation, n_dims):
    return observation

def train_model( replay_memory, model, target_model, done):
    learning_rate = 0.7 # Learning rate
    discount_factor = 0.618

    MIN_REPLAY_SIZE = 1000
    #if len(replay_memory) < MIN_REPLAY_SIZE:
    #    return

    #print(replay_memory)
    batch_size = 4
    mini_batch = random.sample(replay_memory, batch_size)
    current_states = np.array([encode_observation(transition[0], 2) for transition in mini_batch])

    current_states /= 1000
    current_qs_list = model.predict(current_states)
    
    new_current_states = np.array([encode_observation(transition[3], 2) for transition in mini_batch])
    new_current_states /= 1000

    future_qs_list = target_model.predict(new_current_states)

    X = []
    Y = []
    for index, (observation, action, reward, new_observation, done) in enumerate(mini_batch):
        if not done:
            max_future_q = reward + discount_factor * np.max(future_qs_list[index])
        else:
            max_future_q = reward

        current_qs = current_qs_list[index]
        ind = get_action_ind(action)
        current_qs[ind] = (1 - learning_rate) * current_qs[ind] + learning_rate * max_future_q

        X.append(encode_observation(observation, 2))
        Y.append(current_qs)
    model.fit(np.array(X), np.array(Y), batch_size=batch_size, verbose=0, shuffle=True)



def agent(state_shape, action_shape):
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
def get_reward(observation):
    sum_score = observation[0] 
    return 1/sum_score

    



class MyGame(arcade.Window):
    """
    Main application class.
    """

    def __init__(self, width, height, title, update_rate = UPDATE_RATE, smart=False, show=True, verbose=False, train=False):
        """
        Initializer
        """
        self.show = show
        self.num_steps_made = 0

        # Variables that will hold sprite lists
        self.player_list = None

        # Set up the player info
        self.player_sprite = None
        self.smart = smart
        
        self.verbose = verbose

        self.pause = True
        self.finished = False

        self.best_actions = []


        # ce zelis pogledati simulacijo
        #dbfile = open('smartdriver/599.pkl', 'rb')     
        #self.best_actions = pickle.load(dbfile)


        self.train = train
        self.train_iteration = 0

        if self.show:
            # Call the parent class initializer
            super().__init__(width, height, title, update_rate=update_rate)
            # Set the working directory (where we expect to find files) to the same
            # directory this .py file is in. You can leave this out of your own
            # code, but it is needed to easily run the examples using "python -m"
            # as mentioned at the top of this program.
            file_path = os.path.dirname(os.path.abspath(__file__))
            os.chdir(file_path)

            

            # Set the background color
            arcade.set_background_color(arcade.color.BLACK)
        else:
            super().__init__(width//4, height//4, title)
            arcade.set_background_color(arcade.color.BLACK)
            arcade.draw_text("Learning", width//8, height//8+20, TRACK_COLOR_PASSED, font_size=20, anchor_x="center")


         
         # 1. Initialize the Target and Main models
        # Main Model (updated every 4 steps)
        model = agent((observation_shape,), action_shape)
        # Target Model (updated every 100 steps)
        target_model = agent((observation_shape,), action_shape)
        target_model.set_weights(model.get_weights())

        replay_memory = deque(maxlen=50_000)




        self.steps_to_update_target_model = 0


        self.model = model
        self.replay_memory = replay_memory
        self.total_training_rewards = 0
        self.target_model = target_model

        self.episode = 1 

        #self.epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * self.episode)

    def setup(self):
        """ Set up the game and initialize the variables. """
        
        '''

        used for measuring distance to the wall
        # Sprite lists

        
        
        TRACK2 = [(300,100),(300,600),(900,600),(900,100),(300,100)]

        img = Image.new("RGB", (200, 200), (255, 255, 255))
        img.save("/tmp/image.png", "PNG")
        self.wall_list = arcade.SpriteList(use_spatial_hash=True)

        self.sprite = arcade.Sprite("/tmp/image.png")

        self.sprite.center_x = 300
        self.sprite.center_y = 750

        self.wall_list.append(self.sprite)
        start_point = (300, 200)
        '''
        self.player_list = arcade.SpriteList()



        # Set up the track
        self.track = Track(TRACK1)
        

        # Set up the player
        self.player_sprite = Player(":resources:images/space_shooter/playerShip1_orange.png", SPRITE_SCALING, self.track.checkpoints[0], self.track, self.smart, self.verbose)
        self.player_list.append(self.player_sprite)

        self.observation = self.player_sprite.states()
        self.epsilon = 1
    

    @staticmethod
    def objects_in_line(point_1: Point,
                      point_2: Point,
                      walls: SpriteList,
                      max_distance: int = -1):

        line_of_sight = LineString([point_1, point_2])
        if 0 < max_distance < line_of_sight.length:
            return False
        if not walls:
            return True
        return ( o for o in walls if Polygon(o.get_adjusted_hit_box()).crosses(line_of_sight))

    def on_draw(self):
        """
        Render the screen.
        """

        if self.show:
            # This command has to happen before we start drawing
            arcade.start_render()
        

            #self.sprite.draw()
            self.track.draw_track()

            arcade.draw_text("S:   toggle smart\nEsc: pause", SCREEN_WIDTH-200, 20, TRACK_COLOR_PASSED, font_size=14)

        if self.pause:
            arcade.draw_text("PAUSED", SCREEN_WIDTH/2, SCREEN_HEIGHT/2+50, TRACK_COLOR_PASSED, font_size=50, anchor_x="center")
        
        arcade.draw_text("time: {}".format(self.num_steps_made), SCREEN_WIDTH-200, SCREEN_HEIGHT-30, TRACK_COLOR_PASSED, font_size=14)
        self.player_list.draw()

        
        '''
        
        #track = ((100,100),(250,300),(1200,100),(500,500))
        if self.pause:
            arcade.draw_text("PAUSED", SCREEN_WIDTH/2, SCREEN_HEIGHT/2+50, TRACK_COLOR_PASSED, font_size=50, anchor_x="center")
        
        
                
        
        def komentar():
            #track = ((100,100),(250,300),(1200,100),(500,500))

            #track_x, track_y = list(zip(*track))

            #tck = interpolate.splrep(track_x, track_y)
            #print(tck)
        
        
        # Draw all the sprites.
        '''

    def on_update(self, delta_time=UPDATE_RATE):
        """ Movement and game logic """
        # Call update on sprite
        #print(self.player_list[0])
        if not self.pause:
            self.num_steps_made += 1
            if self.player_list[0].smart:
                if self.train:
                    self.steps_to_update_target_model += 1

                    model = self.model
                    replay_memory = self.replay_memory
                    observation = self.observation
                    total_training_rewards = self.total_training_rewards
                    steps_to_update_target_model = self.steps_to_update_target_model
                    target_model = self.target_model

                    random_number = np.random.rand()
                    # 2. Explore using the Epsilon Greedy Exploration Strategy
                    if random_number <= self.epsilon:
                        # Explore
                        action = random.choice(actions)
                        is_prediction = False
                    else:
                        # Exploit best known action
                        # model dims are (batch, env.observation_space.n)
                        is_prediction = True

                        encoded_reshaped = np.array((self.player_sprite.states(),)) / 1000
                        predicted = model.predict(encoded_reshaped).flatten()
                        print(f"predicted {predicted}")

                        action = actions[np.argmax(predicted)]
                        
                    

                    #new_observation, reward, done, info = env.step(action)
                    is_not_finished = self.player_list[0].next_move_and_update(action)
                    self.finished = not is_not_finished
                    new_observation = self.player_sprite.states()
                    reward = get_reward(new_observation)

                    print(f"observation: {observation} action: {action} reward: {reward}, new_observation: {new_observation} is_prediction {is_prediction}")
                    replay_memory.append([observation, action, reward, new_observation, self.finished])

                    # 3. Update the Main Network using the Bellman Equation
                    if steps_to_update_target_model % 4 == 0 or self.finished:
                        print("jo")
                        train_model(replay_memory, model, target_model, self.finished)


                    observation = new_observation
                    total_training_rewards += reward

                    if self.finished:
                        print('Total training rewards: {} after n steps = {} with final reward = {}'.format(total_training_rewards, self.episode, reward))
                        total_training_rewards += 1

                        if steps_to_update_target_model >= 100:
                            print('Copying main network weights to the target network weights')
                            target_model.set_weights(model.get_weights())
                            steps_to_update_target_model = 0

                        self.pause = True

                        
                    self.model = model
                    self.replay_memory = replay_memory
                    self.observation = observation
                    self.total_training_rewards = total_training_rewards
                    self.steps_to_update_target_model = steps_to_update_target_model
                    self.target_model = target_model

                    self.epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * self.episode)
                    print(self.epsilon)
                    self.episode += 1 
                    

                    print("Train iteration: ", self.train_iteration)                    
                    self.train_iteration += 1
                    
                    
            else:
                if not self.player_list[0].update():
                    self.pause = True
                    self.finished = True
            
            




    def on_key_press(self, key, modifiers):
        """Called whenever a key is pressed. """
        if key == arcade.key.ESCAPE:
            self.pause = False if self.pause else True


        if key == arcade.key.R:
            self.setup()

            self.player_sprite.recorded_actions = self.best_actions
            self.player_sprite.action_index = 0

            self.num_steps_made = 0
            self.pause = True
            self.finished = False
            

        if key == arcade.key.S:
            self.smart = False if self.smart else True
            self.player_list[0].smart = self.smart
            self.player_list[0].on_release_key_up()
            self.player_list[0].on_release_key_down()
            self.player_list[0].on_release_key_left()
            self.player_list[0].on_release_key_right()


        if not self.smart:
            # Forward/back
            if not self.finished and key in [arcade.key.UP, arcade.key.DOWN, arcade.key.RIGHT, arcade.key.LEFT]:
                self.pause = False
            if key == arcade.key.UP:
                self.player_sprite.on_press_key_up()
            elif key == arcade.key.DOWN:
                self.player_sprite.on_press_key_down()

            # Rotate left/right
            elif key == arcade.key.LEFT:
                self.player_sprite.on_press_key_left()
            elif key == arcade.key.RIGHT:
                self.player_sprite.on_press_key_right()
    def on_key_release(self, key, modifiers):
        """Called when the user releases a key. """
        if not self.player_list[0].smart:
            if key == arcade.key.UP:
                self.player_sprite.on_release_key_up()
            if key == arcade.key.DOWN:
                self.player_sprite.on_release_key_down()
            if key == arcade.key.LEFT:
                self.player_sprite.on_release_key_left()
            if key == arcade.key.RIGHT:
                self.player_sprite.on_release_key_right()           



        '''
            used for measuring distance to the wall

            arcade.draw_point(250, 900, (255,0,0),6.0)
            has = arcade.has_line_of_sight((self.player_sprite.center_x, self.player_sprite.center_y),
            (250, 900),
            self.wall_list        
            )
            anyone = self.objects_in_line((self.player_sprite.center_x, self.player_sprite.center_y),
            (250, 900),
            self.wall_list)
            print(has,list(anyone))

            track_points = (
            ((200, 200),(250,200)),
            ((200,500),(250,500)),
            )
            
            outer_track_points = (
                (200,0),(200,700),(1000, 700),(1000,0),(200,0)
            )
            inner_track_points = (
                (400,200),(400,500),(800,500),(800,200),(400,200)
            )

            arcade.draw_line_strip(inner_track_points, COLOR_WHITE)
            arcade.draw_line_strip(outer_track_points, COLOR_WHITE)
            
        '''
