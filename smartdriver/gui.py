import os
import arcade
import numpy as np
import pickle

from smartdriver.constants import *
from smartdriver.player import Player
from smartdriver.track import Track
from smartdriver.agent import Agent

state_shape = 2
action_shape = 3
train_episodes = 300
test_episodes = 100


class MyGame(arcade.Window):
    """
    Main application class.
    """

    def __init__(self, width, height, title, update_rate = UPDATE_RATE, smart=True, show=True, verbose=False, train=False):

        self.show = show
        self.num_steps_made = 0

        self.player_list = None

        self.smart = smart
        self.train = train
        
        self.verbose = verbose

        self.pause = True
        self.finished = False

        self.best_actions = []

        # ce zelis pogledati simulacijo
        #dbfile = open('smartdriver/best.pkl', 'rb')     
        #self.best_actions, self.state_actions_dict = pickle.load(dbfile)        

        if self.show:
            super().__init__(width, height, title, update_rate=update_rate)
            file_path = os.path.dirname(os.path.abspath(__file__))
            os.chdir(file_path)
            arcade.set_background_color(arcade.color.BLACK)
        else:
            super().__init__(width//4, height//4, title, update_rate=update_rate)
            arcade.set_background_color(arcade.color.BLACK)
            

    def setup(self):
        """ Set up the game and initialize the variables. """

        self.player_list = arcade.SpriteList()

        # Set up the track
        self.track = Track(TRACK1)

        # Set up the player
        player_sprite = Player(":resources:images/space_shooter/playerShip1_orange.png", SPRITE_SCALING, self.track.checkpoints[0], self.track, self.smart, best_run=len(self.best_actions) if self.best_actions else np.Inf, verbose=self.verbose)

        if self.smart:
            #weights = None
            try:
                weights_file = open('weights.pkl','rb')
                weights = pickle.load(weights_file)
            except:
                weights = None
            # naredimo agenta, ki drži playerja, model in target_model
            self.agent = Agent(player_sprite, (state_shape,), weights)
            self.train_iteration = 0

        else:
            self.player_sprite = player_sprite


    def on_draw(self):
        arcade.start_render()

        if self.show:
            arcade.draw_text("time: {}".format(self.num_steps_made), SCREEN_WIDTH-200, SCREEN_HEIGHT-30, TRACK_COLOR_PASSED, font_size=14)
            #self.sprite.draw()
            self.track.draw_track()
            if self.smart:
                self.agent.player_sprite.draw()
            else:
                self.player_sprite.draw()
            arcade.draw_text("S:   toggle smart\nEsc: pause", SCREEN_WIDTH-200, 20, TRACK_COLOR_PASSED, font_size=14)
        else:
            arcade.draw_text("Learning", SCREEN_WIDTH//8, SCREEN_HEIGHT//8+20, TRACK_COLOR_PASSED, font_size=20, anchor_x="center")
            arcade.draw_text("time: {}".format(self.num_steps_made), 100, 30, TRACK_COLOR_PASSED, font_size=14)

        if self.pause:
            arcade.draw_text("PAUSED", SCREEN_WIDTH/2, SCREEN_HEIGHT/2+50, TRACK_COLOR_PASSED, font_size=50, anchor_x="center")

    def on_update(self, delta_time=UPDATE_RATE):
        """ Movement and game logic """
        if not self.pause:
            self.num_steps_made += 1
            if not self.smart:
                if not self.player_sprite.update():
                    self.pause = True
            elif self.agent.player_sprite.smart:
                if self.train:
                    self.agent.do_training_step()
                else:
                    self.agent.do_predicted_move()
            else:
                if not self.agent.player_sprite.update():
                    self.pause = True
                    self.finished = True
            
    def on_key_press(self, key, modifiers):
        """Called whenever a key is pressed. """
        if key == arcade.key.ESCAPE:
            self.pause = False if self.pause else True

        if key == arcade.key.W:
            weights = self.agent.target_model.get_weights()
            filehandler = open('weights.pkl', 'wb') 
            pickle.dump(weights, filehandler)

        if key == arcade.key.R:
            self.setup()

            #self.player_sprite.recorded_actions = self.best_actions
            #self.player_sprite.action_index = 0

            self.num_steps_made = 0
            self.pause = True
            self.finished = False

        if key == arcade.key.S:
            self.smart = False if self.smart else True
            self.agent.player_sprite.smart = self.smart
            self.agent.player_sprite.on_release_key_up()
            self.agent.player_sprite.on_release_key_down()
            self.agent.player_sprite.on_release_key_left()
            self.agent.player_sprite.on_release_key_right()


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
        if not self.smart:
            if key == arcade.key.UP:
                self.player_sprite.on_release_key_up()
            if key == arcade.key.DOWN:
                self.player_sprite.on_release_key_down()
            if key == arcade.key.LEFT:
                self.player_sprite.on_release_key_left()
            if key == arcade.key.RIGHT:
                self.player_sprite.on_release_key_right()