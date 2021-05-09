import os
import time

import arcade

from smartdriver.constants import *
from smartdriver.player import Player
from smartdriver.track import Track

import numpy as np
from scipy import interpolate

class MyGame(arcade.Window):
    """
    Main application class.
    """

    def __init__(self, width, height, title, update_rate = UPDATE_RATE, smart=False, show=True, verbose=False):
        """
        Initializer
        """

        # Call the parent class initializer
        super().__init__(width, height, title)

        # Set the working directory (where we expect to find files) to the same
        # directory this .py file is in. You can leave this out of your own
        # code, but it is needed to easily run the examples using "python -m"
        # as mentioned at the top of this program.
        file_path = os.path.dirname(os.path.abspath(__file__))
        os.chdir(file_path)

        # Variables that will hold sprite lists
        self.player_list = None

        # Set up the player info
        self.player_sprite = None
        self.smart = smart
        self.show = show
        self.verbose = verbose

        self.view_left = 0

        arcade.Window.set_update_rate(self, update_rate)

        # Set the background color
        arcade.set_background_color(arcade.color.BLACK)

    def setup(self):
        """ Set up the game and initialize the variables. """

        # Sprite lists
        self.player_list = arcade.SpriteList()

        # Set up the track
        self.track = Track([[100, 100], [100, SCREEN_HEIGHT-500], [SCREEN_WIDTH-100, SCREEN_HEIGHT-100]])

        # Set up the player
        self.player_sprite = Player(":resources:images/space_shooter/playerShip1_orange.png", SPRITE_SCALING, self.track, self.smart, self.show, self.verbose)
                
        self.player_list.append(self.player_sprite)

    def on_draw(self):
        """
        Render the screen.
        """

        # This command has to happen before we start drawing
        arcade.start_render()

        self.track.draw_track()



        track = ((100,100),(250,300),(1200,100),(500,500))

        track_x, track_y = list(zip(*track))

        #tck = interpolate.splrep(track_x, track_y)
        #print(tck)
        



        # Draw all the sprites.
        self.player_list.draw()

    def on_update(self, delta_time):
        """ Movement and game logic """

        # Call update on sprite
        #print(self.player_list[0])
        if self.player_list[0].smart:
            self.player_list[0].next_move_and_update()
        else:
            self.player_list[0].update()

        
        '''
        TRACK_WIDTH = 15
        WHITE = (255,255,255)

        main_points = ((100, 100), (2000, 200))
        
        
        def track_points(point):
            
            return ((point[0], point[1] + TRACK_WIDTH), (point[0] , point[1] - TRACK_WIDTH))

        main_points = list(map(track_points, main_points))

        #self.view_left += 2
        
        #arcade.set_viewport(self.view_left, SCREEN_WIDTH + self.view_left, 0, SCREEN_HEIGHT)
        #for i, element in enumerate(main_points[:-1]):
        #    
        #    arcade.draw_line(element[0][0], element[0][1], main_points[i][0][0], main_points[i][0][1], WHITE)
        #    arcade.draw_line(element[0][0], element[0][1], main_points[i][0][0], main_points[i][0][1], WHITE)

        #arcade.draw_line(150, 100, 350, 100, WHITE, line_width=3)

        #points = list(range(100))
        #points = list(zip(points,points)) 
        #arcade.draw_points(points, color=(255,255,255))
        '''


    def on_key_press(self, key, modifiers):
        """Called whenever a key is pressed. """
        if not self.player_list[0].smart:
            # Forward/back
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