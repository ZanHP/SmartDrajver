import os
import time

import arcade
from pyglet.libs.x11.xlib import Screen

from smartdriver.constants import *
from smartdriver.player import Player
from smartdriver.track import Track

from shapely.geometry import LineString, Polygon # type: ignore

from arcade import Point
from arcade import check_for_collision_with_list
from arcade import Sprite
from arcade import SpriteList

from PIL import Image


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
        print(super().current_view)
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

        self.pause = False

        self.view_left = 0

        arcade.Window.set_update_rate(self, update_rate)

        # Set the background color
        arcade.set_background_color(arcade.color.BLACK)

    def setup(self):
        """ Set up the game and initialize the variables. """
        
        '''

        used for measuring distance to the wall
        # Sprite lists

        self.player_list = arcade.SpriteList()
        
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



        # Set up the track
        self.track = Track(TRACK1)
        

        # Set up the player
        self.player_sprite = Player(":resources:images/space_shooter/playerShip1_orange.png", SPRITE_SCALING, start_point, self.track, self.smart, self.show, self.verbose)
                
        self.player_list.append(self.player_sprite)
        #self.physics_engine = arcade.PhysicsEngineSimple(self.player_sprite,
        #                                                 self.wall_list)

        #arcade.has_line_of_sight()

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

        # This command has to happen before we start drawing
        arcade.start_render()

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
        

        self.track.draw_track()
        self.sprite.draw()

        arcade.draw_text("S:   toggle smart\nEsc: pause", SCREEN_WIDTH-200, 20, TRACK_COLOR_PASSED, font_size=14)

        if self.pause:
            arcade.draw_text("PAUSED", SCREEN_WIDTH/2, SCREEN_HEIGHT/2+50, TRACK_COLOR_PASSED, font_size=50, anchor_x="center")
        
        '''
            #track = ((100,100),(250,300),(1200,100),(500,500))

            #track_x, track_y = list(zip(*track))

            
        '''
       
        
        # Draw all the sprites.
        self.player_list.draw()

    def on_update(self, delta_time):
        """ Movement and game logic """

        # Call update on sprite
        #print(self.player_list[0])
        if not self.pause:
            if self.player_list[0].smart:
                self.player_list[0].next_move_and_update()
            else:
                self.player_list[0].update()

        
        '''
        def komentar():
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
        if key == arcade.key.ESCAPE:
            self.pause = False if self.pause else True

        if key == arcade.key.S:
            self.smart = False if self.smart else True
            self.player_list[0].smart = self.smart
            self.player_list[0].on_release_key_up()
            self.player_list[0].on_release_key_down()
            self.player_list[0].on_release_key_left()
            self.player_list[0].on_release_key_right()

        if not self.smart:
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
