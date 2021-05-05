from constants import *
import arcade
import os
from player import Player

class MyGame(arcade.Window):
    """
    Main application class.
    """

    def __init__(self, width, height, title, update_rate = UPDATE_RATE):
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

        arcade.Window.set_update_rate(self, update_rate)

        # Set the background color
        arcade.set_background_color(arcade.color.BLACK)

    def setup(self):
        """ Set up the game and initialize the variables. """

        # Sprite lists
        self.player_list = arcade.SpriteList()

        # Set up the player
        self.player_sprite = Player(":resources:images/space_shooter/playerShip1_orange.png", SPRITE_SCALING)
        self.player_sprite.center_x = SCREEN_WIDTH / 2
        self.player_sprite.center_y = SCREEN_HEIGHT / 2
        self.player_list.append(self.player_sprite)

    def on_draw(self):
        """
        Render the screen.
        """

        # This command has to happen before we start drawing
        arcade.start_render()

        # Draw all the sprites.
        self.player_list.draw()

    def on_update(self, delta_time):
        """ Movement and game logic """

        # Call update on all sprites (The sprites don't do much in this
        # example though.)
        
        self.player_list.update()

    def on_key_press(self, key, modifiers):
        """Called whenever a key is pressed. """

        # Forward/back
        if key == arcade.key.UP:
            self.player_sprite.on_press_key_up()
            #self.player_sprite.speed = (self.player_sprite.speed + ACCELERATION_UNIT)# * FRICTION
            #print(round(self.player_sprite.speed,2))
        elif key == arcade.key.DOWN:
            self.player_sprite.on_press_key_down()
            #self.player_sprite.speed = (self.player_sprite.speed - ACCELERATION_UNIT)# * FRICTION

        # Rotate left/right
        elif key == arcade.key.LEFT:
            self.player_sprite.on_press_key_left()
        elif key == arcade.key.RIGHT:
            self.player_sprite.on_press_key_right()

    def on_key_release(self, key, modifiers):
        """Called when the user releases a key. """

        if key == arcade.key.UP:
            self.player_sprite.on_release_key_up()
        if key == arcade.key.DOWN:
            self.player_sprite.on_release_key_down()
        if key == arcade.key.LEFT:
            self.player_sprite.on_release_key_left()
        if key == arcade.key.RIGHT:
            self.player_sprite.on_release_key_right()           