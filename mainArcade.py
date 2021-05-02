"""
Move Sprite by Angle

Simple program to show basic sprite usage.

Artwork from http://kenney.nl

If Python and Arcade are installed, this example can be run from the command line with:
python -m arcade.examples.sprite_move_angle
"""
import arcade
import os
import math

SPRITE_SCALING = 0.5

SCREEN_WIDTH = 1400
SCREEN_HEIGHT = 1000
SCREEN_TITLE = "Move Sprite by Angle Example"

MOVEMENT_SPEED = 5
ANGLE_SPEED = 5
ACCELERATION_UNIT = 0.1
FRICTION = 0.995 # z intervala [0,1]
GRIP = 0.04 # z intervala [0,1]
BRAKING_POWER = 0.6

MAX_SPEED = 10
TOL = 0.01
TOL_ANGLE = 0.01

class Player(arcade.Sprite):
    """ Player class """

    def __init__(self, image, scale):
        """ Set up the player """

        # Call the parent init
        super().__init__(image, scale)

        # Create a variable to hold our speed. 'angle' is created by the parent
        #self.center_x;
        #self.center_y;
        #self.angle;

        self.speed_angle = 0
        self.speed = 0

        self.accelerating = False
        self.braking = False

    def update(self):
        if self.accelerating:
            speed_temp = (self.speed + ACCELERATION_UNIT)
            self.speed = min(speed_temp, MAX_SPEED)
        elif self.braking:
            speed_temp = (self.speed - ACCELERATION_UNIT * BRAKING_POWER) * FRICTION
            self.speed = max(speed_temp, 0)
        else:
            speed_temp = self.speed * FRICTION
            self.speed = speed_temp if speed_temp > TOL else 0

        #print('center_x: {}, center_y: {}, v: {}, a: {}, sa: {}'.format(*list(map(lambda x : round(x,2),[self.center_x, self.center_y, self.speed, self.angle, self.speed_angle]))))
        # Convert angle in degrees to radians.
        angle_rad = math.radians(self.angle)
        change_angle_rad = math.radians(self.change_angle)

        # Rotate the ship
        self.angle += self.change_angle        

        # Use math to find our change based on our speed and angle
        speed_angle_temp = (1 - (1-GRIP) * (self.speed / MAX_SPEED)**0.3) * (self.angle - self.speed_angle)
        if abs(speed_angle_temp) > TOL_ANGLE:
            self.speed_angle += speed_angle_temp    
        #elif abs(speed_angle_temp) > TOL_ANGLE and self.change_angle == 0:
            #print(0)
            #self.speed_angle = self.angle
        #else:

        #self.speed_angle = self.speed_angle + speed_angle_temp if abs(speed_angle_temp) > TOL_ANGLE and self.change_angle else self.angle
        speed_angle_rad = math.radians(self.speed_angle)

        #speed_temp = (self.speed[0] - norm(self.speed) * math.sin(angle_rad + 5*change_angle_rad))

        x_temp = self.center_x + self.speed * (-math.sin(speed_angle_rad))
        y_temp = self.center_y + self.speed * math.cos(speed_angle_rad)
        if x_temp < 0:
            self.center_x = 0
        elif x_temp > SCREEN_WIDTH:
            self.center_x = SCREEN_WIDTH
        else:
            self.center_x = x_temp
        if y_temp < 0:
            self.center_y = 0
        elif y_temp > SCREEN_HEIGHT:
            self.center_y = SCREEN_HEIGHT
        else:
            self.center_y = y_temp
        
        

        #self.center_x += -self.speed * math.sin(angle_rad - change_angle_rad)
        #self.center_y += self.speed * math.cos(angle_rad - change_angle_rad)



class MyGame(arcade.Window):
    """
    Main application class.
    """

    def __init__(self, width, height, title, update_rate = 1/60):
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
            self.player_sprite.accelerating = True
            #self.player_sprite.speed = (self.player_sprite.speed + ACCELERATION_UNIT)# * FRICTION
            print(round(self.player_sprite.speed,2))
        elif key == arcade.key.DOWN:
            self.player_sprite.braking = True
            #self.player_sprite.speed = (self.player_sprite.speed - ACCELERATION_UNIT)# * FRICTION

        # Rotate left/right
        elif key == arcade.key.LEFT:
            self.player_sprite.change_angle = ANGLE_SPEED
        elif key == arcade.key.RIGHT:
            self.player_sprite.change_angle = -ANGLE_SPEED

    def on_key_release(self, key, modifiers):
        """Called when the user releases a key. """

        if key == arcade.key.UP:
            self.player_sprite.accelerating = False
        if key == arcade.key.DOWN:
            self.player_sprite.braking = False
        if key == arcade.key.LEFT or key == arcade.key.RIGHT:
            self.player_sprite.change_angle = 0


def main():
    """ Main method """
    window = MyGame(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
    window.setup()
    arcade.run()


if __name__ == "__main__":
    main()