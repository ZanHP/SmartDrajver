import arcade
from constants import *
from gui import MyGame

def main():
    """ Main method """
    window = MyGame(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE, smart=True, show=True)
    window.setup()
    arcade.run()

if __name__ == "__main__":
    
    main()