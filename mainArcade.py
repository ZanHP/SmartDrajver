import arcade
from smartdriver.constants import *
from smartdriver.gui import MyGame

def main():
    """ Main method """
    window = MyGame(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE, smart=False, show=True, verbose=False)
    window.setup()
    arcade.run()

    # 20.63

if __name__ == "__main__":
    
    main()