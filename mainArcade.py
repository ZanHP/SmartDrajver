import arcade
from smartdriver.constants import *
from smartdriver.gui import MyGame
import numpy as np
import tensorflow as tf

def main():
    """ Main method """
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED) 
    window = MyGame(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE, smart=False, show=True, verbose=False, train=True)
    window.setup()
    arcade.run()

    # 20.63

if __name__ == "__main__":
    
    main()