import arcade
from smartdriver.constants import *
from smartdriver.gui import MyGame, GeneticGame
import numpy as np
import tensorflow as tf
import neat
import os


GEN = 0


def main(genomes = [], config = []):
    """ Main method """
    global GEN
    GEN += 1
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED) 

    window = GeneticGame(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE, smart=True, show=True, verbose=False, train=True, genomes=genomes, config=config)
    window.setup()
    
    arcade.run()
     
    arcade.close_window()

    for _,genome in genomes:
        print(genome.fitness)

    # 20.63

if __name__ == "__main__":


    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config")

    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    

    winner = p.run(main, 10)
