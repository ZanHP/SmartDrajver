from collections import deque

from arcade.sprite_list import SpriteList
from smartdriver.constants import *
import tensorflow as tf
import numpy as np
from tensorflow import keras
import random
import neat
import os
from smartdriver.player import Player


from smartdriver.population import Population as SmartPopulation

epsilon = 1 # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start
max_epsilon = 1 # You can't explore more than 100% of the time
min_epsilon = 0.01 # At a minimum, we'll always explore 1% of the time


class GeneticAgent():
    def __init__(self, state_shape, weights=None, track=None, smart=None):
        #self.actions = np.array(["L", "R", ""])#, "U", "D"])
        self.actions = np.array(["L","", "R"])#, "U", "D"])
        self.state_shape = state_shape

        self.finished = False
        self.total_training_rewards = 0

        self.track = track
        self.smart = smart

        self.max_time_steps = 60 * 30

        self.gen = 1
        self.max_gen = 100
        self.player_sprites = SpriteList()

        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, "config")

        config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

        p = SmartPopulation(config)

        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)

        p.start_generation(self.max_gen)
        genomes, config = p.get_genome_and_config()

        self.p = p

        self.genomes = genomes

        self.player_neat = {}
        self.time_step = 0

        self.create_nets(genomes, config)
       
        #self.update_target_model()


    def create_players(self, number):
        self.player_sprites = SpriteList()
        for _ in range(number):
            player_sprite = Player(":resources:images/space_shooter/playerShip1_orange.png", SPRITE_SCALING, self.track.checkpoints[0], self.track, self.smart)
            self.player_sprites.append(player_sprite)

        
    
    def create_nets(self, genomes, config):
        self.create_players(len(genomes))
        self.player_neat = {}
        i = 0
        for _, g in genomes:
            net = neat.nn.FeedForwardNetwork.create(g, config)
            g.fitness = 0

            self.player_neat[self.player_sprites[i]] = {
                "genome": g,
                "net": net
            }
            i += 1

    def draw(self):
        #print(len(self.player_sprites))
        for player_sprite in self.player_sprites:
            player_sprite.draw()


    def get_model(self,player_sprite):
        return self.player_neat[player_sprite]["net"]

    

    def do_summary_fitness_scores(self):

        for player_sprite in self.player_sprites:
            self.player_neat[player_sprite]["genome"].fitness = self.get_fitness_score(player_sprite)



    def do_training_step(self):

        #model = self.model
        #replay_memory = self.replay_memory
        
        #total_training_rewards = self.total_training_rewards
        #target_model = self.target_model

        self.time_step += 1
        if self.time_step == self.max_time_steps:
            self.gen += 1
            self.time_step = 1

            # player je prišel do konca kroga

            self.do_summary_fitness_scores()

            self.p.evaluate()

            self.p.start_generation(self.max_gen)
            genome, config = self.p.get_genome_and_config()
            self.create_nets(genome, config)

            print(len(self.player_sprites), len(self.player_neat))

            return True

        for player_sprite in self.player_sprites:
            
            if player_sprite.finished:
                continue
                
            state = player_sprite.get_current_state()
            net = self.get_model(player_sprite)

            predicted = net.activate((state))
            action = self.actions[np.argmax(predicted)]
            #print(action)
            #encoded_reshaped = np.array((player_sprite.get_current_state(),))
            #redicted = model.predict(encoded_reshaped).flatten()

            # odločimo se, kaj bo naslednja poteza
            #action = self.actions[np.argmax(predicted)]

            # glede na izbrano potezo se premaknemo
            is_not_finished = player_sprite.next_move_and_update(action)
            is_finished = not is_not_finished
            if is_finished:
                player_sprite.finished = True
                player_sprite.finish_time = self.time_step
            


        #print("Train iteration: ", self.train_iteration)                    
        return False


    def get_action_ind(self, action):
        #print("get_action:", action,np.where(self.actions == action))
        return np.where(self.actions == action)

    def encode_state(self, state, n_dims):
        return state




    def get_fitness_score(self, player_sprite):
        if player_sprite.finished:
            return - player_sprite.finish_time
        else:
            return - (self.max_time_steps +  player_sprite.distance_to_nn_checkpoint())


