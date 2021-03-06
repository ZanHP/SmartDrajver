from collections import deque
from smartdriver.track import Track

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
from smartdriver.checkpoint import Checkpointer as SmartCheckpointer

epsilon = 1 # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start
max_epsilon = 1 # You can't explore more than 100% of the time
min_epsilon = 0.01 # At a minimum, we'll always explore 1% of the time


class GeneticAgent():
    def __init__(self, state_shape, TRACK=None, smart=None, start_gen=None):
        #self.actions = np.array(["L", "R", ""])#, "U", "D"])
        self.actions = np.array(["UL","UR", "DR", "DL", "D", "U", "L", "R"])#, "U", "D"])
        self.state_shape = state_shape

        self.finished = False
        self.total_training_rewards = 0

        self.TRACK = TRACK
        self.smart = smart

        self.max_time_steps = 60 * 13

        self.gen = 1
        self.max_gen = 100
        self.player_sprites = SpriteList()

        self.filename = 'gen-res.txt'
        self.filewriter = open(self.filename,'w')

        self.start_gen = start_gen
        if start_gen:
            self.gen = start_gen

        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, "config")

        config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

        if start_gen:
            p = SmartCheckpointer.restore_checkpoint(f'neat-checkpoint-{start_gen-1}')
        else:
            p = SmartPopulation(config)

        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        p.add_reporter(SmartCheckpointer(10))

        

        p.start_generation(self.max_gen)
        genomes, config = p.get_genomes_and_config()

        self.p = p

        self.genomes = genomes

        self.player_neat = {}
        self.time_step = 0

        self.create_nets(genomes, config)
       
        #self.update_target_model()

    def max_fitness_score(self):
        genomes, _ = self.p.get_genomes_and_config()

        # this could be better sigh
        max_val = -9999999
        for _,g in genomes:
            if g.fitness > max_val:
                max_val = g.fitness

        return max_val
    
    def lowest_time(self):

        # this could be better sigh
        min_val = None
        for player in self.player_sprites:
            if player.finished:
                if min_val is None or player.finish_time < min_val:
                    min_val = player.finish_time
        return min_val

    def create_players(self, number):
        self.player_sprites = SpriteList()
        for _ in range(number):

            track = Track(self.TRACK)
            player_sprite = Player(":resources:images/space_shooter/playerShip1_orange.png", SPRITE_SCALING, track.checkpoints[0], track, self.smart)
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

            # player je pri??el do konca kroga
            print("hej evaluate")
            self.do_summary_fitness_scores()
            print(self.max_fitness_score())


            max_fitness = self.max_fitness_score()
            min_time = self.lowest_time()
            min_time = 0 if min_time is None else min_time
            write_string = f"{max_fitness}, {min_time}\n"
            self.filewriter.write(write_string)
            self.filewriter.close()
            self.filewriter = open(self.filename,'a')
        

            self.p.evaluate()
            self.p.start_generation(self.max_gen)
            genomes, config = self.p.get_genomes_and_config()

            
            self.create_nets(genomes, config)
            print(len(self.player_sprites), len(self.player_neat))
            return True
            

        for player_sprite in self.player_sprites:
            
            if player_sprite.finished:
                continue
                
            state = player_sprite.get_current_state()
            
            normalize_state = np.array(state) / 180

            net = self.get_model(player_sprite)

            predicted = net.activate((normalize_state))
            action = self.actions[np.argmax(predicted)]
            #print(action)
            #encoded_reshaped = np.array((player_sprite.get_current_state(),))
            #redicted = model.predict(encoded_reshaped).flatten()

            # odlo??imo se, kaj bo naslednja poteza
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
        max_val = 4_000
        if player_sprite.finished:
            return max_val - player_sprite.finish_time
        else:
            penalty = self.max_time_steps +  player_sprite.distance_to_finish()
            return max_val - penalty



