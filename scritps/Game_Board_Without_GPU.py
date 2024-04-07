import sys
import random
import time
import numpy as np
import math
import os

import pandas as pd
sys.path.insert(0, "path_finder")

try:
    import Genetic_Algorithm as gn
except:
    import scritps.Genetic_Algorithm as gn

class Game_Board():
     
    def __init__(self, board_size, model, sample_speed, obstacles):
        super().__init__()
        
        # Essential attributes
        self.board_width, self.board_height = board_size

        # 0 -> Empty, 1 -> Obstacle, 2 -> Start, 3 -> End
        self.board = np.zeros(shape=(self.board_width, self.board_height), dtype=np.int8)
        
        # Set the sample speed and the loop count
        self.sample_speed = sample_speed
        self.loop_count = 0

        # Default choices
        self.current_obstacle = None;self.paused = False

        # Initialization of counters & arrays
        self.frame_count = 0;self.epoch_count = 0
        self.obstacles = obstacles
        self.population = []
        
        # Set the End Points
        self.init_end_points()
        
        # Initialize the end points and create the first generation
        self.distance_between_start_and_end = math.sqrt(
            (self.board_width - self.end_point[0])**2 + (self.board_height - self.end_point[1])**2
        )     
        self.refresh_rate = 8 * self.distance_between_start_and_end / self.sample_speed
        
        # Upload board attributes to the model
        self.model = model
        self.model.board = self
        self.model.assign_board_attributes()

        self.model.prepare_next_generation()
        self.init_screen()
        print("Game Board is initialized correctly")
        
        while True:
            self.update_samples()
           
    def update_samples(self):
        self.frame_count += 1

        # If the frame count is greater than the reset limit, reset the samples
        if self.refresh_rate < self.frame_count:
            self.model.reset_samples();self.frame_count = 0
            

        # If the number of samples is less than the number of samples, create a new generation
        if not self.paused:
            self.loop_count += 1
            if len(self.model.get_population()) == 0:
                if self.model.no_change_counter > 10:
                    self.model.change_parameters(
                        self.model.learning_rate, self.model.learning_rate
                        )

                self.model.prepare_next_generation()
                return
                
            for sample in self.model.get_population():
                new_x, new_y = np.array(sample.move()).astype(int)
                new_position_color = self.get_color((new_x, new_y))
                self.model.handle_status(sample, new_position_color)
    
    def get_color(self, position):
        x, y = position
        if x < 0 or x >= self.board_width or y < 0 or y >= self.board_height:
            return "Out of bounds"
        else:
            val = self.board[x,y]
            if val == 1: # Black
                return "#000000"
            elif val == 3: # Green
                return "#00ff00"
            else:
                return None

    # Inıtialize the screen (obstacles etc.)
    def init_screen(self):
        for obstacle in self.obstacles:
            self.board[
                obstacle["x"]:obstacle["x"]+obstacle["width"],
                obstacle["y"]:obstacle["y"]+obstacle["height"]
            ] = 1
            
    def init_end_points(self):
        self.base_size = (self.board_width // 20, self.board_height // 20)
        self.end_point = (self.board_width - self.board_width // 20, self.board_height // 2)

        self.board[self.end_point[0] - self.base_size[0] // 2:self.end_point[0] + self.base_size[0] // 2,
                   self.end_point[1] - self.base_size[1] // 2:self.end_point[1] + self.base_size[1] // 2] = 3

if __name__ == "__main__":
    #Test
    data_path = "log/results.hdf5" 
    modal = gn.Genetic_Algorithm(
        learning_rate=0.1, mutation_rate=0.1, select_per_epoch=20, generation_multiplier=50, sample_speed=20,
        dataframe_path=data_path, save_flag= True, load_flag= False
    )

    BOARD_SIZE = (700, 700)

    default_objects =  [
        {'x': 146, 'y': 145, 'width': 424, 'height': 128},
        {'x': 205, 'y': 406, 'width': 222, 'height': 245},
        {'x': 309, 'y': 257, 'width': 40, 'height': 92},
        {'x': 493, 'y': 321, 'width': 63, 'height': 174},
        {'x': 584, 'y': 214, 'width': 43, 'height': 116},
        {'x': 612, 'y': 308, 'width': 90, 'height': 30}
    ]

    board = Game_Board(
        board_size=BOARD_SIZE, model=modal, 
        sample_speed=20, obstacles=default_objects,
    )
