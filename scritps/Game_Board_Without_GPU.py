import sys
import random
import time
import numpy as np
import math
import os

import pandas as pd
sys.path.insert(0, "path_finder")

import Genetic_Algorithm as gn
import Sample

class Game_Board():
     
    def __init__(self, board_size, model, sample_speed, obstacles, data_path):
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
        self.data_path = data_path

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

        if data_path is not None:
            self.upload_dataframe(data_path)
        
        save_lim=20
        save_cnt=0
        while True:
            save_cnt+=1
            self.update_samples()
            if save_cnt > save_lim:
                save_lim += save_lim
                
                result_pd = pd.DataFrame(self.model.sorted_evulation_results[:save_lim])
                result_pd = result_pd.drop( result_pd[result_pd['Status'] == 'inital'].index)
                
                store = pd.HDFStore(self.data_path)
                store.put('results', result_pd)
                
                store.get_storer('results').attrs.metadata = {
                    "LEARNING_RATE": self.model.learning_rate,
                    "MUTATION_RATE": self.model.mutation_rate,
                    "SELECT_PER_EPOCH": self.model.select_per_epoch,
                    "MULTIPLIER": self.model.generate_multiplier,
                    "BOARD_SIZE": f"{self.board_width}x{self.board_height}",
                    "EPOCH_COUNT": self.epoch_count,
                    "TIME": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                store.close()

                #result_pd.to_csv("log/results.csv", index=False)

    def upload_dataframe(self, data_path):
        if not os.path.exists(data_path):
            print("The file does not exist!")
            return
        try:
            with pd.HDFStore(data_path) as store:
                result_pd = store['results']
                metadata = store.get_storer('results').attrs.metadata

                # Generate sorted_evulation_results from the csv file
                self.model.evulation_results = []
                for index, row in result_pd.iterrows():
                    #string_numbers = row["control_history"].replace('[', '').replace(']', '')
                    #numbers_list = [float(num) for num in string_numbers.split(', ')]
                    self.model.evulation_results.append({
                        "ID": row["ID"],
                        "score": row["score"],
                        "Status": row["Status"],
                        "control_history": row["control_history"],
                    })
                self.model.update_moves_container( self.model.get_sorted_evulation_results() )
                self.model.sort_moves_container()
                print("The dataframe file is uploaded successfully")
                # Print attrs of the csv file
                for key, value in metadata.items():
                    print(f"{key}: {value}")
                self.epoch_count = metadata["EPOCH_COUNT"]
                input("Press any key to continue...")
                
        except Exception as e:
            print(f"An error occured while uploading the file: {e}")
            input("Press any key to continue...")
    def update_samples(self):
        self.frame_count += 1

        # If the frame count is greater than the reset limit, reset the samples
        if self.refresh_rate < self.frame_count:
            self.frame_count = 0
            self.model.reset_samples()

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


    # Print the epoch information
    def print_epoch_info(self):
        self.epoch_count += 1
        print(f"""      
        ===================================== Epoch: "{self.epoch_count} " =====================================
        CONSTANTS:
            LEARNING_RATE   : {self.model.learning_rate:7.2f} | MUTATION_RATE: {self.model.mutation_rate:7.2f} 
            SELECT_PER_EPOCH: {self.model.select_per_epoch:7.1f} | MULTIPLIER   : {self.model.generate_multiplier:7.1f}
            SAMPLE_SPEED    : {self.sample_speed:7.1f} | BOARD_SIZE   : {self.board_width}x{self.board_height}
            SAMPLE_COUNT    : {len(self.model.get_population()):7.1f} | NO CHANGE COUNTER: {self.model.no_change_counter:7.1f}
        """)

import matplotlib.pyplot as plt

#Test 
modal = gn.Genetic_Algorithm(
    learning_rate=0.1, mutation_rate=0.1, select_per_epoch=10, generation_multiplier=50, sample_speed=20
)

BOARD_SIZE = (700, 700)

obj_width = 200
obj_height = 400

default_object = {
    "x": (BOARD_SIZE[0])//2-(obj_width//2),
    "y": (BOARD_SIZE[1])//2-(obj_height//2),
    "width": obj_width,
    "height": obj_height
}


board = Game_Board(
    board_size=BOARD_SIZE, model=modal, 
    sample_speed=20, obstacles=[default_object],
    data_path="log/results.hdf5"
)
