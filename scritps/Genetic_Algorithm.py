import numpy as np
import random
import sys
import os
import time
import pandas as pd

sys.path.insert(0, "path_finder")
try:
    from scritps.Sample import Sample
    import scritps.Common as Common
    import scritps.pandas_operations as pandas_operations

except:
    from Sample import *
    import Common
    import pandas_operations


class Genetic_Algorithm:
    def __init__(
            self, learning_rate, mutation_rate, select_per_epoch, 
            generation_multiplier, board_object=None, sample_speed=20,
            dataframe_path=None, save_flag = False, load_flag = True,
            exit_reached_flag = False, not_learning_flag = False
            ):

        # The board size is the size of the game board
        if board_object is not None:
            self.board = board_object;self.assign_board_attributes()
        
        # The no change counter is the counter that holds the number of epochs that the best score does not change
        self.no_change_counter = 0; self.hold_best_score = 0;self.epoch = 0

        # The sample speed is the speed of the samples
        self.sample_speed = sample_speed
        
        # The population size is the number of samples in the population
        self.population_size = select_per_epoch * generation_multiplier
        self.population = []
        
        # The learning rate is the how much of the angle will be changed
        # The mutation rate is the probability of a gene to be mutated
        self.learning_rate = learning_rate
        self.mutation_rate = mutation_rate

        # Hold the original values of the learning rate and the mutation rate
        self.learning_rate_original = self.learning_rate
        self.mutation_rate_original = self.mutation_rate

        # It selects best {select_per_epoch} samples from the previous generation
        self.select_per_epoch = select_per_epoch 
        
        # It generates {select_per_epoch*multiplier} samples for the next generation
        self.generation_multiplier = generation_multiplier 

        # It contains control histories, scores and status of the samples
        self.evulation_results = []

        # The dataframe path is the path of the dataframe
        self.dataframe_path = dataframe_path
        self.not_learning_flag = not_learning_flag
        self.save_flag = save_flag;self.exit_reached_flag = exit_reached_flag
        
        if os.path.exists(dataframe_path) and load_flag:
            self.upload_dataframe()

    # Manipulates evulation_results due to the uploaded dataframe
    def upload_dataframe(self):
        if not os.path.exists(self.dataframe_path):
            print("The file is not found")
            return
        
        result_pd, metadata = pandas_operations.load_dataframe_hdf5(
            self.dataframe_path
        )
        
        # Update the model due to the uploaded dataframe ( Manipulate the evulation_results )
        self.reset_model()
        for _, row in result_pd.iterrows():
            self.evulation_results.append({
                "score": row["score"],
                "sample": row["sample"],
                "status": row["status"],
                }
            )
        self.sort_evulation_results()
        if self.not_learning_flag:
            self.evulation_results = [self.evulation_results[0]]
                
        # Metadata operations
        Common.print_dict(metadata)
        self.epoch = int(metadata["EPOCH_COUNT"]) + 1

        input("Press any key to continue...")
            
    def change_parameters(self, learning_rate, mutation_rate):
    
        self.no_change_counter = 0
        if learning_rate is not None:
            learning_rate = learning_rate + learning_rate * 0.2
            self.learning_rate = learning_rate
        
        if mutation_rate is not None:
            mutation_rate = mutation_rate - mutation_rate * 0.2
            self.mutation_rate = mutation_rate
        
        self.mutation_rate = self.mutation_rate_original if mutation_rate < 0.01 else mutation_rate
        self.learning_rate = self.learning_rate_original if learning_rate > 0.80 else learning_rate
        

    def sort_evulation_results(self):
        self.evulation_results.sort(key=lambda x: x["score"], reverse=True)
        
    def mutation(self):

        #opt1
        # For each move, a random angle will be chosen
        self.sort_evulation_results()
        angles = np.zeros(self.max_move_count, dtype=int)
        for index in range(self.max_move_count):
            angles[index] = np.random.choice(
                    self.evulation_results[0:self.select_per_epoch]
                )["sample"].controls[index]

        angles = self.evulation_results[0]["sample"].controls if self.not_learning_flag else angles

        # %mutation_rate of the angles will be mutated    
        mask_enable_filter = np.random.uniform(0, 1, len(angles)) < self.mutation_rate

        # The mask will be multiplied by the angles and then by the learning rate
        mask_coefficients = np.random.uniform(-self.learning_rate,self.learning_rate,len(angles))
        
        # The mask is ready to multiply with the angles
        mask = mask_coefficients * mask_enable_filter

        # The angles are mutated
        mutated_angles = (angles + angles * mask).astype(int)

        return mutated_angles
    
    # Returns totally random angles for the first generation
    def initial_generation(self):
        if len(self.population) == 0:
            for i in range(self.population_size):
                self.population.append(Sample(
                    self.board_size, 
                    self.sample_speed,
                    external_controls=[random.randint(0, 360) for j in range(self.max_move_count)]
                    ))

        self.evulation_results = [{
            "sample": self.population[i],
            "score": 0,
            "status": "initial",
        } for i in range(self.population_size)]

    def handle_status(self, sample, color):
        if color is not None:
            return_data = self.kill_sample(sample)
            
            if color == "#00ff00":
                return_data["score"] = 1000
                return_data["score"] += 1 / len(return_data["sample"].controls)
                return_data.update({"status": "Reached the end"})
                        
            elif color == "Out of bounds":
                return_data.update({"status": "Out of bounds"})
            elif color == "#000000": 
                return_data.update({"status": "Hit the obstacle"})
            
            self.evulation_results.append(return_data)
        
    def reset_model(self):
        self.evulation_results.clear()
        self.reset_samples()
        
    # It creates the new generation's samples
    def create_new_generation_samples(self):
        for index,samp in enumerate(self.population):
            sample = self.population[index]
            # Get mutated angles which is created by the evulation_results dict
            sample.controls = self.mutation()
            
            # Add the sample to the population
            #self.population.append(sample)
        
    # It creates the new generation
    def prepare_next_generation(self):
        
        # If there is no result, create the initial generation
        if len(self.evulation_results) < self.select_per_epoch:
            self.initial_generation()
        self.sort_evulation_results()
        
        # Update the moves container
        self.create_new_generation_samples()

        best_score = self.evulation_results[0]["score"]
        if best_score == self.hold_best_score:
            self.no_change_counter += 1
        else:
            self.no_change_counter = 0
            self.learning_rate = self.learning_rate_original
            self.mutation_rate = self.mutation_rate_original
        self.hold_best_score = best_score

        if self.no_change_counter > 20:
            self.change_parameters(
                self.learning_rate, self.mutation_rate
            )

        if  self.epoch % 40 == 0 and self.save_flag and \
            len(self.evulation_results) > 2*self.select_per_epoch*self.generation_multiplier:
            
            metadata = {
                "LEARNING_RATE": self.learning_rate,"MUTATION_RATE": self.mutation_rate,
                "SELECT_PER_EPOCH": self.select_per_epoch,"MULTIPLIER": self.generation_multiplier,
                "BOARD_SIZE": (self.board_width, self.board_height),"EPOCH_COUNT": self.epoch,
            }

            # save the dataframe
            pandas_operations.save_dataframe_hdf5(
                self.get_dataframe(), save_lim=10000, path=self.dataframe_path, metadata=metadata
            )
            sys.exit(0) if self.evulation_results[0]["score"] > 1000 and self.exit_reached_flag else None

        self.print_epoch_info() if self.epoch % 1 == 0 else None
        self.print_epoch_summary() if self.epoch % 1 == 0 else None

    # Remove the sample from the scene and the samples list then return the data of the sample
    def kill_sample(self, sample):
        # Before removing the sample, get the control history of the sample
        final_result = sample.get_control_history_and_final_score()
        sample.status = "Dead"
        return final_result
    
    # It kills the sample and returns the control history and the final score of the sample    
    def reset_samples(self):
        for sample in self.population:
            final_result = self.kill_sample(sample)
            final_result.update({"status": "Reset"})
            self.evulation_results.append(final_result)
            
    def assign_board_attributes(self):
        self.board_width, self.board_height = self.board.board_width, self.board.board_height
        self.max_move_count = self.board_width * 10 // self.sample_speed
        self.board_size = (self.board_width, self.board_height)

    def get_dataframe(self):
        self.sort_evulation_results()
        return pd.DataFrame(self.evulation_results)

    def calculate_average_score(self):
        return sum([result["score"] for result in self.evulation_results]) / len(self.evulation_results)
            
    def get_population(self):
        return self.population

    def print_epoch_summary(self):
        move_count_of_best = len(self.evulation_results[0]["sample"].controls)
        ratio_success = len(
                [result for result in self.evulation_results if result["status"] == "Reached the end"]
            ) / len(self.evulation_results)
        print(f"""
        STATISTICS:
            BEST SCORE       : {self.evulation_results[0]["score"]:15.10f} | NUMBER OF SAMPLES: {len(self.population):7.1f} |
            NUMBER OF RESULTS: {len(self.evulation_results):15.0f} | NO CHANGE COUNTER: {self.no_change_counter:7.0f}
            RATIO OF SUCCESS : {ratio_success:15.11f} | AVERAGE SCORE    : {self.calculate_average_score():7.3f}
            MOVE COUNT OF BEST: {move_count_of_best:15.0f} |
        """)
        
    def print_epoch_info(self):
        self.epoch += 1
        print(f"""      
        ===================================== Epoch: "{self.epoch} " =====================================
        CONSTANTS:
            LEARNING_RATE   : {self.learning_rate:7.2f} | MUTATION_RATE: {self.mutation_rate:7.2f} 
            SELECT_PER_EPOCH: {self.select_per_epoch:7.1f} | MULTIPLIER   : {self.generation_multiplier:7.1f}
            SAMPLE_SPEED    : {self.sample_speed:7.1f} | BOARD_SIZE   : {self.board_width}x{self.board_height}
            SAMPLE_COUNT    : {len(self.get_population()):7.1f} | NO CHANGE COUNTER: {self.no_change_counter:7.1f}
        """)

if __name__ == "__main__":
    # Test the Genetic_Algorithm class
    pass