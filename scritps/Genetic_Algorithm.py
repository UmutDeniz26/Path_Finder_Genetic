import numpy as np;import random;import sys;import os
import time;import copy;import pandas as pd

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
            exit_reached_flag = False, not_learning_flag = False,
            timer = None
            ):

        # The board size is the size of the game board
        if board_object is not None:
            self.board = board_object;self.assign_board_attributes()
        
        self.timer = timer
        if timer is not None:
            self.timer.start_new_timer("Main Timer")
            self.timer.start_new_timer("Initialization Timer")

        # The no change counter is the counter that holds the number of epochs that the best score does not change
        self.no_change_counter = 0; self.hold_best_score = 0;self.epoch = 0

        # The sample speed is the speed of the samples
        self.sample_speed = sample_speed
        
        # The learning rate is the how much of the angle will be changed
        # The mutation rate is the probability of a gene to be mutated
        self.learning_rate = learning_rate
        self.mutation_rate = mutation_rate

        # Hold the original values of the learning rate and the mutation rate
        self.learning_rate_original = self.learning_rate
        self.mutation_rate_original = self.mutation_rate

        # It selects best {select_per_epoch} samples from the previous generation
        # It generates {select_per_epoch*multiplier} samples for the next generation
        self.select_per_epoch = select_per_epoch 
        self.generation_multiplier = generation_multiplier 

        # The population size is the number of samples in the population
        self.population_size = select_per_epoch * generation_multiplier
        self.no_change_limit = int((200 / self.population_size) * 40) + 5
        
        # Population is the list of samples, evulation_results is the list of the results of the samples
        self.evulation_results = []
        self.population = []
        self.best_control = []
        
        # The dataframe path is the path of the dataframe
        self.dataframe_path = dataframe_path

        # Flags
        self.save_flag = save_flag
        self.not_learning_flag = not_learning_flag
        self.exit_reached_flag = exit_reached_flag
        
        if os.path.exists(dataframe_path) and load_flag:
            self.upload_dataframe()
        
        timer.stop_timer("Initialization Timer") if timer is not None else None

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
            self.add_result_dict(
                row = row
            )
        self.sort_evulation_results()
        self.best_control = self.evulation_results[0]["controls"]
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

    # It creates the new generation
    def model_loop(self):
        self.timer.start_new_timer("Model Loop Timer") if self.timer is not None else None

        # If there is no result, create the initial generation
        self.initialize_generation() if len(self.population)==0 else None

        # Sort the evulation results
        self.sort_evulation_results()

        # Manipulates the learning rate and the mutation rate
        self.handle_learning_parameters()

        self.timer.start_new_timer("New Generation Timer") if self.timer is not None else None
        # Updates population with the new generation
        self.create_new_generation_samples()
        self.timer.stop_timer("New Generation Timer") if self.timer is not None else None
        
        # Save the process managment
        self.save_process_managment()
        
        self.print_epoch_info() if self.epoch % 1 == 0 else None
        self.print_epoch_summary() if self.epoch % 1 == 0 else None

        self.timer.stop_timer("Model Loop Timer") if self.timer is not None else None
        self.timer.print_timers() if self.timer is not None else None
        
        self.timer.print_ratio("Main Timer", "New Generation Timer") if self.timer is not None else None
        self.timer.print_ratio("Mutation Timer", "Mutation Timer Part 2") if self.timer is not None else None

    # It creates the new generation's samples
    def create_new_generation_samples(self):
        for index,_ in enumerate(self.population):
            # Get mutated angles which is created by the evulation_results dict
            self.population[index].set_controls(
                assign_mode="copy",
                external_controls = self.mutation( len( self.population[index].controls ) ) if self.not_learning_flag == False else self.best_control
            )
            self.population[index].set_status("Alive")
            
    def mutation(self,angles_len):
        self.timer.start_new_timer("Mutation Timer") if self.timer is not None else None
        self.timer.start_new_timer("Mutation Timer Part 2") if self.timer is not None else None
        
        # If there is no result, return an empty list
        if angles_len == 0:
            return []

        # For each move, a random angle will be chosen
        self.sort_evulation_results()
        angles = []
        self.timer.stop_timer("Mutation Timer Part 2") if self.timer is not None else None
        try:
            for index in range(angles_len):    
                angles.append(
                    np.random.choice(self.evulation_results[0:self.select_per_epoch])["controls"][index])
        except:
            pass

        # If the model is not learning, the angles will be the best angles
        angles = self.evulation_results[0]["controls"] if self.not_learning_flag else angles

        # %mutation_rate of the angles will be mutated    
        mask_enable_filter = np.random.uniform(0, 1, len(angles)) < self.mutation_rate

        # The mask will be multiplied by the angles and then by the learning rate
        mask_coefficients = np.random.uniform(-self.learning_rate,self.learning_rate,len(angles))

        # The angles are mutated
        mutated_angles = ( angles + angles * ( mask_coefficients * mask_enable_filter ) ).astype(int)

        self.timer.stop_timer("Mutation Timer") if self.timer is not None else None
        return mutated_angles
    
    # Returns totally random angles for the first generation
    def initialize_generation(self):
        for i in range(self.population_size):
            self.population.append(Sample(
                self.board_size, 
                self.sample_speed,
            ))
 
    def handle_status(self, sample, color):
        if color is not None:
            return_data = sample.kill_sample_get_score()
            
            if color == "#00ff00":
                return_data["sample"].set_score(1000 + 1 / return_data["sample"].final_move_count)
                return_data.update({"status": "Reached the end"})
            elif color == "Out of bounds":
                return_data.update({"status": "Out of bounds"})
            elif color == "#000000": 
                return_data.update({"status": "Hit the obstacle"})
            return_data["sample"].set_status(return_data["status"])
            
            self.add_result_dict(
                sample=return_data["sample"],
                status=return_data["status"]
            )

    # Manipulates the learning rate and the mutation rate
    def handle_learning_parameters(self):
        best_score = self.evulation_results[0]["score"] if len(self.evulation_results) > 0 else 0
        if best_score == self.hold_best_score:
            self.no_change_counter += 1
        else:
            self.no_change_counter = 0
            self.learning_rate, self.mutation_rate = self.learning_rate_original, self.mutation_rate_original
        self.hold_best_score = best_score

        if self.no_change_counter > self.no_change_limit:
            self.change_parameters( self.learning_rate, self.mutation_rate )
    
    def save_process_managment(self):
        if  self.epoch % (self.no_change_limit * 2) == 0 and self.save_flag and \
            len(self.evulation_results) > 0: 
            print("Saving the progress...");time.sleep(1)
            # Create the metadata
            metadata = {
                "LEARNING_RATE": self.learning_rate,"MUTATION_RATE": self.mutation_rate,
                "SELECT_PER_EPOCH": self.select_per_epoch,"MULTIPLIER": self.generation_multiplier,
                "BOARD_SIZE": (self.board_width, self.board_height),"EPOCH_COUNT": self.epoch,
            }

            # Save the dataframe
            pandas_operations.save_dataframe_hdf5(
                self.get_dataframe(), save_lim=10000, path=self.dataframe_path, metadata=metadata
            )

            # Exit if the best score is greater than 1000 and the exit_reached_flag is True
            sys.exit(0) if self.evulation_results[0]["score"] > 1000 and self.exit_reached_flag else None

    def reset_model(self):
        self.evulation_results.clear()
        self.reset_samples()
        
    def add_result_dict(self, sample=None, status=None, row=None):
        if row is not None:
            add_dict = {}
            for key, value in row.items():
                add_dict.update({key: value})
            self.evulation_results.append(add_dict)
        elif sample is not None and status is not None:
            self.evulation_results.append({
                "controls": sample.get_controls(),
                "score": sample.get_score(),
                "status": status,
                "final_move_count": sample.final_move_count,
                "ID": sample.ID,
            })
        else:
            raise ValueError("Sample or status are None")
        pass

    def sort_evulation_results(self):
        self.evulation_results.sort(key=lambda x: x["score"], reverse=True)
        self.evulation_results = self.evulation_results[:self.select_per_epoch]

    # It kills the sample and returns the control history and the final score of the sample    
    def reset_samples(self):
        for sample in self.population:
            final_result = sample.kill_sample_get_score()
            final_result.update({"status": "Reset"})
            self.add_result_dict(sample=final_result["sample"], status=final_result["status"])

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
        if len(self.evulation_results) == 0:
            return

        move_count_of_best = self.evulation_results[0]["final_move_count"]
        ratio_success = len(
                [result for result in self.evulation_results if result["status"] == "Reached the end"]
            ) / len(self.evulation_results)
        print(f"""
        STATISTICS:
            BEST SCORE        : {self.evulation_results[0]["score"]:15.10f} | MOVE COUNT OF BEST: {move_count_of_best:7.3f} 
            NUMBER OF RESULTS : {len(self.evulation_results):15.0f} | AVERAGE SCORE     : {self.calculate_average_score():7.3f}
            RATIO OF SUCCESS  : {ratio_success:15.11f} | 
        """)
        
    def print_epoch_info(self):
        self.epoch += 1

        print(f"""      
        ===================================== Epoch: "{self.epoch} " =====================================
        CONSTANTS:
            LEARNING_RATE   : {self.learning_rate:7.2f} | MUTATION_RATE     : {self.mutation_rate:7.2f} 
            SELECT_PER_EPOCH: {self.select_per_epoch:7.1f} | MULTIPLIER        : {self.generation_multiplier:7.1f}
            SAMPLE_SPEED    : {self.sample_speed:7.1f} | BOARD_SIZE        : {self.board_width}x{self.board_height}
            NO CHANGE LIMIT : { self.no_change_limit:7.1f} | NO CHANGE COUNTER : {self.no_change_counter:7.1f}
            REFRESH RATE    : {self.board.refresh_rate if hasattr(self.board, "refresh_rate") else 0:7.1f}
        """)

if __name__ == "__main__":
    # Test the Genetic_Algorithm class
    pass