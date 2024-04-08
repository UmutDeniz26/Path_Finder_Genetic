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
            exit_reached_flag = False
            ):

        # The board size is the size of the game board
        if board_object is not None:
            self.board = board_object;self.assign_board_attributes()
        
        # The no change counter is the counter that holds the number of epochs that the best score does not change
        self.no_change_counter = 0; self.hold_best_score = 0;self.epoch = 0

        # Population is the list of samples
        self.population = []

        # The sample speed is the speed of the samples
        self.sample_speed = sample_speed

        # The board size is the size of the game board
        self.population_size = select_per_epoch * generation_multiplier
        
        # The learning rate is the how much of the angle will be changed
        self.learning_rate = learning_rate
        
        # The mutation rate is the probability of a gene to be mutated
        self.mutation_rate = mutation_rate

        # Hold the original values of the learning rate and the mutation rate
        self.learning_rate_original = self.learning_rate
        self.mutation_rate_original = self.mutation_rate

        # It selects best {select_per_epoch} samples from the previous generation
        self.select_per_epoch = select_per_epoch 
        
        # It generates {select_per_epoch*multiplier} samples for the next generation
        self.generation_multiplier = generation_multiplier 

        # The moves container is a list of lists that contains the angles of the moves
        self.moves_container = [[]]
        self.sorted_moves_container = [[]]

        # It contains control histories, scores and status of the samples
        self.evulation_results = []
        self.sorted_evulation_results = []

        # The dataframe path is the path of the dataframe
        self.dataframe_path = dataframe_path
        self.save_flag = save_flag
        self.exit_reached_flag = exit_reached_flag
        if dataframe_path is not None and os.path.exists(dataframe_path) and load_flag:
            self.upload_dataframe()

    def upload_dataframe(self):
        if self.dataframe_path is None or not os.path.exists(self.dataframe_path):
            print("The file is not found")
            return
        
        result_pd, metadata = pandas_operations.load_dataframe_hdf5(
            self.dataframe_path
        )
        
        # Update the model due to the uploaded dataframe
        self.reset_model()
        for _, row in result_pd.iterrows():
            self.evulation_results.append({
                "ID": row["ID"],
                "score": row["score"],
                "Status": row["Status"],
                "control_history": row["control_history"],
            })
        if self.learning_rate == 0.0:
            self.evulation_results = [self.evulation_results[0]]
                
        self.update_moves_container( self.get_sorted_evulation_results() )
        
        # Metadata operations
        Common.print_dict(metadata)
        self.epoch = int(metadata["EPOCH_COUNT"])

        input("Press any key to continue...")
            
        
    def assign_board_attributes(self):
        self.board_width, self.board_height = self.board.board_width, self.board.board_height
        self.board_size = (self.board_width, self.board_height)

    def get_dataframe(self):
        return pd.DataFrame(self.sorted_evulation_results)

    def change_parameters(self, learning_rate, mutation_rate):
    
        self.no_change_counter = 0
        if learning_rate is not None:
            learning_rate = learning_rate + learning_rate * 0.2
            self.learning_rate = learning_rate
        
        if mutation_rate is not None:
            mutation_rate = mutation_rate - mutation_rate * 0.2
            self.mutation_rate = mutation_rate
        
        if mutation_rate < 0.01:
            self.mutation_rate = self.mutation_rate_original

        if learning_rate > 0.80:
            self.learning_rate = self.learning_rate_original
    
    def get_sorted_evulation_results(self):
        self.sorted_evulation_results = sorted(self.evulation_results, key=lambda x: x["score"], reverse=True)
        return self.sorted_evulation_results

    def mutation(self, angles):

        #opt1
        """
        """
        # For each move, a random angle will be chosen
        for index,_ in enumerate(self.sorted_moves_container):
            # if learning rate is 0 then dont select randomly, only select best ones
            if self.learning_rate == 0.0 and index < len(angles):
                angles = self.sorted_evulation_results[0]["control_history"]
                break

            if index < len(angles):
                angles[index] = np.random.choice(self.sorted_moves_container[index][0:self.select_per_epoch])["angle"]
        
        #opt2 sort and trim moves_container
        """
        for index, moves in enumerate(self.moves_container):
            new_moves = sorted(moves, key = lambda x: x["score"], reverse=True)[0:self.select_per_epoch]
            self.moves_container[index] = new_moves
        np_moves_container = np.array(self.moves_container)
        random_index = random.randint(0, np_moves_container.shape[1]-1)
        angles = [elem["angle"] for elem in  np_moves_container[:, random_index]]
        """

        # Convert the angles to a numpy array
        angles = np.array(angles)

        # %mutation_rate of the angles will be mutated    
        mask_enable_filter = [random.uniform(0, 1) < self.mutation_rate for _ in angles]

        # The mask will be multiplied by the angles and then by the learning rate
        mask_coefficients = [random.uniform(-self.learning_rate, self.learning_rate) for _ in angles]
        
        # The mask is ready to multiply with the angles
        mask = np.array(mask_coefficients) * np.array(mask_enable_filter)

        # The angles are mutated
        mutated_angles = (angles + angles * mask).astype(int)

        return mutated_angles
    
    # Returns totally random angles for the first generation
    def initial_generation(self):
        return [{
            "Status":"inital",
            "score": 0,
            "control_history": [random.randint(0, 360) for i in range(
                self.select_per_epoch * self.generation_multiplier
            )]
        } for _ in range(self.select_per_epoch)]

    def handle_status(self, sample, color):
        if color is not None:
            return_data = self.kill_sample(sample)
            
            if color == "#00ff00":
                return_data["score"] = 1000
                return_data["score"] += 1 / len(return_data["control_history"])
                return_data.update({"Status": "Reached the end"})
                        
            elif color == "Out of bounds":
                return_data.update({"Status": "Out of bounds"})
            elif color == "#000000": 
                return_data.update({"Status": "Hit the obstacle"})
            
            self.evulation_results.append(return_data)
        
    def reset_model(self):
        self.evulation_results.clear();self.population.clear()
        self.moves_container.clear();self.sorted_moves_container.clear()
        self.sorted_evulation_results.clear()
        
    # It creates the new generation's samples
    def create_new_generation_samples(self,sorted_results):
        for i in range(self.population_size):
            # Prepare the sample
            sample = Sample(self.board_size, self.sample_speed)
            sample.controls = self.mutation(sorted_results[i % self.select_per_epoch]["control_history"])
            
            # Add the sample to the population
            self.population.append(sample)
        
    # It creates the new generation
    def prepare_next_generation(self):
        
        # If there is no result, create the initial generation
        if len(self.evulation_results) < self.select_per_epoch:
            self.evulation_results = self.initial_generation()
        sorted_results = self.get_sorted_evulation_results()
        
        # Update the moves container
        self.update_moves_container(sorted_results)
        self.create_new_generation_samples(sorted_results)

        best_score = sorted_results[0]["score"]
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
        
        if  self.epoch % 40 == 0 and \
            len(self.evulation_results) > 2*self.select_per_epoch*self.generation_multiplier and \
            self.save_flag:
            print("Saving the data");#time.sleep(1)

            metadata = {
                "LEARNING_RATE": self.learning_rate,
                "MUTATION_RATE": self.mutation_rate,
                "SELECT_PER_EPOCH": self.select_per_epoch,
                "MULTIPLIER": self.generation_multiplier,
                "BOARD_SIZE": (self.board_width, self.board_height),
                "EPOCH_COUNT": self.epoch,
            }

            pandas_operations.save_dataframe_hdf5(
                self.get_dataframe(), save_lim=10000, path=self.dataframe_path, metadata=metadata
            )
            if sorted_results[0]["score"] > 1000 and self.exit_reached_flag:
                print("The best score is reached")
                sys.exit(0)

        self.print_epoch_info() if self.epoch % 1 == 0 else None
        self.print_epoch_summary(sorted_results) if self.epoch % 1 == 0 else None
        
    def print_epoch_summary(self, sorted_results):
        move_count_of_best = len(sorted_results[0]["control_history"])
        ratio_success = len([result for result in sorted_results if result["Status"] == "Reached the end"]) / len(sorted_results)
        print(f"""
        STATISTICS:
            BEST SCORE       : {sorted_results[0]["score"]:15.10f} | NUMBER OF SAMPLES: {len(self.population):7.1f} |
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

    def calculate_average_score(self):
        return sum([result["score"] for result in self.evulation_results]) / len(self.evulation_results)
            
    def get_population(self):
        return self.population

    # Remove the sample from the scene and the samples list then return the data of the sample
    def kill_sample(self, sample, reset_flag=False):
        # Before removing the sample, get the control history of the sample
        final_result = sample.get_control_history_and_final_score()
        # Remove the sample from the scene and the samples list
        if not reset_flag:
            self.population.remove(sample)
        del sample
        return final_result
    
    # It kills the sample and returns the control history and the final score of the sample    
    def reset_samples(self):
        for sample in self.population:
            final_result = self.kill_sample(sample, reset_flag=True)
            final_result.update({"Status": "Reset"})
            self.evulation_results.append(final_result)
        
        self.population.clear()
    
    # IT AUTOMATICLY SORTS THE MOVES CONTAINER, I DONT UNDERSTAND
    # because it is a list of lists, it sorts the inner lists
    def update_moves_container(self, sorted_evulation_results):
        # For each sample, update the moves container ( only the first {select_per_epoch} samples will be considered )
        for result in sorted_evulation_results[0:self.select_per_epoch]:
            control_history = result["control_history"]
            for index,angle in enumerate(control_history):
                if index >= len(self.moves_container):
                    self.moves_container.append([])
                self.moves_container[index].append({"angle":angle, "score":result["score"], "index":index})
        
        self.sort_moves_container()

    # This function sorts the moves container
    def sort_moves_container(self):
        sorted_local = []
        for moves in self.moves_container:
            sorted_local.append(sorted(moves, key=lambda x: x["score"], reverse=True))
        self.sorted_moves_container = sorted_local

        return self.sorted_moves_container
    

if __name__ == "__main__":
    # Test the Genetic_Algorithm class
    pass