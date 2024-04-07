import numpy as np
import random
import sys
import os
import time
import pandas as pd
import Common
import pandas_operations

sys.path.insert(0, "path_finder")
try:
    from scritps.Sample import Sample
except:
    from Sample import *

class Genetic_Algorithm:
    def __init__(
            self, learning_rate, mutation_rate, select_per_epoch, board_size,
            generation_multiplier, board_object=None, sample_speed=20,
            dataframe_path=None, save_flag = False, load_flag = True
            ):

        # The board size is the size of the game board
        self.board_size = board_size
        if board_object is not None:
            self.board = board_object
            self.assign_board_attributes()
        self.no_change_counter = 0; self.hold_best_score = 0;self.epoch = 0

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
        self.max_move_count = int(self.board_size[0]/self.sample_speed * self.board_size[1]/self.sample_speed)
        
        # The moves container is a list of lists that contains the angles of the moves
        self.moves_container = np.zeros(shape=(self.max_move_count,self.select_per_epoch) ,dtype=dict)

        # Population is the list of samples
        self.population = np.array( 
            [
                Sample(
                    self.board_size, self.sample_speed,external_controls=np.zeros(shape=self.max_move_count,dtype=np.float32)
                ) 
            
            for _ in range(self.population_size)],
            dtype=Sample
        )

        # It contains control histories, scores and status of the samples
        self.evulation_results = np.array(
            [{
                "sample": sample,
            } for sample in self.population] * 2, # * 2 means -> [newGen(400),oldGen(400)]
            dtype=dict
        )
        self.init_moves_container()

        # The dataframe path is the path of the dataframe
        self.dataframe_path = dataframe_path
        self.save_flag = save_flag
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
                "control_history": row["sample"].control_history,
            })
        self.update_moves_container( self.sort_evulation_results() )
        
        # Metadata operations
        Common.print_dict(metadata)
        self.epoch = int(metadata["EPOCH_COUNT"])

        input("Press any key to continue...")
            
        
    def assign_board_attributes(self):
        self.board_width, self.board_height = self.board.board_width, self.board.board_height
        self.board_size = (self.board_width, self.board_height)

    def get_dataframe(self):
        self.sort_evulation_results()
        return pd.DataFrame(self.evulation_results)

    def change_parameters(self, learning_rate, mutation_rate):
        
        self.no_change_counter = 0
        if learning_rate is not None:
            learning_rate = learning_rate + learning_rate * 0.2
            self.learning_rate = learning_rate

        if mutation_rate is not None:
            self.mutation_rate = mutation_rate - 0.01
        
        if mutation_rate < 0.01:
            self.mutation_rate = self.mutation_rate_original

        if learning_rate > 0.80:
            self.learning_rate = self.learning_rate_original

    def sort_evulation_results(self):
        self.evulation_results = sorted(self.evulation_results, key=lambda x: x["sample"].get_score(), reverse=True)
        return self.evulation_results
        
    def mutation(self, angles):
        # For each move, a random angle will be chosen
        
        #opt1
        """
        """
        for angle_index in range(self.moves_container.shape[0]):
            if self.learning_rate == 0.0:
                angles = self.evulation_results[0]["sample"].control_history
                break
            angles[angle_index] = np.random.choice(self.moves_container[angle_index])["angle"]
        #opt2
        """
        moves_container_index = np.random.random_integers(0,self.moves_container.shape[1]-1)        
        angles = self.moves_container[:,moves_container_index]
        """

        # %mutation_rate of the angles will be mutated    
        mask_enable_filter =  np.random.uniform(0, 1, len(angles)) < self.mutation_rate

        # The mask will be multiplied by the angles and then by the learning rate
        mask_coefficients = np.random.uniform(-self.learning_rate, self.learning_rate, len(angles))

        # angles = angles + angles * mask
        return angles + angles * (mask_coefficients * mask_enable_filter)
    
    # Returns totally random angles for the first generation
    def init_moves_container(self):
        
        for index_i,elem in enumerate(self.moves_container):
            for index_j, angle in enumerate(elem):
                self.moves_container[index_i][index_j] = {"angle":angle, "score":0}
        """
        [{
            "Status":"initial",
            "score": 0,
            "control_history": [random.uniform(0, 360) for i in range(
                self.select_per_epoch * self.generation_multiplier
            )]
        } for _ in range(self.select_per_epoch)]
        """
        
    def handle_status(self, sample, color, index, move_cnt):
        if color is not None:
            return_data = self.kill_sample(sample)
            
            if color == "#00ff00":
                sample.status = "Success"
                return_data["score"] = 1000
                return_data["score"] += 1 / ( 100 * move_cnt )
                
                if self.learning_rate == 0.0:
                    self.reset_samples()
            elif color == "Out of bounds":
                sample.status = "Dead"
            elif color == "#000000": 
                sample.status = "Dead"
            
            sample.set_score(return_data["score"])
            sample.final_move_cnt = move_cnt

            return_data = {"sample": sample}

            
            # Shift the old generation's results to the new generation's results
            self.evulation_results[index + self.population_size] = self.evulation_results[index]
            self.evulation_results[index] = return_data
            
    def reset_model(self):
        self.evulation_results.clear();self.population.clear()
        self.moves_container.clear();self.sorted_moves_container.clear()
        
    # It creates the new generation's samples
    def create_new_generation_samples(self):
        for i in range(self.population_size):
            # Prepare the sample
            sample = self.population[i]
            sample.controls = self.mutation(self.evulation_results[i % self.select_per_epoch]["sample"].control_history)
            
        
    # It creates the new generation
    def loop(self):
        # Sort the evulation results
        self.sort_evulation_results()
        
        # Update the moves container
        self.update_moves_container()
        self.create_new_generation_samples()

        self.print_epoch_info() if self.epoch % 1 == 0 else None
        self.print_epoch_summary(self.evulation_results) if self.epoch % 1 == 0 else None


        #X
        best_score = self.evulation_results[0]["sample"].get_score()
        if best_score == self.hold_best_score:
            self.no_change_counter += 1
        else:
            self.no_change_counter = 0
            self.learning_rate = self.learning_rate_original
            self.mutation_rate = self.mutation_rate_original
        self.hold_best_score = best_score
        #X

        # Saving the best scores
        if  self.epoch % 20 == 0 and \
            len(self.evulation_results) > 2*self.select_per_epoch*self.generation_multiplier and \
            self.save_flag:
            print("Saving the data");time.sleep(1)

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
        #X

    def print_epoch_summary(self, sorted_results):
        move_count_of_best = sorted_results[0]["sample"].final_move_cnt
        ratio_success = len([result for result in sorted_results if result["sample"].status == "Reached the end"]) / len(sorted_results)
        print(f"""
        STATISTICS:
            BEST SCORE       : {sorted_results[0]["sample"].get_score():10.4f} | NUMBER OF SAMPLES: {len(self.population):7.1f} |
            NUMBER OF RESULTS: {len(self.evulation_results):10.1f} | NO CHANGE COUNTER: {self.no_change_counter:7.1f}
            RATIO OF SUCCESS : {ratio_success:10.2f} | AVERAGE SCORE    : {self.calculate_average_score():7.2f}
            MOVE COUNT OF BEST: {move_count_of_best:7.1f}
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
        return sum([result["sample"].get_score() for result in self.evulation_results]) / len(self.evulation_results)
            
    def get_population(self):
        return self.population

    # Remove the sample from the scene and the samples list then return the data of the sample
    def kill_sample(self, sample):
        # Before removing the sample, get the control history of the sample
        final_result = sample.get_control_history_and_final_score()
        # Remove the sample from the scene and the samples list

        final_result.update({"sample": sample})
        
        return final_result
    
    # It kills the sample and returns the control history and the final score of the sample    
    def reset_samples(self):
        for sample in self.population:
            final_result = self.kill_sample(sample)
            final_result.update({"Status": "Reset"})
            self.evulation_results.append(final_result)
        
        del self.population
        self.population = np.array( 
            [Sample(self.board_size, self.sample_speed) for _ in range(self.population_size)],
            dtype=Sample
        )

    
    # because of the algorithm, moves container always sorted
    def update_moves_container(self):
        # For each sample, update the moves container ( only the first {select_per_epoch} samples will be considered )
        for index_i,result in enumerate(self.evulation_results[0:self.select_per_epoch]):
            for index_j,angle in enumerate(result["sample"].control_history):
                self.moves_container[index_j][index_i] = {"angle":angle, "score":result["sample"].get_score()}    
    
    
def main():
    model = Genetic_Algorithm(
        learning_rate=0.1,
        mutation_rate=0.1,
        select_per_epoch=10,
        generation_multiplier=10,
        board_size=(700, 700),
        sample_speed=20
    )

    model.prepare_next_generation()
    model.sort_evulation_results()
    model.sort_moves_container()

    print("The model is working correctly")

if __name__ == "__main__":
    # Test the Genetic_Algorithm class
    main()