import numpy as np
import random
import sys
import math
from PyQt5.QtCore import Qt

sys.path.insert(0, "path_finder")
from scritps.Sample import Sample

class Genetic_Algorithm:
    def __init__(self, learning_rate, mutation_rate, select_per_epoch, generation_multiplier, board_size, sample_speed=20):

        # The board size is the size of the game board
        self.board_width, self.board_height = board_size
        self.board_size = board_size
        self.no_change_counter = 0; self.hold_best_score = 0

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

        # It selects best {select_per_epoch} samples from the previous generation
        self.select_per_epoch = select_per_epoch 
        
        # It generates {select_per_epoch*multiplier} samples for the next generation
        self.generate_multiplier = generation_multiplier 

        # The moves container is a list of lists that contains the angles of the moves
        self.moves_container = [[]]
        self.sorted_moves_container = [[]]

        # It contains control histories, scores and status of the samples
        self.evulation_results = []
        self.sorted_evulation_results = []


    def change_parameters(self, learning_rate, mutation_rate):
        if self.no_change_counter > 20:
            self.no_change_counter = 0
            if learning_rate is not None:
                learning_rate = learning_rate + learning_rate * 0.01
                self.learning_rate = learning_rate
            if mutation_rate is not None:
                mutation_rate = mutation_rate - mutation_rate * 0.01
                self.mutation_rate = mutation_rate

        return learning_rate, mutation_rate
    
    def get_sorted_evulation_results(self):
        self.sorted_evulation_results = sorted(self.evulation_results, key=lambda x: x["score"], reverse=True)
        return self.sorted_evulation_results

    def mutation(self, angles):
        # For each move, a random angle will be chosen
        for index,_ in enumerate(self.sorted_moves_container):
            if index < len(angles):
                angles[index] = np.random.choice(self.sorted_moves_container[index][0:self.select_per_epoch])["angle"]
        
        # Convert the angles to a numpy array
        angles = np.array(angles)

        # %mutation_rate of the angles will be mutated    
        mask_enable_filter = [random.uniform(0, 1) < self.mutation_rate for _ in angles]

        # The mask will be multiplied by the angles and then by the learning rate
        mask_coefficients = [random.uniform(-self.learning_rate, self.learning_rate) for _ in angles]
        
        # The mask is ready to multiply with the angles
        mask = np.array(mask_coefficients) * np.array(mask_enable_filter)

        # The angles are mutated
        mutated_angles = angles + angles * mask

        return mutated_angles
    
    # Returns totally random angles for the first generation
    def initial_generation(self):
        return [{
            "Status":"inital",
            "score": 0,
            "control_history": [random.uniform(0, 360) for i in range(
                self.select_per_epoch * self.generate_multiplier
            )]
        } for _ in range(self.select_per_epoch)]

    def handle_status(self, sample, color):
        if color is not None:
            return_data = self.kill_sample(sample)
            if color == Qt.green:
                return_data["score"] *= 100
                return_data.update({"Status": "Reached the end"})                
            elif color == "Out of bounds":
                return_data["score"] /= 10
                return_data.update({"Status": "Out of bounds"})
            elif color == Qt.black:
                return_data["score"] /= 10
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
        self.sort_moves_container()
        self.create_new_generation_samples(sorted_results)

        best_score = sorted_results[0]["score"]
        self.no_change_counter = self.no_change_counter + 1 if best_score == self.hold_best_score else 0
        self.hold_best_score = best_score
        
        self.print_epoch_summary(sorted_results, best_score)


    def print_epoch_summary(self, sorted_results, best_score):
        ratio_success = len([result for result in sorted_results if result["Status"] == "Reached the end"]) / len(sorted_results)
        print(f"""
        STATISTICS:
            BEST SCORE       : {best_score:7.2f} | NUMBER OF SAMPLES: {len(self.population):7.1f} |
            NUMBER OF RESULTS: {len(self.evulation_results):7.1f} | NO CHANGE COUNTER: {self.no_change_counter:7.1f}
            RATIO OF SUCCESS : {ratio_success:7.2f} | AVERAGE SCORE    : {self.calculate_average_score():7.2f}
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

    # This function sorts the moves container
    def sort_moves_container(self):
        sorted_local = []
        for moves in self.moves_container:
            sorted_local.append(sorted(moves, key=lambda x: x["score"], reverse=True))
        self.sorted_moves_container = sorted_local

        return self.sorted_moves_container
    
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
    model.get_sorted_evulation_results()
    model.sort_moves_container()

    print("The model is working correctly")

if __name__ == "__main__":
    # Test the Genetic_Algorithm class
    main()