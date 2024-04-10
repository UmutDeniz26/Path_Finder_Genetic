import os
import sys
import time
import logging

import numpy as np
import pandas as pd

try:
    import scritps.Common as Common
    import scritps.pandas_operations as pandas_operations
    from scritps.Sample_qt import Sample_qt
    from scritps.Sample import Sample

except ImportError:
    import Common
    import pandas_operations
    from Sample_qt import Sample_qt
    from Sample import Sample

class Genetic_Algorithm:
    def __init__(
        self: object,
        learning_rate: float,
        mutation_rate: float,
        select_per_epoch: int,
        generation_multiplier: int,
        board_object: object = None,
        sample_speed: int = 20,
        dataframe_path: str = None,
        save_flag: bool = False,
        load_flag: bool = True,
        exit_reached_flag: bool = False,
        not_learning_flag: bool = False,
        hybrid_flag: bool = False,
        constant_learning_parameter_flag: bool = False,
        GPU_board_flag: bool = False,
        timer: object = None
    ):
        """
        Initialize a Genetic_Algorithm object.

        Args:
            learning_rate (float): The learning rate determining how much the angle will change.
            mutation_rate (float): The probability of a gene being mutated.
            select_per_epoch (int): Number of best samples selected per epoch.
            generation_multiplier (int): Multiplier for generating new samples.
            board_object (object): The object representing the game board (optional).
            sample_speed (int, optional): The speed of the samples (default: 20).
            dataframe_path (str, optional): Path to the dataframe (default: None).
            save_flag (bool, optional): Flag indicating whether to save progress (default: False).
            load_flag (bool, optional): Flag indicating whether to load data (default: True).
            exit_reached_flag (bool, optional): Flag indicating whether to exit when reaching a certain score (default: False).
            not_learning_flag (bool, optional): Flag indicating whether the model is not learning (default: False).
            hybrid_flag (bool, optional): Flag indicating whether the model is hybrid (default: False).
            constant_learning_parameter_flag (bool, optional): Flag indicating whether the learning parameters are constant (default: False).
            GPU_board_flag (bool, optional): Flag indicating whether the board is a GPU board (default: False).
            timer (object, optional): Timer object for tracking execution time (default: None).

        Returns:
            None
        """

        # Logging operations
        logging.basicConfig(filename='log/app.log', filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')
        logging.getLogger().setLevel(logging.DEBUG)

        # Initialize attributes related to the board object
        if board_object is not None:
            self.board = board_object
            self.assign_board_attributes()
        
        # Initialize timer if provided
        self.timer = timer
        if timer is not None:
            self.timer.start_new_timer("Main Timer")
            self.timer.start_new_timer("Initialization Timer")

        # Initialize counters and flags
        self.no_change_counter = 0
        self.hold_best_score = 0
        self.loop_count = 0
        self.epoch = 0

        # Set the model parameters
        self.sample_speed = sample_speed
        self.learning_rate = learning_rate
        self.mutation_rate = mutation_rate
        self.mutation_rate_sign = -1
        self.learning_rate_original = self.learning_rate
        self.mutation_rate_original = self.mutation_rate

        # Set parameters for generating new samples
        self.select_per_epoch = select_per_epoch 
        self.generation_multiplier = generation_multiplier 
        self.population_size = select_per_epoch * generation_multiplier

        # No change limit controls the change_parameters function
        # Refresh rate is the time to reset the samples
        self.no_change_limit = int(10000 / self.population_size) + 5

        # Initialize lists and flags
        self.evulation_results = []
        self.average_controls = []
        self.population = []
        self.best_control = []
        self.dataframe_path = dataframe_path
        self.save_flag = save_flag
        self.hybrid_flag = hybrid_flag
        self.not_learning_flag = not_learning_flag
        self.exit_reached_flag = exit_reached_flag
        self.constant_learning_parameter_flag = constant_learning_parameter_flag

        # If hybrid flag is set, the model will be run in GPU_enabled mode
        self.GPU_board_flag = True if self.hybrid_flag else GPU_board_flag

        # Load dataframe if path exists and load_flag is True
        if os.path.exists(dataframe_path) and load_flag:
            self.upload_dataframe()
        
        # Stop initialization timer if timer is provided
        self.timer.stop_timer("Initialization Timer") if timer is not None else None
    
    # Each trigger of the main loop provides the working of the model
    def main_loop(self): 
        self.timer.start_new_timer("Main Loop Timer") if self.timer is not None else None

        living_samples = self.get_living_samples()
        if len(living_samples):
            if living_samples[0].move_counter > self.refresh_rate:
                self.reset_samples()
            else:
                self.timer.start_new_timer("Update Living Samples") if self.timer is not None else None
                self.update_living_samples()
                self.timer.stop_timer("Update Living Samples") if self.timer is not None else None
        else:
            self.progress_to_next_epoch()
        
        self.timer.stop_timer("Main Loop Timer") if self.timer is not None else None
            
    # It creates the new generation
    def progress_to_next_epoch(self):
        self.timer.start_new_timer("Progress_to_next_epoch") if self.timer is not None else None

        # If there is no result, create the initial generation
        self.initialize_generation() if len(self.get_population())==0 else None

        # Sort the evulation results
        self.sort_evulation_results()

        # Updates population with the new generation
        self.timer.start_new_timer("New Generation Timer") if self.timer is not None else None
        self.create_new_generation_samples()
        self.timer.stop_timer("New Generation Timer") if self.timer is not None else None

        # Manage saving process and learning parameters
        self.save_process_managment()
        self.handle_learning_parameters()

        self.print_epoch_info() if self.epoch % 1 == 0 else None
        self.print_epoch_summary() if self.epoch % 1 == 0 else None

        # Timer operations
        if self.timer is not None:
            self.timer.stop_timer("Progress_to_next_epoch")
            self.timer.print_timers()
            self.timer.print_ratio("Progress_to_next_epoch", "Main Loop Timer")
            self.timer.print_ratio("Update Living Samples", "Main Loop Timer")
            self.timer.print_ratio("update part", "Update Living Samples")
        
    def update_living_samples(self):
        for sample in self.get_living_samples():
            x, y = sample.move()
            color = self.board.get_color(x, y)
            self.handle_status(sample, color)

    def calculate_average_controls(self):
        self.average_controls = [ 
            np.mean(
                [result["controls"][i] for result in self.evulation_results if i < len(result["controls"])]
            ) for i in range(int(self.refresh_rate))
        ]
        self.average_controls = [int(angle) for angle in self.average_controls if angle < 720]
        return self.average_controls

    # It creates the new generation's samples
    def create_new_generation_samples(self):
            
        for index,samp in enumerate(self.population):
            
            self.population[index].set_controls(
                assign_mode="copy",
                external_controls = self.mutation( )
            )
            self.population[index].set_status("Alive")
            
    def mutation(self):
        self.timer.start_new_timer("Mutation Timer") if self.timer is not None else None
        
        if len(self.average_controls) == 0:
            return []
        
        # The angles will be the average of the best angles
        #angles = self.calculate_average_controls()

        #random_indicies = np.random.choice(self.select_per_epoch, angles_len)
        random_indices = np.random.randint(0, self.select_per_epoch, int(self.refresh_rate) )

        angles = []
        for j, i in enumerate(random_indices):
            if j >= len(self.evulation_results[i]["controls"]):
                break
            angles.append(self.evulation_results[i]["controls"][j])


        mutation_limit = 0 if np.random.uniform(0, 1) < 0.1 else\
                np.random.randint( 0, len(angles) )
        
        # If the model is not learning, the angles will be the best angles
        angles = self.evulation_results[0]["controls"] if self.not_learning_flag else angles

        # %mutation_rate of the angles will be mutated    
        if mutation_limit:
            mask_enable_filter = np.concatenate(
                    (np.zeros(mutation_limit), np.ones( len(angles) - mutation_limit ))
                )
        else:
            mask_enable_filter = np.random.uniform(0, 1, len(angles)) < self.mutation_rate

        # The mask will be multiplied by the angles and then by the learning rate
        mask_coefficients = np.random.uniform(-self.learning_rate,self.learning_rate,len(angles))

        # The angles are mutated
        mutated_angles = np.mod(angles + angles * (mask_coefficients * mask_enable_filter), 360).astype(int)

        self.timer.stop_timer("Mutation Timer") if self.timer is not None else None
        return mutated_angles
    
    # Returns totally random angles for the first generation
    def initialize_generation(self):
        for i in range(self.population_size):
            
            if self.GPU_board_flag:
                self.population.append(Sample_qt(
                    self.board_size, 
                    self.sample_speed,
                ))
            else:
                self.population.append(Sample(
                    self.board_size, 
                    self.sample_speed,
                ))
 
    def handle_status(self, sample, color):
        if color is not None:
            # Kill the sample and get the score
            return_data = sample.kill_sample_get_score()

            if color == "Reached the end":
                return_data["sample"].set_score(
                    1000 + 1000 / return_data["sample"].final_move_count + return_data["sample"].get_score()
                )
            
            return_data["sample"].set_status(color)
            self.add_result_dict(
                sample=return_data["sample"],
                status=color
            )

    # Manipulates the learning rate and the mutation rate
    def handle_learning_parameters(self):
        if not self.constant_learning_parameter_flag:
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
        if self.epoch % (self.no_change_limit * 4) == 0 and self.save_flag and \
            len(self.evulation_results) > 0:
            logging.info("Saving the progress...")
            print("Saving the progress...")
            time.sleep(1)
            # Create the metadata
            metadata = {
                "LEARNING_RATE": self.learning_rate, "MUTATION_RATE": self.mutation_rate,
                "SELECT_PER_EPOCH": self.select_per_epoch, "MULTIPLIER": self.generation_multiplier,
                "BOARD_SIZE": (self.board_width, self.board_height), "EPOCH_COUNT": self.epoch,
            }

            # Save the dataframe
            pandas_operations.save_dataframe_hdf5(
                self.get_dataframe(), save_lim=10000, path=self.dataframe_path, metadata=metadata
            )

            # Exit if the best score is greater than 1000 and the exit_reached_flag is True
            sys.exit(0) if self.evulation_results[0]["score"] > 1000 and self.exit_reached_flag else None

    # Manipulates evulation_results due to the uploaded dataframe
    def upload_dataframe(self):
        if not os.path.exists(self.dataframe_path):
            message = "The specified file path does not exist."
            logging.error(message)
            Common.exit_with_print(message)
        
        try:
            result_pd, metadata = pandas_operations.load_dataframe_hdf5(self.dataframe_path)
        except Exception as e:
            logging.error(f"Error loading dataframe: {e}")
            Common.exit_with_print(e)

        if result_pd is None or metadata is None:
            message = "Failed to load dataframe or metadata."
            logging.error(message)
            Common.exit_with_print(message)
            
        # Update the model due to the uploaded dataframe ( Manipulate the evulation_results )
        self.reset_model()
        for _, row in result_pd.iterrows():
            self.add_result_dict(row=row)
        self.sort_evulation_results()
        self.best_control = self.evulation_results[0]["controls"]
        if self.not_learning_flag:
            self.evulation_results = [self.evulation_results[0]]

        # Metadata operations
        Common.print_dict(metadata)
        self.epoch = int(metadata["EPOCH_COUNT"]) + 1

        input("Press any key to continue...")

        logging.info("Dataframe uploaded successfully.")

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
            # if there is a result with same score, then continue
            if len(self.evulation_results) > 0 and self.evulation_results[0]["score"] == sample.get_score():
                return

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

    def change_parameters(self, learning_rate, mutation_rate):
        self.no_change_counter = 0

        if self.evulation_results[0]["score"] > 1000:
            self.mutation_rate_sign = -1
        else:
            self.mutation_rate_sign = 1

        if learning_rate is not None:
            learning_rate = learning_rate + 0.1
            self.learning_rate = learning_rate

        if mutation_rate is not None:
            mutation_rate = mutation_rate + 0.1 * self.mutation_rate_sign
            self.mutation_rate = mutation_rate

        if (mutation_rate > 0.90 and self.mutation_rate_sign == 1) or\
            (mutation_rate < 0.001 and self.mutation_rate_sign == -1):
            self.mutation_rate = self.mutation_rate_original 
            
        if learning_rate > 0.90:
            self.learning_rate = self.learning_rate_original

    # It kills the sample and returns the control history and the final score of the sample    
    def reset_samples(self):
        for sample in self.get_living_samples():
            final_result = sample.kill_sample_get_score()
            final_result["sample"].set_status("Reset")         
            self.add_result_dict(sample=final_result["sample"], status="Reset")

    def assign_board_attributes(self, board):
        self.refresh_rate = ( 6 * board.distance_between_start_and_end / self.sample_speed )
        self.board_width, self.board_height = board.board_width, board.board_height
        self.max_move_count = self.board_width * 10 // self.sample_speed
        self.board_size = (self.board_width, self.board_height)

    def get_dataframe(self):
        self.sort_evulation_results()
        return pd.DataFrame(self.evulation_results)

    def calculate_average_score(self):
        return sum([result["score"] for result in self.evulation_results]) / len(self.evulation_results)
            
    def get_population(self):
        return self.population

    def get_living_samples(self):
        return [elem for elem in self.get_population() if elem.status == "Alive"]

    def print_epoch_summary(self):
        if len(self.evulation_results) == 0:
            return

        move_count_of_best = self.evulation_results[0]["final_move_count"]
        ratio_success = len(
                [result for result in self.evulation_results if result["status"] == "Reached the end"]
            ) / len(self.evulation_results)
        print(f"""
        STATISTICS:
            BEST SCORE        : {self.evulation_results[0]["score"]:15.10f} | MOVE COUNT OF BEST  : {move_count_of_best:7.3f} 
            NUMBER OF RESULTS : {len(self.evulation_results):15.0f} | AVERAGE SCORE       : {self.calculate_average_score():15.11f}
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
            REFRESH RATE    : {self.refresh_rate:7.1f} | SAMOLE COUNT      : {len(self.population):7.1f}
        """)

if __name__ == "__main__":
    pass