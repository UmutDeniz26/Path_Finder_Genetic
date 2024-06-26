import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QGraphicsRectItem
from scritps.Game_Board import Game_Board
from scritps.Game_Board_Without_GPU import Game_Board as Game_Board_Without_GPU
from scritps.Genetic_Algorithm import Genetic_Algorithm
from scritps.timer import Timer
import time

def main():
    default_objects=[]    
    #easy default_objects = [{'x': 238, 'y': 66, 'width': 87, 'height': 306}, {'x': 459, 'y': 312, 'width': 81, 'height': 383}, {'x': 167, 'y': 469, 'width': 539, 'height': 123}, {'x': 170, 'y': 7, 'width': 186, 'height': 88}, {'x': 143, 'y': 222, 'width': 144, 'height': 50}]
    #beginner default_objects = [{"x":300, "y":300, "width":100, "height":100}]
    #norm default_objects = [{'x': 116, 'y': 14, 'width': 98, 'height': 403}, {'x': 285, 'y': 250, 'width': 90, 'height': 432}, {'x': 393, 'y': 234, 'width': 206, 'height': 42}, {'x': 623, 'y': 272, 'width': 40, 'height': 35}, {'x': 449, 'y': 310, 'width': 52, 'height': 45}, {'x': 562, 'y': 357, 'width': 52, 'height': 160}, {'x': 391, 'y': 68, 'width': 268, 'height': 43}, {'x': 18, 'y': 543, 'width': 118, 'height': 124}, {'x': 139, 'y': 502, 'width': 69, 'height': 23}, {'x': 626, 'y': 393, 'width': 69, 'height': 33}]
    #norm2 
    default_objects = [{'x': 139, 'y': 101, 'width': 119, 'height': 362}, {'x': 301, 'y': 309, 'width': 151, 'height': 374}, {'x': 405, 'y': 105, 'width': 259, 'height': 119}, {'x': 506, 'y': 267, 'width': 74, 'height': 336}, {'x': 625, 'y': 251, 'width': 70, 'height': 31}]
    #hard default_objects = [{'x': 84, 'y': 17, 'width': 85, 'height': 506}, {'x': 240, 'y': 257, 'width': 113, 'height': 435}, {'x': 295, 'y': 230, 'width': 299, 'height': 58}, {'x': 641, 'y': 151, 'width': 60, 'height': 161}, {'x': 615, 'y': 300, 'width': 41, 'height': 88}, {'x': 557, 'y': 381, 'width': 100, 'height': 40}, {'x': 446, 'y': 359, 'width': 52, 'height': 251}, {'x': 537, 'y': 508, 'width': 142, 'height': 81}, {'x': 647, 'y': 463, 'width': 32, 'height': 25}, {'x': 630, 'y': 279, 'width': 22, 'height': 39}, {'x': 621, 'y': 198, 'width': 81, 'height': 132}]
    #expert default_objects = [{'x': 81, 'y': 18, 'width': 102, 'height': 532}, {'x': 257, 'y': 169, 'width': 114, 'height': 518}, {'x': 430, 'y': 21, 'width': 118, 'height': 583}, {'x': 572, 'y': 451, 'width': 65, 'height': 97}, {'x': 616, 'y': 374, 'width': 33, 'height': 67}, {'x': 672, 'y': 510, 'width': 23, 'height': 100}, {'x': 448, 'y': 646, 'width': 190, 'height': 21}, {'x': 578, 'y': 580, 'width': 61, 'height': 36}, {'x': 394, 'y': 338, 'width': 22, 'height': 26}, {'x': 211, 'y': 296, 'width': 21, 'height': 33}, {'x': 226, 'y': 70, 'width': 25, 'height': 40}, {'x': 273, 'y': 127, 'width': 24, 'height': 33}, {'x': 340, 'y': 69, 'width': 23, 'height': 20}, {'x': 361, 'y': 126, 'width': 17, 'height': 22}, {'x': 401, 'y': 119, 'width': 12, 'height': 25}, {'x': 201, 'y': 410, 'width': 26, 'height': 20}, {'x': 232, 'y': 503, 'width': 18, 'height': 12}, {'x': 136, 'y': 564, 'width': 37, 'height': 22}, {'x': 180, 'y': 623, 'width': 34, 'height': 36}, {'x': 83, 'y': 630, 'width': 40, 'height': 47}, {'x': 44, 'y': 559, 'width': 12, 'height': 38}, {'x': 376, 'y': 556, 'width': 23, 'height': 34}]
    

    GPU = True if input("Do you want to use GPU? (y/n): ") == "y" else False
    BOARD_SIZE = (700, 700)
    data_path = "log/results.hdf5" 

    # Hybrid settings
    hybrid_flag = False
    hybrid_interval = 40
    hybrid_GPU_coeff = 0.25

    timer = Timer()

    # Set flags
    save_flag = not GPU
    load_flag = GPU   

    if GPU is False:
        load_flag = True if input("Do you want to load? (y/n): ") == "y" else False

    model = Genetic_Algorithm(
        learning_rate = 0.1, 
        mutation_rate = 0.1,
        select_per_epoch      = 1 if GPU else 30,
        generation_multiplier = 1 if GPU else 100,
        save_flag= save_flag,
        load_flag= load_flag,
        sample_speed = 20,
        dataframe_path=data_path,
        constant_learning_parameter_flag=True,
        exit_reached_flag=False,
        hybrid_flag=hybrid_flag,
        not_learning_flag= GPU,
        GPU_board_flag=GPU,
        #timer=timer
    )

    # If GPU is available
    if GPU:
        app = QApplication(sys.argv)
        board = Game_Board(
            obstacles=object_dist_to_Qt(default_objects),
            board_size=BOARD_SIZE, 
            model=model, 
        )
        board.show()
        sys.exit(app.exec_())
    
    # If GPU is not available
    else:
        board = Game_Board_Without_GPU(
            obstacles=default_objects,
            board_size=BOARD_SIZE, 
            model=model, 
        )

        # Run infinite loop for only non-GPU model
        if not hybrid_flag:
            while True:
                board.update_samples()

        # Run hybrid mode
        else:
            # Create the GPU boardlen(self.evulation_results):15
            app = QApplication(sys.argv)
            board_GPU = Game_Board(
                obstacles=object_dist_to_Qt(default_objects),
                hybrid_interval = hybrid_interval * hybrid_GPU_coeff,
                board_size=BOARD_SIZE,
                model=model,
                timer=timer
            )        

            # Start the timers
            timer.start_new_timer("with-GPU")
            timer.start_new_timer("non-GPU")    
            while True:
                # Run the non-GPU model 
                timer_non_GPU_index = timer.get_timer_index("non-GPU")
                timer.reset_timer("non-GPU");timer.update_timers()
                
                # While the timer is not up, run the model
                while timer.timers[timer_non_GPU_index]["current"] < hybrid_interval:
                    model.main_loop()
                    timer.update_timers()
                
                # Run the GPU model and its timer
                timer.reset_timer("with-GPU")
                board_GPU.show();app.exec_()   

def object_dist_to_Qt(object_list):
    return_dist = []
    for obj in object_list:
        return_dist.append(QGraphicsRectItem(obj["x"], obj["y"], obj["width"], obj["height"]))
    return return_dist

if __name__ == "__main__":
    main()