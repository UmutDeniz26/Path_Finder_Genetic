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
    #ez default_objects = [{'x': 238, 'y': 66, 'width': 87, 'height': 306}, {'x': 459, 'y': 312, 'width': 81, 'height': 383}, {'x': 167, 'y': 469, 'width': 539, 'height': 123}, {'x': 170, 'y': 7, 'width': 186, 'height': 88}, {'x': 143, 'y': 222, 'width': 144, 'height': 50}]
    #1 default_objects = [{"x":300, "y":300, "width":100, "height":100}]
    #norm default_objects = [{'x': 116, 'y': 14, 'width': 98, 'height': 403}, {'x': 285, 'y': 250, 'width': 90, 'height': 432}, {'x': 393, 'y': 234, 'width': 206, 'height': 42}, {'x': 623, 'y': 272, 'width': 40, 'height': 35}, {'x': 449, 'y': 310, 'width': 52, 'height': 45}, {'x': 562, 'y': 357, 'width': 52, 'height': 160}, {'x': 391, 'y': 68, 'width': 268, 'height': 43}, {'x': 18, 'y': 543, 'width': 118, 'height': 124}, {'x': 139, 'y': 502, 'width': 69, 'height': 23}, {'x': 626, 'y': 393, 'width': 69, 'height': 33}]
    default_objects = [{'x': 139, 'y': 101, 'width': 119, 'height': 362}, {'x': 301, 'y': 309, 'width': 151, 'height': 374}, {'x': 405, 'y': 105, 'width': 259, 'height': 119}, {'x': 506, 'y': 267, 'width': 74, 'height': 336}, {'x': 625, 'y': 251, 'width': 70, 'height': 31}]

    GPU = False
    BOARD_SIZE = (700, 700)
    data_path = "log/results.hdf5" 
    hybrid_flag = False
    hybrid_interval = 80

    timer = Timer()

    save_flag = False if GPU else True
    load_flag = True if GPU else False    
    
    modal = Genetic_Algorithm(
        learning_rate = 0 if GPU else 0.1, 
        mutation_rate = 0 if GPU else 0.1,
        select_per_epoch=1 if GPU else 100,
        generation_multiplier=1 if GPU else 10,
        save_flag= save_flag,  
        load_flag= load_flag,
        sample_speed = 20,
        dataframe_path=data_path,
        exit_reached_flag=False,
        not_learning_flag= True if GPU else False,
        #timer=timer
    )

    # If GPU is available
    if GPU:
        app = QApplication(sys.argv)
        board = Game_Board(
            board_size=BOARD_SIZE, model=modal, 
            obstacles=object_dist_to_Qt(default_objects)
        )
        board.show()
        sys.exit(app.exec_())
    
    # If GPU is not available
    else:
        board = Game_Board_Without_GPU(
            board_size=BOARD_SIZE, model=modal, 
            obstacles=default_objects,
        )

        if hybrid_flag:
            app = QApplication(sys.argv)
            board_GPU = Game_Board(
                board_size=BOARD_SIZE, model=modal, 
                obstacles=object_dist_to_Qt(default_objects),
                timer=timer, hybrid_interval = hybrid_interval//8
            )
        

            timer.start_new_timer("with-GPU")
            timer.start_new_timer("non-GPU")    
            while True:
                timer_non_GPU_index = timer.get_timer_index("non-GPU")
                timer.reset_timer("non-GPU")
                while True:
                    board.update_samples()
                    timer.update_timers()
                    if timer.timers[timer_non_GPU_index]["current"] > hybrid_interval:
                        break
                print("non-GPU is finished")
                time.sleep(1)
                
                timer.reset_timer("with-GPU")
                board_GPU.show();app.exec_()
                print("with-GPU is finished")
                time.sleep(1)
        else:
            while True:
                board.update_samples()

def object_dist_to_Qt(object_list):
    return_dist = []
    for obj in object_list:
        return_dist.append(QGraphicsRectItem(obj["x"], obj["y"], obj["width"], obj["height"]))
    return return_dist

if __name__ == "__main__":
    main()
