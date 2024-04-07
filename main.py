import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QGraphicsRectItem
from scritps.Game_Board import Game_Board
from scritps.Game_Board_Without_GPU import Game_Board as Game_Board_Without_GPU
from scritps.Genetic_Algorithm import Genetic_Algorithm


def main():    
    default_objects = [
        {'x': 243, 'y': 166, 'width': 274, 'height': 109}, {'x': 160, 'y': 249, 'width': 48, 'height': 257}, {'x': 335, 'y': 429, 'width': 350, 'height': 49}, {'x': 354, 'y': 354, 'width': 55, 'height': 312}, {'x': 474, 'y': 15, 'width': 105, 'height': 373}
    ]
    GPU = False
    BOARD_SIZE = (700, 700)
    data_path = "log/results.hdf5" 

    save_flag = False if GPU else True
    load_flag = True if GPU else False    
    
    modal = Genetic_Algorithm(
        learning_rate=0 if GPU else 0.1, 
        mutation_rate=0 if GPU else 0.1,
        select_per_epoch=1 if GPU else 20,
        generation_multiplier=1 if GPU else 50,
        save_flag= save_flag,  
        load_flag= load_flag,
        sample_speed=20,
        dataframe_path=data_path
    )

    # If GPU is available
    if GPU:
        app = QApplication(sys.argv)
        board = Game_Board(
            board_size=BOARD_SIZE, model=modal, 
            sample_speed=20, obstacles=[]#object_dist_to_Qt(default_objects)
        )
        board.show()
        sys.exit(app.exec_())
    
    # If GPU is not available
    else:
        board = Game_Board_Without_GPU(
            board_size=BOARD_SIZE, model=modal, 
            sample_speed=20, obstacles=default_objects,
        )

def object_dist_to_Qt(object_list):
    return_dist = []
    for obj in object_list:
        return_dist.append(QGraphicsRectItem(obj["x"], obj["y"], obj["width"], obj["height"]))
    return return_dist

if __name__ == "__main__":
    main()
