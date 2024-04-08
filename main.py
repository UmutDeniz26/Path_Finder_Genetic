import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QGraphicsRectItem
from scritps.Game_Board import Game_Board
from scritps.Game_Board_Without_GPU import Game_Board as Game_Board_Without_GPU
from scritps.Genetic_Algorithm import Genetic_Algorithm


def main():
    default_objects=[]    
    #default_objects = [{'x': 238, 'y': 66, 'width': 87, 'height': 306}, {'x': 459, 'y': 312, 'width': 81, 'height': 383}, {'x': 167, 'y': 469, 'width': 539, 'height': 123}, {'x': 170, 'y': 7, 'width': 186, 'height': 88}, {'x': 143, 'y': 222, 'width': 144, 'height': 50}]
    default_objects = [{"x":300, "y":300, "width":100, "height":100}]

    GPU = False
    BOARD_SIZE = (700, 700)
    data_path = "log/results.hdf5" 

    save_flag = False if GPU else True
    load_flag = True if GPU else False    
    
    modal = Genetic_Algorithm(
        learning_rate=0 if GPU else 0.1, 
        mutation_rate=0 if GPU else 0.1,
        select_per_epoch=1 if GPU else 50,
        generation_multiplier=1 if GPU else 10,
        save_flag= True,#save_flag,  
        load_flag= False,#load_flag,
        sample_speed=20,
        dataframe_path=data_path,
        exit_reached_flag=False
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

def object_dist_to_Qt(object_list):
    return_dist = []
    for obj in object_list:
        return_dist.append(QGraphicsRectItem(obj["x"], obj["y"], obj["width"], obj["height"]))
    return return_dist

if __name__ == "__main__":
    main()
