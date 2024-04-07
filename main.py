import sys
from PyQt5.QtWidgets import QApplication
from scritps.Game_Board import Game_Board

def main():
    app = QApplication(sys.argv)
    game_board = Game_Board( board_size= (700, 700) )
    game_board.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
