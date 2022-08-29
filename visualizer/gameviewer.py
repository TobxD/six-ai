from tkinter import *
import time
import math
import sys
import threading
import queue

sys.path.append('.')
import board

class GameCanvas(Canvas):
    x_diff = math.sin(math.pi/3)
    y_diff = 1 + math.cos(math.pi/3)

    def __init__(self, master,*args,**kwargs):
        Canvas.__init__(self, master=master, *args, **kwargs)
        y_diff_poly = math.cos(math.pi/3)
        self.poly = [0, 0, 0, 1, self.x_diff, 1+y_diff_poly, 2*self.x_diff, 1, 2*self.x_diff, 0, self.x_diff, -y_diff_poly]
        self.is_x = [True, False, True, False, True, False, True, False, True, False, True, False]

    def drawHex(self, x, y, side_length, color):
        new_poly = [(x if is_x else y) + side_length*coord for (coord, is_x) in zip(self.poly, self.is_x)]
        self.create_polygon(new_poly, outline='black', fill=color, width=3)

    def drawBoard(self, board):
        colors = ["white", "red", "black"]
        side_length = 25
        off = 20
        for i in range(board.size):
            for j in range(board.size):
                self.drawHex(off + side_length*self.x_diff*(2*j+i), off + side_length*self.y_diff*i, side_length, colors[board[i, j]])
        self.pack()

class GameViewer():
    q = queue.Queue()

    def __init__(self):
        self.root = Tk()
        self.canvas = GameCanvas(self.root, height=450, width=650)

    def start(self, target):
        threading.Thread(target=target).start()
        self.root.after(100, lambda: self.redraw_if_needed())
        self.root.mainloop()

    def redraw_if_needed(self):
        while self.q.qsize() > 1:
            self.q.get()
        if not self.q.empty():
            self.canvas.drawBoard(self.q.get())
        self.root.after(100, lambda: self.redraw_if_needed())

    def drawBoard(self, board):
        self.q.put(board)

"""
b = board.Board(10, True)
gv = GameViewer()

def testF(x):
    for i in range(10):
        time.sleep(1)
        print(i)
        b.move(i, i)
        gv.drawBoard(b)

gv.drawBoard(b)
gv.start(testF)
"""