from tkinter import *
import math
import sys
import threading
import copy

import multiprocessing

sys.path.append(".")


class GameCanvas(Canvas):
    x_diff = math.sin(math.pi / 3)
    y_diff = 1 + math.cos(math.pi / 3)

    def __init__(self, master, *args, **kwargs):
        Canvas.__init__(self, master=master, *args, **kwargs)
        y_diff_poly = math.cos(math.pi / 3)
        self.poly = [
            0,
            0,
            0,
            1,
            self.x_diff,
            1 + y_diff_poly,
            2 * self.x_diff,
            1,
            2 * self.x_diff,
            0,
            self.x_diff,
            -y_diff_poly,
        ]
        self.is_x = [
            True,
            False,
            True,
            False,
            True,
            False,
            True,
            False,
            True,
            False,
            True,
            False,
        ]
        self.listener = []
        self.mid_points = []

    def drawHex(self, x, y, side_length, color):
        new_poly = [
            (x if is_x else y) + side_length * coord
            for (coord, is_x) in zip(self.poly, self.is_x)
        ]
        self.create_polygon(new_poly, outline="black", fill=color, width=3)

    def drawBoard(self, board):
        colors = ["white", "red", "black"]
        # side_length = 25
        side_length = 12
        off = 20
        self.mid_points = []
        for i in range(board.size):
            for j in range(board.size):
                self.drawHex(
                    off + side_length * self.x_diff * (2 * j + i),
                    off + side_length * self.y_diff * i,
                    side_length,
                    colors[board[i, j]],
                )
                self.mid_points.append(
                    (
                        off + side_length * self.x_diff * (2 * j + i + 1),
                        off + side_length * (self.y_diff * i + 0.5),
                    )
                )
        self.pack()

    def add_click_listener(self, listener_queue):
        self.bind("<Button-1>", self.on_click)
        self.listener.append(listener_queue)

    def on_click(self, event):
        clicked_field = self.calculate_clicked_field(event.x, event.y)
        if clicked_field is not None:
            for l in self.listener:
                l.put(clicked_field)

    def calculate_clicked_field(self, x, y):
        bst, bstD = None, float("inf")
        board_size = int(math.sqrt(len(self.mid_points)))
        for i, (x_mid, y_mid) in enumerate(self.mid_points):
            dist = (x_mid - x) ** 2 + (y_mid - y) ** 2
            if dist < bstD:
                bst, bstD = (i // board_size, i % board_size), dist
        return bst


class GameViewer:
    def __init__(self):
        self.q = multiprocessing.Manager().Queue()
        self.root = Tk()
        self.windows = {}
        # self.windows[0] = GameCanvas(self.root, height=450, width=650)
        self.windows[0] = GameCanvas(self.root, height=2000, width=3000)

    def start(self, target):
        threading.Thread(target=target, args=(self.q,)).start()
        self.root.after(100, lambda: self.redraw_if_needed())
        self.root.mainloop()

    def redraw_if_needed(self):
        while not self.q.empty():
            action_type, (window, content) = self.q.get()
            if window not in self.windows:
                tl = Toplevel(self.root)
                self.windows[window] = GameCanvas(tl, height=450, width=650)
                self.windows[window].pack()
            if action_type == "listener":
                self.register_click_listener(window, content)
                continue
            else:
                assert action_type == "move"
                self.windows[window].drawBoard(content)
        self.root.after(100, lambda: self.redraw_if_needed())

    def drawBoard(self, window: int, board):
        print("draw board called:")
        print(board)
        self.q.put((window, copy.deepcopy(board)))

    def register_click_listener(self, window, listener):
        self.windows[window].add_click_listener(listener)

    def quit(self):
        return self.root.destroy()


"""
# testing code

listener_queue = multiprocessing.Manager().Queue()
b = board.Board(10, True)
gv = GameViewer()
# gv.register_click_listener(0, listener_queue)

def testF(gv_queue):
    gv_queue.put(("listener", (0, listener_queue)))
    for i in range(10):
        time.sleep(1)
        print(i)
        b.move(i, i)
        gv_queue.put(("move", (0, b)))
        # gv.drawBoard(b)

# gv.drawBoard(b)
gv.start(testF)

while True:
    print(listener_queue.get())
"""
