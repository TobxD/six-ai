import multiprocessing
import queue

from board import Board

class HumanPlayer:
    def __init__(self, gv_queue):
        self.q = multiprocessing.Manager().Queue()
        gv_queue.put(("listener", (0, self.q)))

    def nextMove(self, board: Board):
        # clear queue
        while True:
            try:
                self.q.get(timeout=0)
            except queue.Empty:
                break

        all_moves = board.movesAvailable()
        next_move = self.q.get()
        policy = {move:(1 if move==next_move else 0) for move in all_moves}
        return next_move, policy
