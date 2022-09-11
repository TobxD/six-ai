from pprint import pprint
import queue
import shutil
from board import Board, SIZE
import util
from randomBot import RandomBot
import PVbot.PVbot_util as PVbot_util
import PVbot.MCTS_PVBot as MCTS_PVBot
import visualizer.gameviewer as gameviewer

import copy
import json, random
from pathlib import Path
import multiprocessing as mp
from timeit import default_timer as timer

import hydra
from omegaconf.dictconfig import DictConfig
from omegaconf import open_dict

from timing import profiler
import os

posCnt = {-1:0, 0:0, 1:0}
gameCnt = {-1:0, 0:0, 1:0}

# there have to be moves available -> at least one stone already set
def simulate(board, player1, player2, gv_queue=None, drawInd = None, startPlayer = 1):
    moveNum = 0
    players = [player1, player2]
    toMove = startPlayer-1
    positions = []
    while True:
        if gv_queue:
            gv_queue.put((drawInd, board))
        winner = board.hasWon()
        if winner != 0:
            result = winner*2 - 3
            print(winner, "has won after", moveNum, "moves")
            break
        if len(board.movesAvailable()) == 0:
            result = 0
            break
        (move_y, move_x), Y_policy = players[toMove].nextMove(board)
        positions.append((copy.deepcopy(board.board), toMove+1, list(Y_policy.items())))
        board.move(move_y, move_x)
        toMove = 1-toMove
        moveNum += 1
    posCnt[result] += moveNum
    gameCnt[result] += 1

    # -1 means first player wins, 1 means second player wins
    return (positions, result)

def getPlayer(player_cfg: DictConfig, cfg: DictConfig, color: int):
    if player_cfg.player_type == "mcts_nn":
        return MCTS_PVBot.getMCTSBot(player_cfg, cfg=cfg, color=color, network=None, randomMove=player_cfg.randomMove)
    elif player_cfg.player_type == "random":
        return RandomBot(color, player_cfg.search_winning, player_cfg.search_losing)
    else:
        exit(1)

def playGame(cfg, randomColor, gv_queue, drawInd):
    board = Board(SIZE, startPieces=True)
    if not randomColor or random.choice([True, False]):
        player1 = getPlayer(cfg.player1, cfg, 1)
        player2 = getPlayer(cfg.player2, cfg, 2)
    else:
        player1 = getPlayer(cfg.player2, cfg, 2)
        player2 = getPlayer(cfg.player1, cfg, 1)
    return simulate(board, player1, player2, gv_queue, drawInd)

def collectGames(cfg, game_in_queue, game_num_queue, gv_queue, drawInd):
    data = []
    try:
        while True:
            game_in_queue.get(timeout=1)
            data.append(playGame(cfg, cfg.play.random_color, gv_queue, drawInd))
            game_num = game_num_queue.get()
            print(f"done with game {game_num}")
    except queue.Empty:
        pass
    return data

def storeGames(games, path):
    if path==None:
        return
    with open(Path(util.toPath(path)), "a") as f:
        print(f"starting writing {len(games)} games to {path}")
        for game in games:
            positions, result = game
            for position in positions:
                f.write(json.dumps(position) + "\n")
                f.write(json.dumps(result) + "\n")

def gameStats(games):
    gameCounter = {-1:0, 0:0, 1:0}
    posCounter = 0
    for game in games:
        (positions, result) = game
        gameCounter[result] += 1
        posCounter += len(positions)
    return (gameCounter, posCounter)

def generateGames(cfg, gv_queue):
    workers = min(cfg.play.workers, cfg.play.num_games)
    print(f"Executing {cfg.play.num_games} games on {workers} of your {mp.cpu_count()} CPUs")

    start = timer()
    with mp.Pool(processes=workers) as pool:
        gameQueue = mp.Manager().Queue()
        gameNumQueue = mp.Manager().Queue()
        for i in range(cfg.play.num_games):
            gameQueue.put(i)
            gameNumQueue.put(i+1)

        args = [(cfg, gameQueue, gameNumQueue, gv_queue, i) for i in range(workers)]
        data = pool.starmap(collectGames, args)
        end = timer()
        print(f'elapsed time: {end - start} s')
        print(f'per Game: {(end - start)/cfg.play.num_games} s')

        games = [game for gameList in data for game in gameList] #flattens the data list
        (gameCounter, posCounter) = gameStats(games)
        print(f"results: {gameCounter}")
        print(f"number of positions: {posCounter}")
        storeGames(games, cfg.play.store_path)
        return gameCounter

# returns fraction won by model 1
def evalModel(cfg, model_path1, model_path2, cnt_per_color, gv_queue):
    newCfg = copy.deepcopy(cfg)
    newCfg.play.num_games = cnt_per_color
    newCfg.play.store_path = None

    newCfg.player1.model_path = model_path1
    newCfg.player2.model_path = model_path2
    stats1 = generateGames(newCfg, gv_queue)

    newCfg.player1.model_path = model_path2
    newCfg.player2.model_path = model_path1
    stats2 = generateGames(newCfg, gv_queue)

    win_cnt = stats1[-1] + stats2[1] + 0.5 * (stats1[0] + stats2[0])
    return win_cnt/(2 * cnt_per_color)

def generateTrainLoop(cfg: DictConfig, gv_queue):
    model_path = cfg.iterate.model_start_path
    for i in range(cfg.iterate.num_iterations):
        game_path = f"games-{i}.json"
        next_model_path = f"model-{i+1}.ckpt"
        cfg.player1.model_path = model_path
        cfg.player2.model_path = model_path
        cfg.play.store_path = game_path
        cfg.data.train_data_path = game_path
        cfg.general_train.input_model_path = model_path
        cfg.general_train.output_model_path = next_model_path
        generateGames(cfg, gv_queue)
        PVbot_util.training(cfg)
        winning_perc = evalModel(cfg, next_model_path, model_path, cfg.iterate.num_evaluation_games, gv_queue)
        print(f"new generation {i} won with frequency {winning_perc}")
        if winning_perc >= cfg.iterate.winning_threshold:
            print(f"continuing with new model {next_model_path}")
            model_path = next_model_path
        else:
            print(f"keeping old model {model_path}")
            # reuse games from this iteration as model does not change
            # this results in more train data and hopefully a better model in the next iteration
            shutil.copyfile(util.toPath(game_path), util.toPath(f"games-{i+1}.json"))

def doWork(cfg: DictConfig, game_viewer, gv_queue):
    #print(evalModel(cfg, "/models/small_test.ckpt", "/models/small_test2.ckpt", 50, gv_queue))
    #return
    print(os.getcwd())
    if cfg.general.mode == "play":
        generateGames(cfg, gv_queue)
    elif cfg.general.mode == "train":
        PVbot_util.training(cfg)
    elif cfg.general.mode == "iterate":
        generateTrainLoop(cfg, gv_queue)
    else:
        print(f"invalid mode \"{cfg.general.mode}\"selected")
    profiler.printStats()

@hydra.main(config_path="conf", config_name="PVconfig", version_base="1.1")
def main(cfg: DictConfig):
    print(util.toPath(cfg.play.store_path))
    if cfg.play.game_viewer:
        gv = gameviewer.GameViewer()
        gv.start(lambda gv_queue: doWork(cfg, gv, gv_queue))
    else:
        doWork(cfg, None, None)

if __name__ == "__main__":
    main()