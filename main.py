import logging
from tabulate import tabulate
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
from omegaconf import OmegaConf
from omegaconf import open_dict

from timing import profiler
import os

logger = logging.getLogger(__file__)

posCnt = {-1:0, 0:0, 1:0}
gameCnt = {-1:0, 0:0, 1:0}

# there have to be moves available -> at least one stone already set
def simulate(board, player1, player2, gv_queue=None, drawInd = None, startPlayer = 1):
    logger = logging.getLogger(__name__)
    moveNum = 0
    players = [player1, player2]
    toMove = startPlayer-1
    positions = []
    while True:
        if gv_queue:
            gv_queue.put((drawInd, board))
        with profiler.getProfiler("has won"):
            winner = board.hasWon()
        if winner != 0:
            result = winner*2 - 3
            print(f"{winner} has won after {moveNum} moves")
            break
        with profiler.getProfiler("moves avail"):
            movesAvailable = board.movesAvailable()
        if len(movesAvailable) == 0:
            result = 0
            break
        with profiler.getProfiler("get next move"):
            (move_y, move_x), Y_policy = players[toMove].nextMove(board)
        with profiler.getProfiler("store position"):
            positions.append((copy.deepcopy(board.board), toMove+1, list(Y_policy.items())))
        board.move(move_y, move_x)
        toMove = 1-toMove
        moveNum += 1
    posCnt[result] += moveNum
    gameCnt[result] += 1

    # profiler.printStats()

    # -1 means first player wins, 1 means second player wins
    return (positions, result)

def getPlayer(player_cfg: DictConfig, cfg: DictConfig, color: int):
    if player_cfg.player_type == "mcts_nn":
        return MCTS_PVBot.getMCTSBot(player_cfg, cfg=cfg, color=color, network=None, randomUpToMove=player_cfg.randomUpToMove)
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

def collectGames(cfg, result_queue, game_in_queue, game_num_queue, gv_queue, drawInd):
    data = []
    try:
        while True:
            game_in_queue.get(timeout=1)
            data.append(playGame(cfg, cfg.play.random_color, gv_queue, drawInd))
            game_num = game_num_queue.get()
            print(f"done with game {game_num}")
    except queue.Empty:
        pass
    for d in data:
        result_queue.put(d)

    profiler.printStats()

def storeGames(games, path):
    if path==None:
        return
    with open(Path(util.toPath(path)), "a") as f:
        logger.info(f"starting writing {len(games)} games to {path}")
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
    logger.info(f"Executing {cfg.play.num_games} games on {workers} of your {mp.cpu_count()} CPUs")

    start = timer()
    """
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
    """
    gameQueue = mp.Queue()
    gameNumQueue = mp.Queue()
    resultQueue = mp.Queue()
    for i in range(cfg.play.num_games):
        gameQueue.put(i)
        gameNumQueue.put(i+1)

    worker_processes = [mp.Process(target=collectGames, args=(cfg, resultQueue, gameQueue, gameNumQueue, gv_queue, i)) for i in range(workers)]
    for w in worker_processes:
        w.start()
    games = []
    for i in range(cfg.play.num_games):
        games.append(resultQueue.get())
    for w in worker_processes:
        w.join()
    end = timer()

    profiler.printStats()

    (gameCounter, posCounter) = gameStats(games)
    logger.info(f'elapsed time: {end - start} s\n'
      f'per Game: {(end - start) / cfg.play.num_games} s\n'
      f'results: {gameCounter}\n'
      f'number of positions: {posCounter}\n'
      f'time per position: {(end - start) / posCounter}')
    storeGames(games, cfg.play.store_path)
    return gameCounter

# returns fraction won by model 1
def evalModel(cfg, model_path1, model_path2, cnt_per_color, gv_queue, player1_config=None, player2_config=None):
    newCfg = copy.deepcopy(cfg)
    newCfg.play.num_games = cnt_per_color
    newCfg.play.store_path = None

    if player1_config:
        newCfg.player1 = player1_config
    if player2_config:
        newCfg.player2 = player2_config

    newCfg.player1.dirichletNoise = False
    newCfg.player2.dirichletNoise = False

    newCfg.player1.model_path = model_path1
    newCfg.player2.model_path = model_path2
    stats1 = generateGames(newCfg, gv_queue)

    newCfg.player1.model_path = model_path2
    newCfg.player2.model_path = model_path1
    stats2 = generateGames(newCfg, gv_queue)

    win_cnt = stats1[-1] + stats2[1] + 0.5 * (stats1[0] + stats2[0])
    return win_cnt/(2 * cnt_per_color)

def addFile(file_list, new_file, max_number_positions):
    cnt = 0
    file_list.append(new_file)
    ret_list = []
    for i in range(len(file_list)-1, -1, -1):
        ret_list = [file_list[i]] + ret_list
        with open(util.toPath(file_list[i])) as f:
            cnt += len(f.readlines())/2
        if cnt > max_number_positions:
            break
    return ret_list

def generateTrainLoop(cfg: DictConfig, gv_queue):
    game_data_files = []
    for path in cfg.iterate.game_data_files:
        game_data_files = addFile(game_data_files, path, cfg.iterate.num_past_train_positions)
    model_path = cfg.iterate.model_start_path
    for i in range(cfg.iterate.num_iterations):
        game_path = f"games-{i}.json"
        next_model_path = f"model-{i+1}.ckpt"
        cfg.player1.model_path = model_path
        cfg.player2.model_path = model_path
        cfg.play.store_path = game_path
        cfg.general_train.input_model_path = model_path
        cfg.general_train.output_model_path = next_model_path
        generateGames(cfg, gv_queue)
        game_data_files = addFile(game_data_files, game_path, cfg.iterate.num_past_train_positions)
        cfg.data.train_data_path = game_data_files
        logger.info("training on " + str(game_data_files))
        PVbot_util.training(cfg)
        winning_perc = evalModel(cfg, next_model_path, model_path, cfg.iterate.num_evaluation_games, gv_queue)
        logger.info(f"new generation {i} won with frequency {winning_perc}")
        if winning_perc >= cfg.iterate.winning_threshold:
            logger.info(f"continuing with new model {next_model_path}")
            model_path = next_model_path
        else:
            logger.info(f"keeping old model {model_path}")
            logger.info(f"deleting new model {next_model_path}")
            os.remove(util.toPath(next_model_path))

def eval_models(cfg: DictConfig, gv_queue):
    cfg = copy.deepcopy(cfg)
    num_models = len(cfg.eval.models)
    results = [["/" for _ in range(num_models)] for _ in range(num_models)]
    for i1 in range(num_models):
        # load config from path
        cfg.eval.models[i1].player = OmegaConf.load(util.toPath(cfg.eval.models[i1].player))
        for i2 in range(i1):
            c1 = cfg.eval.models[i1]
            c2 = cfg.eval.models[i2]
            res = evalModel(cfg, c1.path, c2.path, cfg.eval.num_evaluation_games, gv_queue, c1.player, c2.player)
            results[i1][i2] = res
            results[i2][i1] = 1-res
    with open(util.toPath("eval_results.json"), "w") as f:
        json.dump(results, f)
    
    headers = ["Model"] + [f"Model {i}" for i in range(num_models)]
    table_data = [[f"Model {i}"] + results[i] for i in range(num_models)]
    info_str = "\n".join([f"Model {i}: {cfg.eval.models[i].path}" for i in range(num_models)])
    table_str = info_str + "\n" + tabulate(table_data, headers, tablefmt="grid")

    with open(util.toPath("eval_results.txt"), "w") as f:
        f.write(table_str)
    print(table_str)

def doWork(cfg: DictConfig, game_viewer, gv_queue):
    logger.info(f"current wd: {os.getcwd()}")
    if cfg.general.mode == "play":
        generateGames(cfg, gv_queue)
    elif cfg.general.mode == "train":
        PVbot_util.training(cfg)
    elif cfg.general.mode == "iterate":
        generateTrainLoop(cfg, gv_queue)
    elif cfg.general.mode == "eval":
        eval_models(cfg, gv_queue)
    else:
        logger.error(f"invalid mode \"{cfg.general.mode}\"selected")
    profiler.printStats()

@hydra.main(config_path="conf", config_name="PVconfig", version_base="1.1")
def main(cfg: DictConfig):
    logger.info(cfg)
    if cfg.play.game_viewer:
        gv = gameviewer.GameViewer()
        gv.start(lambda gv_queue: doWork(cfg, gv, gv_queue))
    else:
        doWork(cfg, None, None)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()