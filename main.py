from pprint import pprint
from board import Board, SIZE
from randomBot import RandomBot
import PVbot.PVbot_util as PVbot_util
import PVbot.MCTS_PVBot as MCTS_PVBot
import visualizer.gameviewer as gameviewer

import json, random
from pathlib import Path
import multiprocessing as mp
from timeit import default_timer as timer

import hydra
from omegaconf.dictconfig import DictConfig
from omegaconf import open_dict
from hydra.utils import to_absolute_path

from timing import profiler
import os

posCnt = {-1:0, 0:0, 1:0}
gameCnt = {-1:0, 0:0, 1:0}

# there have to be moves available -> at least one stone already set
def simulate(board, player1, player2, gvQ=None, drawInd = None, startPlayer = 1):
    moveNum = 0
    players = [player1, player2]
    toMove = startPlayer-1
    positions = []
    while True:
        if gvQ:
            gvQ.put((drawInd, board))
        winner = board.hasWon()
        if winner != 0:
            result = winner*2 - 3
            print(winner, "has won after", moveNum, "moves")
            break
        if len(board.movesAvailable()) == 0:
            result = 0
            #print("no more moves possible -> draw")
            break
        #with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as prof:
        #    with record_function("model_inference"):
        move, Y_policy = players[toMove].nextMove(board)
        #print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=15))
        positions.append((board.board, toMove+1, list(Y_policy.items())))
        #print(move)
        move_y, move_x = move
        board.move(move_y, move_x)
        toMove = 1-toMove
        moveNum += 1
    posCnt[result] += moveNum
    gameCnt[result] += 1

    return (positions, result)

def storeGames(games, path=None):
    if path==None:
        return
    with open(Path(path), "a") as f:
        for game in games:
            positions, result = game
            for position in positions:
                f.write(json.dumps(position) + "\n")
                f.write(json.dumps(result) + "\n")

def gameStats(games):
    gameCounter = {-1:0, 0:0, 1:0}
    posCounter = 0
    for game in games:
        pprint(game)
        (positions, result) = game
        gameCounter[result] += 1
        posCounter += len(positions)
    return (gameCounter, posCounter)

def getPlayer(player_cfg: DictConfig, cfg: DictConfig, color: int):
    if player_cfg.player_type == "mcts_nn":
        return MCTS_PVBot.getMCTSBot(player_cfg, cfg=cfg, color=color, network=None, randomMove=player_cfg.randomMove)
    else:
        exit(1)

def playGame(cfg, randomColor, q, drawInd):
    board = Board(SIZE, startPieces=True)
    if not randomColor or random.choice([True, False]):
        player1 = getPlayer(cfg.player1, cfg, 1)
        player2 = getPlayer(cfg.player2, cfg, 2)
    else:
        player1 = getPlayer(cfg.player2, cfg, 2)
        player2 = getPlayer(cfg.player1, cfg, 1)
    return simulate(board, player1, player2, q, drawInd)

def collectGames(cfg, count: int, q, drawInd):
    data = []
    for i in range(count):
        data.append(playGame(cfg, cfg.play.random_color, q, drawInd))

    return data

def generateGames(cfg, gv_queue):
    workers = min(cfg.play.workers, cfg.play.num_games)
    cfg.play.num_games -= cfg.play.num_games % workers
    print(f"Executing {cfg.play.num_games} games on {workers} of your {mp.cpu_count()} CPUs")

    start = timer()
    with mp.Pool(processes=workers) as pool:
        args = [(cfg, cfg.play.num_games//workers, gv_queue, i) for i in range(workers)]
        data = pool.starmap(collectGames, args)
        end = timer()
        print(f'elapsed time: {end - start} s')
        print(f'per Game: {(end - start)/cfg.play.num_games} s')

        games = []
        map(games.extend, data) #flattens the data list
        (gameCounter, posCounter) = gameStats(games)
        print(f"results: {gameCounter}")
        print(f"number of positions: {posCounter}")
        storeGames(games, cfg.play.store_path)

def playGames(cfg:DictConfig, game_viewer):
    player1 = getPlayer(cfg.player1, cfg, 1)
    player2 = getPlayer(cfg.player2, cfg, 2)
    #player1 = RandomBot(1, search_winning=True, search_losing=True)

    gameCounter = {-1:0, 0:0, 1:0}
    moves = 0
    with profiler.getProfiler("play games"):
        numGames=cfg.bot_test.num_games
        games = []
        for i in range(numGames):
            print(f"Game {i+1}:")
            board = Board(SIZE, startPieces=True)
            game = simulate(board, player1, player2, game_viewer)
            gameCounter[game[1]] += 1
            moves += len(game[0])
            print(f"{gameCounter} in {len(game[0])} moves")

            games.append(game)
            if (cfg.bot_test.break_on_loose and game[1] == -1):
                break

            if cfg.bot_test.store_games:
                storeGames(games, cfg.bot_test.store_path)
                games = []
        print(i)
        gamesPlayed = i+1
    timeUsed = profiler.getTime("play games")
    print(f'Total moves: {moves}, moves per game {moves/gamesPlayed}')
    print(f'elapsed time: {timeUsed} s')
    print(f'per Game: {timeUsed/gamesPlayed} s')
    print(f'per Move: {timeUsed/moves} s')
    print(gameCounter)

def generateTrainLoop(cfg: DictConfig, game_viewer):
    cwd = os.getcwd()
    for i in range(2):
        if i == 0:
            model_path = to_absolute_path(f"models/model-{i}.ckpt")
        else:
            model_path = f"{cwd}/model-{i}.ckpt"
        game_path = f"{cwd}/games-{i}.json"
        next_model_path = f"{cwd}/model-{i+1}.ckpt"
        with open_dict(cfg):
            cfg.mcts_bot.model_path = model_path
            cfg.bot_test.store_path = game_path
            cfg.data.train_data_path = game_path
        playGames(cfg, game_viewer)
        PVbot_util.training(cfg, model_path, next_model_path)

def doWork(cfg: DictConfig, game_viewer, gvQueue):
    print(os.getcwd())
    generateGames(cfg, gvQueue)
    #PVbot_util.training(cfg)
    #playGames(cfg, game_viewer)
    #generateTrainLoop(cfg, game_viewer)
    profiler.printStats()

@hydra.main(config_path="conf", config_name="PVconfig", version_base="1.1")
def main(cfg: DictConfig):
    pprint(cfg.network_conf.board_size)
    gv = gameviewer.GameViewer()
    gv.start(lambda gvQueue: doWork(cfg, gv, gvQueue))

if __name__ == "__main__":
    main()