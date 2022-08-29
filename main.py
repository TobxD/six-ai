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
def simulate(board, player1, player2, game_viewer, startPlayer = 1):
    moveNum = 0
    players = [player1, player2]
    toMove = startPlayer-1
    positions = []
    while True:
        game_viewer.drawBoard(board)
        #print(board)
        winner = board.hasWon()
        if winner != 0:
            result = winner*2 - 3
            #print(winner, "has won after", moveNum, "moves")
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

def storeGames(games, path = "data.json"):
    with open(Path(path), "a") as f:
        for game in games:
            positions, result = game
            for position in positions:
                f.write(json.dumps(position) + "\n")
                f.write(json.dumps(result) + "\n")

def testRandom(randomColor = False):
    board = Board(SIZE, startPieces=True)
    if not randomColor or random.choice([True, False]):
        player1 = RandomBot(1, search_winning=True, search_losing=True)
        player2 = RandomBot(2, search_winning=True, search_losing=True)
    else:
        player1 = RandomBot(1, search_winning=True, search_losing=True)
        player2 = RandomBot(2, search_winning=True, search_losing=True)
    return simulate(board, player1, player2)

def collectGames(count: int):
    data = []
    for i in range(count):
        data.append(testRandom(randomColor=False))

    return data

def generateGames(cnt, randomColor):
    workers = mp.cpu_count() - 2
    print(f"Executing on {workers} of your {mp.cpu_count()} CPUs")
    print(f"Estimated time (4 games per process per second): {cnt/workers//4} s")

    part_count = [cnt//workers for i in range(workers)]

    start = timer()
    with mp.Pool(processes=workers) as pool:
        data = pool.map(collectGames,part_count)
        end = timer()
        print(f'elapsed time: {end - start} s')
        print(f'per Game: {(end - start)/cnt} s')

        with open(Path("data/policy_data.json"), "a") as f:
            gameCounter = {-1:0, 0:0, 1:0}
            for processData in data:
                for game in processData:
                    positions, result = game
                    gameCounter[result] += 1
                    for position in positions[-2:]:
                        f.write(position + "\n")
                        f.write(json.dumps(result) + "\n")
            print(gameCounter)

def playGames(cfg:DictConfig, game_viewer):
    player1 = MCTS_PVBot.getMCTSBot(cfg, color=1, network=None, randomMove=True)
    player2 = MCTS_PVBot.getMCTSBot(cfg, color=2, network=None, randomMove=False)
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

def doWork(cfg: DictConfig, game_viewer):
    print(os.getcwd())
    #PVbot_util.training(cfg)
    playGames(cfg, game_viewer)
    #generateTrainLoop(cfg, game_viewer)
    profiler.printStats()

@hydra.main(config_path="conf", config_name="PVconfig")
def main(cfg: DictConfig):
    gv = gameviewer.GameViewer()
    gv.start(lambda: doWork(cfg, gv))

if __name__ == "__main__":
    main()