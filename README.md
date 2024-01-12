# Six AI

This repository applies an adaptation of Alpha Zero to the game Six

## The Game Six

The game is adapted from (Six)[https://boardgamegeek.com/boardgame/20195/six].
We only consider the first phase of the game and on a limited board size. We play on a (e.g.) 10x10 hex grid, the two players place pieces in alternating turns, and the first player to complete one of three winning shapes wins the game.
For more information on the game, see the (rules)[https://cdn.shopify.com/s/files/1/0760/5141/5360/files/Six_All.pdf?v=1694100235]

## Algorithm

We mainly follow the AlphaZero approach. We pitch new versions of the network against the old version and use it only if its winrate is high enough (as done in AlphaGo). We also changed the network architecture a bit, particularly the policy head is fully convolutional and the the value head uses a pooling layer -- this makes the network work for different board sizes and seems to be a good inductive bias against overfitting in the policy head.

## How to get started

- Install the requirements
- Change the config file in `conf/PVconfig.yaml` to fit your needs: particularly, select whether you want to play, train, etc. and modify the respective config sections
- To just play against the current version, you can leave the config file as is and fun `python main.py`
