# Game of Pick Beans
## Env Settings
1. There are N beans and M players.
2. The M players pick the beans in order, they can know the number of the left beans before they pick.
3. After each player has picked his beans, we will compute the result:
   1. All the players with the most and least beans will die, receiving a death reward (e.g., -10).
      1. An additional settings: the dead players can get a conciliatory reward (e.g., 5).
   2. Other players will live and receives a live reward (e.g., 10).

## Result
1. dead reward = -10, alive reward = 10, other dead reward = 5, M = 5, N = 50.
   1. Final selecting beans: 8,8,7,8,8 or 50,0,0,0,0 with every player dies.
   2. The player finds that it is hard for them to live and so they begin to optimize to make the others to die to get the other dead reward.
   3. The following image shows the dead and live reward every player receives (i.e., other dead reward excluded in this image )
   ![reward from dead/alive](./img/exp1/result1_pick_reward.png)
