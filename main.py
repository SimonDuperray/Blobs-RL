from Blob import Blob
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt, time, numpy as np, pickle
plt.style.use('ggplot')

SIZE = 10
HM_EPISODES = 25000
MOVE_PENALTY = 1
ENEMY_PENALTY = 300
FOOD_REWARD = 25
EPS_DECAY = 0.9998
SHOW_EVERY = 3000
LEARNING_RATE = 0.1
DISCOUNT = 0.95
PLAYER_N = 1
FOOD_N = 2
ENEMY_N = 3

d = {
     1: (255, 175, 0),
     2: (0, 255, 0),
     3: (0, 0, 255)
}
epsilon = 0.9
start_q_table = None # or filename

if start_q_table is None:
   q_table = {}
   for x1 in range(-SIZE+1, SIZE):
      for y1 in range(-SIZE+1, SIZE):
         for x2 in range(-SIZE+1, SIZE):
            for y2 in range(-SIZE+1, SIZE):
               q_table[((x1, y1), (x2, y2))] = [np.random.uniform(-5, 0) for i in range(4)]
else:
  with open(start_q_table, "rb") as f:
    q_table = pickle.load(f)




for i in range(10):
   player = Blob(size=SIZE)
   enemy = Blob(size=SIZE)
   food = Blob(size=SIZE)
   plt.title(f'Blobs Environment #{i+1}')
   plt.scatter(player.x, player.y, s=100, marker='>', color="blue", label="player")
   plt.scatter(enemy.x, enemy.y, s=100, marker='s', color="red", label="enemy")
   plt.scatter(food.x, food.y, s=100, marker='o', color="green", label="food")
   plt.legend(loc="upper left")
   plt.plot([0, 0], [10, 0], color="black", linewidth=2)
   plt.plot([10, 0], [10, 10], color="black", linewidth=2)
   plt.plot([10, 10], [10, 0], color="black", linewidth=2)
   plt.plot([10, 0], [0, 0], color="black", linewidth=2)
   plt.xlim(-1, SIZE+1)
   plt.ylim(-1, SIZE+1)
   plt.draw()
   plt.pause(1)
   plt.clf()