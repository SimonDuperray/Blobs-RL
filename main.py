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


episode_rewards = []
for episode in range(HM_EPISODES):
   
   player = Blob(size=SIZE)
   enemy = Blob(size=SIZE)
   food = Blob(size=SIZE)

   if episode % SHOW_EVERY == 0:
      print(f"on #{episode}, epsilon is {epsilon}")
      print(f"{SHOW_EVERY} episodes shown")
      show = True
   else:
      show = False

   episode_reward = 0
   for i in range(200):
      obs = (player-food, player-enemy)
      if np.random.random() > epsilon:
         action = np.argmax(q_table[obs])
      else:
         action = np.random.randint(0, 4)

      player.action(action)

      if player.x==enemy.x and player.y==enemy.y:
         reward = -ENEMY_PENALTY
      elif player.x==food.x and player.y==food.y:
         reward = FOOD_REWARD
      else:
         reward = -MOVE_PENALTY

      new_obs = (player-food, player-enemy)
      max_future_q = np.max(q_table[new_obs])
      current_q = q_table[obs][action]

      if reward==FOOD_REWARD:
         new_q = FOOD_REWARD
      elif reward==-ENEMY_PENALTY:
         new_q = -ENEMY_PENALTY
      else:
         new_q = (1-LEARNING_RATE)*current_q+LEARNING_RATE*(reward+DISCOUNT*max_future_q)
      
      q_table[obs][action] = new_q

      if show:
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
         plt.pause(0.05)
         plt.clf()

      episode_reward += reward
      if reward==FOOD_REWARD or reward==-ENEMY_PENALTY:
         break
   episode_rewards.append(episode_reward)
   epsilon*=EPS_DECAY

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode="valid")
plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"reward {SHOW_EVERY} ma")
plt.xlabel("episode #")
plt.show()
