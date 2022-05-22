from Blob import Blob
from BlobTypes import BlobTypes
import matplotlib.pyplot as plt, time, numpy as np, pickle, math

plt.style.use('ggplot')

SIZE = 10
HM_EPISODES = 100000
MOOVE_PENALTY = 1
ENEMY_PENALTY = 300
ENEMY_RANGE_PENALTY = 150
FOOD_REWARD = 25
EPS_DECAY = 0.9998
SHOW_EVERY = 2000
LEARNING_RATE = 0.1
DISCOUNT = 0.95
PLAYER_N = 1
FOOD_N = 2
ENEMY_N = 3
ENEMY_RANGE = 1.5
ALLOWED_TIME_IN_RANGE = 10

epsilon = 0.9
start_q_table = None # or filename

def is_in_circle(ex, ey, px, py):
   return math.sqrt((px-ex)**2+(py-ey)**2)<=ENEMY_RANGE

def run_blobs():
   epsilon = 0.9
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
      
      player = Blob(size=SIZE, blob_type=BlobTypes.PLAYER)
      enemy = Blob(size=SIZE, blob_type=BlobTypes.ENEMY)
      # enemies = [Blob(size=SIZE, type=BlobTypes.ENEMY, idx=0), Blob(size=SIZE, type=BlobTypes.ENEMY, idx=1)]
      food = Blob(size=SIZE, blob_type=BlobTypes.FOOD)

      if episode % SHOW_EVERY == 0:
         print(f"on #{episode}, epsilon is {epsilon}")
         print(f"{SHOW_EVERY} episodes shown")
         show = True
      else:
         show = False

      episode_reward = 0
      px_path, py_path = [], []
      ex_path, ey_path = [], []
      # enemies_position = [[] for _ in range(len(enemies))]
      rewards_plot = []
      time_in_range = 0
      times_in_range_plot = []

      for i in range(200):
         # for enemy in enemies:
         obs = (player-food, player-enemy)
         if np.random.random() > epsilon:
            action = np.argmax(q_table[obs])
         else:
            action = np.random.randint(0, 4)

         player.action(action)
         px_path.append(player.x)
         py_path.append(player.y)

         # enemy.moove()
         # ex_path.append(enemy.x)
         # ey_path.append(enemy.y)
         # enemies_position[enemy.idx].append((enemy.x, enemy.y))

         if player.x==enemy.x and player.y==enemy.y:
            reward = -ENEMY_PENALTY
         elif is_in_circle(enemy.x, enemy.y, player.x, player.y):
            time_in_range+=1
            times_in_range_plot.append(time_in_range)
            reward = -ENEMY_RANGE_PENALTY
         elif player.x==food.x and player.y==food.y:
            reward = FOOD_REWARD
         else:
            reward = -MOOVE_PENALTY

         rewards_plot.append(reward)

         new_obs = (player-food, player-enemy)
         max_future_q = np.max(q_table[new_obs])
         current_q = q_table[obs][action]

         if reward==FOOD_REWARD:
            new_q = FOOD_REWARD
         elif reward==-ENEMY_PENALTY:
            new_q = -ENEMY_PENALTY
         elif reward==-ENEMY_RANGE_PENALTY:
            new_q = -ENEMY_RANGE_PENALTY
         else:
            new_q = (1-LEARNING_RATE)*current_q+LEARNING_RATE*(reward+DISCOUNT*max_future_q)
         
         q_table[obs][action] = new_q

      
         if show:
            plt.suptitle(f'Blobs Environment - Iteration #{i+1}')
            plt.scatter(player.x, player.y, s=100, marker='>', color="blue", label="player")
            plt.scatter(enemy.x, enemy.y, s=100, marker='s', color="red", label="enemy")
            enemy_border = plt.Circle((enemy.x, enemy.y), ENEMY_RANGE, color="red", fill=False)
            plt.gcf().gca().add_artist(enemy_border)
            # for enemy in enemies:
               # plt.scatter(enemy.x, enemy.y, s=100, marker='s', color="red", label="enemy")
               # enemy_border = plt.Circle((enemy.x, enemy.y), ENEMY_RANGE, color="red", fill=False)
               # plt.gcf().gca().add_artist(enemy_border)
            plt.scatter(food.x, food.y, s=100, marker='o', color="green", label="food")
            # plt.gcf().gca().set_aspect(1)
            plt.plot(px_path, py_path, color="blue", alpha=0.4)
            # plt.plot(ex_path, ey_path, color="red", linestyle="dashed", alpha=0.4)
            # for enemy in enemies_position:
            #    plt.plot(enemy, enemy., color="red", linestyle="dashed", alpha=0.4)
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
         if reward==FOOD_REWARD or reward==-ENEMY_PENALTY or reward==-ENEMY_RANGE_PENALTY*ALLOWED_TIME_IN_RANGE:
            break

      episode_rewards.append(episode_reward)
      epsilon*=EPS_DECAY



def test():
   for i in range(10):
      player = Blob(size=SIZE, type=BlobTypes.PLAYER)
      enemy = Blob(size=SIZE, type=BlobTypes.ENEMY)
      food = Blob(size=SIZE, type=BlobTypes.FOOD)
      plt.title(f'Blobs Environment #{i+1}')
      plt.scatter(player.x, player.y, s=100, marker='>', color="blue", label="player")
      plt.scatter(enemy.x, enemy.y, s=100, marker='s', color="red", label="enemy")
      enemy_border = plt.Circle((enemy.x, enemy.y), ENEMY_RANGE, color="red", fill=False)
      plt.gcf().gca().add_artist(enemy_border)
      print(f"{player} in {enemy} range: {is_in_circle(enemy.x, enemy.y, player.x, player.y)}")
      plt.scatter(food.x, food.y, s=100, marker='o', color="green", label="food")
      plt.legend(loc="upper left")
      plt.plot([0, 0], [10, 0], color="black", linewidth=2)
      plt.plot([10, 0], [10, 10], color="black", linewidth=2)
      plt.plot([10, 10], [10, 0], color="black", linewidth=2)
      plt.plot([10, 0], [0, 0], color="black", linewidth=2)
      plt.xlim(-1, SIZE+1)
      plt.ylim(-1, SIZE+1)
      plt.draw()
      plt.pause(0.5)
      plt.clf()


run_blobs()