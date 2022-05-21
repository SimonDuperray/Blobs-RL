import matplotlib.pyplot as plt, numpy as np
plt.style.use('ggplot')

fig, axs = plt.subplots(2,1)

line1, = axs.scatter([], [])
line2, = axs.plot([], [], color='r')

for i in range(10):
   # scatter
   xs = [np.random.randint(-1, 11) for _ in range(10)]
   ys = [np.random.randint(-1, 11) for _ in range(10)]
   axs[0,0].scatter(xs, ys, color='r')
   plt.draw()
   plt.pause(0.5)
   plt.clf()

   # plot
   xp = [np.random.randint(-1, 11) for _ in range(10)]
   yp = [np.random.randint(-1, 11) for _ in range(10)]
   plt.plot(xp, yp, color='green')
   plt.draw()
   plt.pause(0.5)
   plt.clf()
