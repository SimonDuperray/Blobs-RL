import numpy as np

class Blob:
   def __init__(self, size, type, idx=None):
      self.size = size
      self.x = np.random.randint(0, self.size)
      self.y = np.random.randint(0, self.size)
      self.type = type
      self.idx = idx

   def __str__(self):
      return f"[{self.type}]: ({self.x}, {self.y})"

   def __sub__(self, blob):
      return (self.x-blob.x, self.y-blob.y)

   def action(self, choice):
      if choice==0:
         self.moove(x=1, y=1)
      elif choice==1:
         self.moove(x=-1, y=-1)
      elif choice==2:
         self.moove(x=-1, y=1)
      elif choice==3:
         self.moove(x=1, y=-1)

   def moove(self, x=False, y=False):
      if not x:
         self.x+=np.random.randint(-1, 2)
      else:
         self.x+=x

      if not y:
         self.y+=np.random.randint(-1, 2)
      else:
         self.y+=y

      if self.x<0:
         self.x=0
      elif self.x>self.size-1:
         self.x=self.size-1

      if self.y<0:
         self.y=0
      elif self.y>self.size-1:
         self.y=self.size-1