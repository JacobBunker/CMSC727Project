import numpy as np

class ActionSpace():
    def __init__(self, size):
        self.size = size
        self.rows = np.power(2, self.size)
        self.actions = np.zeros((self.rows,self.size))
        c = 0
        t = self.rows
        while(c < size):
            t = t/2
            x = t
            i = 0
            while(i < self.rows):
                if(x > 0):
                    self.actions[i,c] = 1
                    x -= 1
                elif(x > -t + 1):
                    self.actions[i,c] = 0
                    x -= 1
                else:
                    x = t
                i += 1
            c += 1
        self.actions[0,:] = np.array([0.,0.,0.,0.,0.,0.,0.,0.])
        self.actions[1,:] = np.array([1.,0.,0.,0.,0.,0.,0.,0.])
    def sample(self):
        t = np.random.randint(self.rows)
        t = np.random.randint(2)
        return t
            