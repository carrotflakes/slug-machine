import numpy as np
from slug_machine import SlugMachine, Slug

tape_width = 16

class MySlug(Slug):

    def __init__(self):
        self.i = 1

    def step(self, chunk):
        n = self.i#sum(v * 2 ** i for i, v in enumerate(chunk)) + 1
        self.i = (self.i + 1) % 10
        return np.array([n & (2 ** i) for i in range(tape_width)])


sm = SlugMachine(tape_width, MySlug())
sm.print()

for i in range(20):
    sm.step()
    sm.print()
