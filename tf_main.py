import numpy as np
from slug_machine import SlugMachine, Slug
from tf_slug import TFSlug
from slug_trace import SlugTrace

tape_width = 8

trace = SlugTrace.load('trace_test.txt', tape_width=tape_width)

slug = TFSlug(tape_width=tape_width, state_size=8)
slug.learn(trace, epoch=500)

sm = SlugMachine(tape_width, slug, trace.episodes[2][0])

sm.print()
for i in range(20):
    sm.step()
    if sm.pos == 0:
        sm.print()
    if sm.end():
        break

print('last')
sm.print()
