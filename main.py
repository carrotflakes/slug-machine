from slug_machine import SlugMachine, Slug
from tf_slug import TFSlug
from slug_trace import SlugTrace
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('trace_file', type=str)
parser.add_argument('--tape_width', type=int, default=8)
parser.add_argument('--state_size', type=int, default=8)
parser.add_argument('--epoch', type=int, default=500)
parser.add_argument('--initial_tape_idx', type=int, default=0)


if __name__ == '__main__':
    args = parser.parse_args()

    trace_file = args.trace_file
    tape_width = args.tape_width
    state_size = args.state_size
    epoch = args.epoch
    initial_tape_idx = args.initial_tape_idx

    trace = SlugTrace.load(trace_file, tape_width=tape_width)

    slug = TFSlug(tape_width=tape_width, state_size=state_size)
    slug.learn(trace, epoch=epoch)

    sm = SlugMachine(tape_width, slug, trace.episodes[initial_tape_idx][0])

    sm.print()
    for i in range(30):
        sm.step()
        if sm.pos == 0:
            sm.print()
        if sm.end():
            break

    print('last')
    sm.print()
