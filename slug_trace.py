import numpy as np

class SlugTrace:

    def __init__(self, tape_width, tapes):
        self.tape_width = tape_width
        self.tapes = tapes

    @staticmethod
    def load(file_path, tape_width):
        tapes = []
        tape = []
        for line in open(file_path):
            line = line.strip()
            if line == '':
                continue
            tape.append(np.array([int(c) for c in line], dtype=np.bool))
            if sum(tape[-1]) == 0:
                tapes.append(tape)
                tape = []
        return SlugTrace(tape_width, tapes)

    def dump(self):
        for tape in self.tapes:
            for chunk in tape:
                print(''.join(str(int(c)) for c in chunk))
            print()

    def get_tape_transitions(self):
        tape_before, tape_after = [], []
        for tape1, tape2 in zip(self.tapes, self.tapes[1:]):
            for c1, c2 in zip(tape1 + [np.zeros((self.tape_width,), dtype=np.bool)] * (len(tape2) - len(tape1)), tape2):
                tape_before.append(c1)
                tape_after.append(c2)
        return tape_before, tape_after


if __name__ == '__main__':
    trace = SlugTrace.load('trace.txt', tape_width=8)
    trace.dump()

    tape1, tape2 = trace.get_tape_transitions()
    for chunk1, chunk2 in zip(tape1, tape2):
        print('{} {}'.format(''.join(str(int(c)) for c in chunk1), ''.join(str(int(c)) for c in chunk2)))
