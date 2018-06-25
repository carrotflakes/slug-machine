import numpy as np

class SlugTrace:

    def __init__(self, tape_width, episodes):
        self.tape_width = tape_width
        self.episodes = episodes

    @staticmethod
    def load(file_path, tape_width):
        episodes = []
        tapes = []
        tape = []
        for line in open(file_path):
            line = line.strip()
            if line == '' or line.startswith('#'):
                continue
            tape.append(np.array([int(c) for c in line], dtype=np.bool))
            if sum(tape[-1]) == 0:
                tapes.append(tape)
                if len(tape) == 1:
                    episodes.append(tapes)
                    tapes = []
                tape = []
        return SlugTrace(tape_width, episodes)

    def dump(self):
        for i, tapes in enumerate(self.episodes):
            print('#{}'.format(i))
            for tape in tapes:
                for chunk in tape:
                    print(''.join(str(int(c)) for c in chunk))
                print()

    def get_tape_transitions(self):
        episodes = []
        for tapes in self.episodes:
            episode = []
            for tape1, tape2 in zip(tapes, tapes[1:]):
                episode.extend(list(zip(tape1 + [np.zeros((self.tape_width,), dtype=np.bool)] * (len(tape2) - len(tape1)), tape2)))
            episodes.append(episode)
        return episodes


if __name__ == '__main__':
    trace = SlugTrace.load('trace.txt', tape_width=8)
    trace.dump()

    episodes = trace.get_tape_transitions()
    for episode in episodes:
        print()
        for chunk1, chunk2 in episode:
            print('{} {}'.format(''.join(str(int(c)) for c in chunk1), ''.join(str(int(c)) for c in chunk2)))
