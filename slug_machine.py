import numpy as np

class SlugMachine:

    def __init__(self, tape_width, slug):
        self.slug = slug
        self.tape_width = tape_width
        self.tape = [np.zeros((self.tape_width,), dtype=np.bool), np.zeros((self.tape_width,), dtype=np.bool)]
        self.tape[0][0] = True
        self.pos = 0

    def step(self):
        vec = self.slug.step(self.tape[self.pos])
        vec = vec >= 0.5
        end = sum(vec) == 0
        self.tape[self.pos] = vec
        if end:
            self.tape = self.tape[:self.pos+1]
            self.pos = 0
        else:
            self.pos += 1
            if len(self.tape) == self.pos:
                self.tape.append(np.zeros((self.tape_width,), dtype=np.bool))

    def end(self):
        return len(self.tape) <= 1

    def print(self):
        for i, chunk in enumerate(self.tape):
            print(self.line_dump(chunk), end='')
            if i == self.pos:
                print(' <- ', end='')
                self.slug.print()
            else:
                print()
        print()

    def line_dump(self, chunk):
        return ''.join(str(int(c)) for c in chunk)

class Slug:

    def __init__(self):
        pass

    def step(self, chunk):
        raise NotImplemented()

    def print(self):
        print("(^-^)")
