class Lexer():
    __slots__=("path", "lines", "cur_line")
    def __init__(self, f: str):
        self.path = f
        with open(f, "r") as fp:
            self.lines = fp.readlines()
        self.cur_line = 0
    
    def is_eof(self):
        return self.cur_line >= len(self.lines)