import sys

class File():
    def __init__(self, log, name, mode='w'):
        if log:
            self.f = open(name, mode, buffering=1)
            self.mode = 'file'
        else:
            self.f = sys.stdout
            self.mode = 'sys'
    def write(self, msg, terminal=False, start='', end='\n', r_white=False):
        if r_white:
            msg = msg + '                                        '
        msg = '{}{}'.format(start, msg)
        if self.mode == 'file':
            self.f.write(msg + end)
        if self.mode == 'sys' or terminal:
            print(msg, end=end, flush=('\r' in end))

def find_nth(haystack, needle, n):
    if n == 0:
        return n
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start

def find_arg(path, parent_dir, arg_index):
    if parent_dir:
        path = path.replace(parent_dir, '')
    start = find_nth(path, '_', arg_index-1)
    end = find_nth(path, '_', arg_index)
    arg = path[(start + 1 if start > 0 else start):end]
    return arg

def createIdentifier(args):
    identifier = ''
    for arg in args:
        if arg:
            identifier += arg + '_'
    return identifier[:-1]