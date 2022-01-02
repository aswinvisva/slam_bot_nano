import termios, fcntl, sys, os, tty
import select

def key_pressed():
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

class KeyboardInput:

    def __init__(self):
        # Save the terminal settings
        fd = sys.stdin.fileno()
        self.oldterm = termios.tcgetattr(fd)
        newattr = termios.tcgetattr(fd)
        newattr[3] = newattr[3] & ~termios.ICANON & ~termios.ECHO
        termios.tcsetattr(fd, termios.TCSANOW, newattr)

        oldflags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, oldflags | os.O_NONBLOCK)

    def __exit__(self, type, value, traceback):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)

    def getch(self):
        return sys.stdin.read(1)
