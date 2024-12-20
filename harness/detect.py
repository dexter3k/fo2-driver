import time

import win32gui
import pymem

PROC_NAME = 'FlatOut2_gog.exe'
WINDOW_NAME = 'FlatOut 2'

class Base:
    def __init__(self, mem, ptr):
        self.mem = mem
        self.ptr = ptr

    def get_position_along_the_track(self):
        return self.mem.read_float(self.ptr + 0x01C)

    def get_game_has_input(self):
        return self.mem.read_int(self.ptr + 0x2DC) != 0

    def get_seconds_since_start(self):
        """
            Starts with -3 (start countdown) as expected
        """
        return self.mem.read_int(self.ptr + 0x3B0) / 1000

    def get_is_on_track(self):
        return self.mem.read_int(self.ptr + 0x390) == 0

    def get_speed_along_the_track(self):
        return self.mem.read_float(self.ptr + 0x41c)

class Harness:
    def __init__(self, window_name=WINDOW_NAME, proc_name=PROC_NAME):
        self.window_name = window_name
        self.proc_name = proc_name

        self.hwnd = win32gui.FindWindow(None, self.window_name)
        if self.hwnd == 0:
            raise Exception(f'Unable to find window {self.window_name}')

        try:
            self.mem = pymem.Pymem(self.proc_name)
        except pymem.exception.ProcessNotFound:
            raise Exception(f'Unable to find process {self.proc_name}')

    def is_foreground_window(self):
        return win32gui.GetForegroundWindow() == self.hwnd

    def wait_for_window_activation(self):
        while True:
            if self.is_foreground_window():
                return True
            time.sleep(1 / 20)

    def get_base(self):
        ptr = self.mem.read_int(0x6B21F8)
        if ptr == 0:
            return None

        return Base(self.mem, ptr)

class TrackTracker:
    def __init__(self, game):
        self.game = game
        self.track_start_offset = 0
        self.last_track_offset = 0
        self.track_lap_counter = 0
        self.track_switch_point = 0

    def reset_position(self):
        pat = self.game.get_base().get_position_along_the_track()
        self.track_start_offset = pat
        self.last_track_offset = pat
        self.track_lap_counter = 0
        self.track_switch_point = 0

    def track_position(self):
        pat = self.game.get_base().get_position_along_the_track()
        if pat - self.last_track_offset < -500:
            self.track_lap_counter += 1
            self.track_switch_point = max(self.track_switch_point, self.last_track_offset)
        elif pat - self.last_track_offset > 500:
            self.track_lap_counter -= 1
            self.track_switch_point = max(self.track_switch_point, pat)
        self.last_track_offset = pat

        return self.get_position()

    def get_position(self):
        pat = self.game.get_base().get_position_along_the_track()
        pat += self.track_lap_counter * self.track_switch_point
        return pat - self.track_start_offset
