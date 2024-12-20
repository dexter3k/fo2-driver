'''
Control - messy user32::DirectInput code to press buttons in the game.

Not that on keyboards with num-keys, the num-pad must be in arrow-key mode for this to work.

Otherwise remap ingame controls to eg awsd and use those

'''

import ctypes as ct
import ctypes.wintypes as w
import time

# Constants
KEYEVENTF_SCANCODE = 0x8
KEYEVENTF_KEYUP = 0x2
KEYEVENTF_UNICODE = 0x4
INPUT_KEYBOARD = 1

# Define ULONG_PTR for 64-bit compatibility
ULONG_PTR = ct.c_size_t

# Structure Definitions
class KEYBDINPUT(ct.Structure):
    _fields_ = [
        ('wVk', w.WORD),          # Virtual Key (unused in scancode mode)
        ('wScan', w.WORD),        # Scan code
        ('dwFlags', w.DWORD),     # Key event flags
        ('time', w.DWORD),        # Timestamp (not used here)
        ('dwExtraInfo', ULONG_PTR) # Extra information
    ]

class HARDWAREINPUT(ct.Structure):
    _fields_ = [
        ('uMsg', w.DWORD),
        ('wParamL', w.WORD),
        ('wParamH', w.WORD)
    ]

class MOUSEINPUT(ct.Structure):
    _fields_ = [
        ('dx', w.LONG),
        ('dy', w.LONG),
        ('mouseData', w.DWORD),
        ('dwFlags', w.DWORD),
        ('time', w.DWORD),
        ('dwExtraInfo', ULONG_PTR)
    ]

class DUMMYUNIONNAME(ct.Union):
    _fields_ = [
        ('mi', MOUSEINPUT),
        ('ki', KEYBDINPUT),
        ('hi', HARDWAREINPUT)
    ]

class INPUT(ct.Structure):
    _anonymous_ = ['u']
    _fields_ = [
        ('type', w.DWORD),
        ('u', DUMMYUNIONNAME)
    ]

# Function for error checking
def zerocheck(result, func, args):
    if result == 0:
        raise ct.WinError(ct.get_last_error())
    return result

# Load user32.dll and configure SendInput
user32 = ct.WinDLL('user32', use_last_error=True)
SendInput = user32.SendInput
SendInput.argtypes = [w.UINT, ct.POINTER(INPUT), ct.c_int]
SendInput.restype = w.UINT
SendInput.errcheck = zerocheck

# Key press and release using scan codes
def press_key(scancode):
    i = INPUT()
    i.type = INPUT_KEYBOARD
    i.ki = KEYBDINPUT(0, scancode, KEYEVENTF_SCANCODE, 0, 0)
    SendInput(1, ct.byref(i), ct.sizeof(INPUT))

def release_key(scancode):
    i = INPUT()
    i.type = INPUT_KEYBOARD
    i.ki = KEYBDINPUT(0, scancode, KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP, 0, 0)
    SendInput(1, ct.byref(i), ct.sizeof(INPUT))

def press_and_release_key(scancode, sleepAfter=True):
    press_key(scancode)
    time.sleep(0.1)
    release_key(scancode)
    if sleepAfter:
        time.sleep(0.1)

# Key mappings (DirectInput scan codes)
KEY_UP = 0xC8
KEY_DOWN = 0xD0
KEY_LEFT = 0xCB
KEY_RIGHT = 0xCD

KEY_RESET = 0x13

KEY_ESCAPE = 0x01
KEY_ENTER = 0x1C
KEY_LEFT_CONTROL = 0x1D

# Example Usage
if __name__ == "__main__":
    print(hex(user32.MapVirtualKeyA(0x26, 0)))
    time.sleep(2)

    while True:
        press_key(KEY_UP)
        time.sleep(2)
        release_key(KEY_UP)
        time.sleep(2)
