import time
import sys
import pyperclip

# Import win32clipboard if on Windows
if sys.platform == "win32":
    try:
        import win32con
        import win32clipboard
    except ImportError:
        win32clipboard = None
else:
    win32clipboard = None

def _fast_win_clip(text: str) -> None:
    if not win32clipboard:
        raise RuntimeError("win32clipboard not available")
    win32clipboard.OpenClipboard()
    try:
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardData(win32con.CF_UNICODETEXT, text)
    finally:
        win32clipboard.CloseClipboard()

TEXT_TO_COPY = "This is a test of clipboard performance."

print("Testing clipboard performance...")

# Test pyperclip
start_pyperclip = time.monotonic()
try:
    pyperclip.copy(TEXT_TO_COPY)
    duration_pyperclip = time.monotonic() - start_pyperclip
    print(f"pyperclip.copy() took: {duration_pyperclip:.4f} seconds")
except Exception as e:
    print(f"pyperclip.copy() failed: {e}")

# Test _fast_win_clip on Windows
if win32clipboard:
    start_fast_clip = time.monotonic()
    try:
        _fast_win_clip(TEXT_TO_COPY)
        duration_fast_clip = time.monotonic() - start_fast_clip
        print(f"_fast_win_clip() took: {duration_fast_clip:.4f} seconds")
    except Exception as e:
        print(f"_fast_win_clip() failed: {e}")

print("\nDone.") 