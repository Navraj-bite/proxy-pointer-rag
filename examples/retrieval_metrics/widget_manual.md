# 1. OVERVIEW

The Acme Widget Press is a fictional machine used only to exercise the
retrieval-metrics tooling. Nothing in this file describes real equipment.

## 1.1. Control panel

The control panel sits on the left side of the press. It has a green START
button, a red STOP button and a key switch with the positions Run and Setup.

## 1.2. Status lamp

The status lamp on top of the press is blue when idle, amber while warming up
and flashes white when a fault is active.

# 2. OPERATION

## 2.1. Start-up procedure

1. Turn the key switch to Run.
2. Wait for the status lamp to turn from amber to blue.
3. Press the green START button.
4. Confirm the batch size on the display and press OK.

## 2.2. Shutdown procedure

1. Press the red STOP button.
2. Turn the key switch to Setup.
3. Wait two minutes before opening the guard.

# 3. FAULT CODES

| Code | Meaning              | Action                         |
| ---- | -------------------- | ------------------------------ |
| W01  | Guard open           | Close the guard, press START.  |
| W02  | Low hydraulic oil    | Top up oil to the MAX mark.    |
| W03  | Die temperature high | Let the die cool for 10 min.   |

## 3.1. Resetting a fault

After fixing the cause, hold STOP for three seconds to clear the fault code.
