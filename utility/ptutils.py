import time
import math
import collections


class Display:
    """Write on terminal statistics in a fancy way.

    Colors are used to signal variations in the data.

    Example:
    
    display = Display("Step {step}/{}   loss: {loss:.2f}")
    display.disp(10, 100, loss=3.14159)

    It would print the message:

    Step 10/100    loss 3.14

    with "3.14" colored according to historical variation of the loss
    value.

    Named fields (such as "loss") are tracked and displayed in color.
    Unnamed fields are not tracked.  "step" is a special untracked field,
    and "steps_s" is a tracked field that is automatically computed.
    

    """
    def __init__(self, format_string):
        """Create the display object.

        The format string encodes how information should be displayed.
        """
        self.fmt = format_string
        self.vars_ = collections.defaultdict(_DisplayVar)
        self.steps_s = _DisplayVar()
        self.last_step = None
        self.last_time = None

    def message(self, step, *fields, **data):
        """Compose a message with the given information."""
        self._update_steps_s(step)
        d = dict((k, self._update_var(k, v)) for (k, v) in data.items())
        return self.fmt.format(*fields, step=step, steps_s=self.steps_s, **d)
        
    def disp(self, step, *fields, **data):
        """Print on stdout the given information according the the format of the display."""
        print(self.message(step, *fields, **data))

    def _update_var(self, k, v):
        dv = self.vars_[k]
        dv.add(v)
        return dv

    def _update_steps_s(self, step):
        tm = time.perf_counter()
        if self.last_step is None or self.last_time >= tm:
            speed = float("nan")
        else:
            speed = (step - self.last_step) / (tm - self.last_time)
        self.last_time = tm
        self.last_step = step
        self.steps_s.add(speed)


class _DisplayVar:
    """Track the history of a value and format its last value accordingly."""

    # Ansi codes for colors and styles
    MIN = "\x1B[1;32m"    # bold green
    LOW = "\x1B[0;32m"    # green
    NORMAL = "\x1B[0;33m" # yellow
    HIGH = "\x1B[0;31m"   # red
    MAX = "\x1B[1;31m"    # bold red
    NAN = "\x1B[1;36m"    # cyan
    RESET = "\x1B[0m"     # default style
    
    def __init__(self, history_len=10):
        """Initialize the object.

        Remembers up to history_len values.
        """
        self.history = collections.deque(maxlen=history_len)
        self.minval = self.maxval = None
        self.lastvalue = float("nan")
        self.state = self.NAN
        
    def add(self, value):
        """Add a new value to the series."""
        self.lastvalue = value
        if math.isnan(value):
            self.state = self.NAN
        elif not self.history:
            self.state = self.NORMAL
            self.history.append(value)
            self.minval = self.maxval = value
        else:
            _, s = min((min(self.history), self.NORMAL), (value, self.LOW))
            _, s = max((max(self.history), s), (value, self.HIGH))
            self.maxval, _, s = max((self.maxval, 1, s), (value, 0, self.MAX))
            self.minval, _, s = min((self.minval, 0, s), (value, 1, self.MIN))
            self.state = s
            self.history.append(value)

    def __format__(self, spec):
        """Format the last added value."""
        s = self.lastvalue.__format__(spec)
        return self.state + s + self.RESET
    

def _demo():
    import random
    fmt = "Step: {step:3d}/{}  Loss: {loss:6.3f}  {steps_s:6.4f} steps/s"
    display = Display(fmt)
    for step in range(1, 101):
        time.sleep(1)
        display.disp(step, 100, loss=random.random() * 100)
    

if __name__ == "__main__":
    _demo()
    # session = Session("model_dir", save_every=100, save_count=5, state=[model, optimizer], max_epocs=3)
    # session.add_state(model)
    # session.add_state(optimizer)
    # session.restore()
    # for x ,y in session.train_loop(loader):
    #     session.step
    #     session.epoc
