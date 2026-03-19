import os
import time
import sys

from dataclasses import dataclass, field
from typing import Callable, Optional
from enum import Enum


@dataclass
class GPIOPin:
    """
    Represents a single GPIO pin and its current state.

    When a controller configures a pin it injects ``_setter`` and ``_getter``
    callbacks so the pin can be driven directly::

        led = controller.get_pin("LED1")
        led.set(1)          # drive HIGH  (also accepts State.HIGH)
        led.set(State.LOW)  # drive LOW
        state = led.get()   # -> State.HIGH / State.LOW
        led.toggle()        # flip output
        print(led.value)    # 0 or 1, kept in sync automatically

    Without injected callbacks the pin acts as a plain data container
    (useful for mock / offline usage).
    """
    name: str
    gpio_number: int
    direction: str          # "in" | "out"  (Direction.value)
    value: int = 0
    enabled: bool = True
    # Injected by the owning controller — hidden from repr / equality checks
    _setter: Optional[Callable] = field(default=None, repr=False, compare=False)
    _getter: Optional[Callable] = field(default=None, repr=False, compare=False)

    # ------------------------------------------------------------------
    # Control API
    # ------------------------------------------------------------------

    def set(self, value: "int | State"):
        """
        Drive this output pin HIGH (``1`` / ``State.HIGH``) or LOW
        (``0`` / ``State.LOW``).

        Raises:
            RuntimeError: If no setter has been injected (pin not bound
                to a controller) or if the pin is not an output.
        """
        if self._setter is None:
            raise RuntimeError(
                f"Pin '{self.name}' has no setter bound. "
                "Make sure it was obtained from a controller via get_pin()."
            )
        if self.direction != "out":
            raise RuntimeError(
                f"Pin '{self.name}' is configured as '{self.direction}' — cannot set value."
            )
        self._setter(value)

    def get(self) -> "State":
        """
        Read and return the current :class:`State` of this pin.

        Raises:
            RuntimeError: If no getter has been injected.
        """
        if self._getter is None:
            raise RuntimeError(
                f"Pin '{self.name}' has no getter bound. "
                "Make sure it was obtained from a controller via get_pin()."
            )
        return self._getter()

    def toggle(self):
        """
        Toggle an output pin between HIGH and LOW.

        Raises:
            RuntimeError: If no setter/getter has been injected or the pin
                is not an output.
        """
        current = self.get()
        self.set(State.LOW if current == State.HIGH else State.HIGH)

class Direction(Enum):
    IN = "in"
    OUT = "out"

class State(int, Enum):
    LOW = 0
    HIGH = 1


class GPIOMode(str, Enum):
    EXTERNAL = "external"
    GPIOD = "gpiod"
    JETSONGPIO = "jetson_gpio"
    RPIGPIO = "rpi_gpio"
    F232H = "f232h"
    CP2102 = "cp2102"


class Pi5GPIOToHeadMap(int, Enum):
    GPIO2 = 3
    GPIO3 = 5
    GPIO4 = 7
    GPIO17 = 11
    GPIO27 = 13
    GPIO22 = 15
    GPIO10 = 19
    GPIO9 = 21
    GPIO11 = 23
    GPIO0 = 27
    GPIO5 = 29
    GPIO6 = 31
    GPIO13 = 33
    GPIO19 = 35
    GPIO26 = 37
    GPIO14 = 8
    GPIO15 = 10
    GPIO18 = 12
    GPIO23 = 16
    GPIO24 = 18 
    GPIO25 = 22
    GPIO8 = 24
    GPIO7 = 26
    GPIO12 = 32
    GPIO16 = 36
    GPIO20 = 38
    GPIO21 = 40


Pi5HeadToGPIOMap = {
    3 : 2,
    5 : 3,
    7 : 4,
    11 : 17,
    13 : 27,
    15 : 22,
    19 : 10,
    21 : 9,
    23 : 11,
    27 : 0,
    29 : 5,
    31 : 6,
    33 : 13,
    35 : 19,
    37 : 26,
    8 : 14,
    10 : 15,
    12 : 18,
    16 : 23,
    18 : 24,
    22 : 25,
    24 : 8,
    26 : 7,
    32 : 12,
    36 : 16,
    38 : 20,
    40 : 21

}
