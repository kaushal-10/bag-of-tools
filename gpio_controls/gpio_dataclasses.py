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


# ---------------------------------------------------------------------------
# Shared pin-identifier / direction / state parsers
# (used by all GPIO controller implementations)
# ---------------------------------------------------------------------------

def parse_pin_offset(pin, head_to_gpio_map: dict) -> int:
    """
    Convert a pin identifier to an integer GPIO line offset.

    Two addressing modes:

    * ``"GPIO17"`` / ``"gpio17"`` — BCM GPIO number used directly → ``17``
    * ``"11"`` / ``11`` (int)     — physical 40-pin header number,
      resolved via *head_to_gpio_map* (e.g. ``Pi5HeadToGPIOMap``)

    Raises:
        ValueError: Header pin not in *head_to_gpio_map*.
        TypeError:  Value cannot be interpreted as a pin identifier.
    """
    if isinstance(pin, int):
        if pin not in head_to_gpio_map:
            raise ValueError(
                f"Header pin {pin} is not a GPIO-capable pin. "
                f"Valid header pins: {sorted(head_to_gpio_map.keys())}"
            )
        return head_to_gpio_map[pin]

    pin_str = str(pin).strip()
    upper = pin_str.upper()
    if upper.startswith("GPIO"):
        return int(upper[4:])

    try:
        header_pin = int(pin_str)
    except ValueError:
        raise TypeError(
            f"Cannot interpret '{pin}' as a pin identifier. "
            "Use 'GPIO<n>' for a direct GPIO offset or a plain integer for a header pin number."
        )
    if header_pin not in head_to_gpio_map:
        raise ValueError(
            f"Header pin {header_pin} is not a GPIO-capable pin. "
            f"Valid header pins: {sorted(head_to_gpio_map.keys())}"
        )
    return head_to_gpio_map[header_pin]


def parse_direction(raw) -> Direction:
    """
    Normalise any direction-like value to a :class:`Direction` enum member.

    Accepted: ``Direction.IN/OUT`` (pass-through), ``"input"/"in"`` → IN,
    ``"output"/"out"`` → OUT.
    """
    if isinstance(raw, Direction):
        return raw
    s = str(raw).strip().lower()
    if s in ("input", "in"):
        return Direction.IN
    if s in ("output", "out"):
        return Direction.OUT
    raise ValueError(
        f"Invalid direction '{raw}'. Must be one of: 'input', 'in', 'output', 'out'."
    )


def parse_state(raw) -> State:
    """
    Normalise any state-like value to a :class:`State` enum member.

    Accepted: ``State.LOW/HIGH`` (pass-through), ``0``/falsy → LOW,
    ``1``/truthy → HIGH.  Because ``State`` is ``(int, Enum)``, the
    returned value can be used directly wherever an ``int`` is expected.
    """
    if isinstance(raw, State):
        return raw
    return State(1 if raw else 0)


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



# ---------------------------------------------------------------------------
# Jetson Nano (JetPack 4.x / L4T 32.x) — 40-pin header GPIO mappings
# Source: NVIDIA Jetson Nano Developer Kit pinout documentation
# ---------------------------------------------------------------------------

JNanoHeadToGPIOMap = {
    7: 216, 11: 50, 12: 79, 13: 14, 15: 194,
    16: 232, 18: 15, 19: 16, 21: 17, 22: 13,
    23: 18, 24: 19, 26: 20, 29: 149, 31: 200,
    32: 168, 33: 38, 35: 76, 36: 51, 37: 12,
    38: 77, 40: 78,
}

JNanoGPIOToHeadMap = {v: k for k, v in JNanoHeadToGPIOMap.items()}


# ---------------------------------------------------------------------------
# Jetson Orin Nano (JetPack 5.x / L4T 35.x)
# Partial map — verified pins only (extend as needed).
# ---------------------------------------------------------------------------

JONanoJP5HeadToGPIOMap = {
    7: 492, 12: 398, 29: 453, 31: 454,
    33: 391, 35: 401, 38: 400, 40: 399,
}

JONanoJP5GPIOToHeadMap = {v: k for k, v in JONanoJP5HeadToGPIOMap.items()}


# ---------------------------------------------------------------------------
# Jetson Orin Nano (JetPack 6.x / L4T 36.x)
# GPIO numbering changed from JP5 due to kernel pin-controller updates.
# Populate with verified values for your specific board.
# ---------------------------------------------------------------------------

JONanoJP6HeadToGPIOMap: dict[int, int] = {}

JONanoJP6GPIOToHeadMap: dict[int, int] = {}

