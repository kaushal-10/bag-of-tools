"""
This script handles the GPIO controlling for the Raspberry Pi.
It utilizes the RPi.GPIO library to interact with the GPIO pins on the Raspberry Pi.

The public API mirrors GPIODController so the two can be used interchangeably:

    controller = RPiGPIOController(gpio_config=cfg)   # or GPIODController(...)
    controller.set_value("LED1", 1)
    controller.get_value("LED1")          # -> State.HIGH
    controller.toggle("LED1")

    led = controller.get_pin("LED1")
    led.set(1)
    led.get()                             # -> State.HIGH
    led.toggle()
"""

import os
import traceback

import RPi.GPIO as GPIO

from gpio_classes.gpio_dataclasses import (
    GPIOPin, Direction, State, GPIOMode,
    Pi5HeadToGPIOMap,
    parse_pin_offset as _parse_pin_offset_base,
    parse_direction as _parse_direction,
    parse_state as _parse_state,
)


def _parse_pin_offset(pin) -> int:
    """Resolve a pin identifier to a BCM GPIO offset using Pi5HeadToGPIOMap."""
    return _parse_pin_offset_base(pin, Pi5HeadToGPIOMap)


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class RPiGPIOController:
    def __init__(self, gpio_config: dict = {}, chip_name: str = "/dev/gpiochip0"):
        """
        Initialise the RPiGPIOController.

        Args:
            gpio_config (dict): GPIO pin configuration dictionary.
                Example::

                    {
                        "LED1": {
                            "pin": "GPIO17",  # BCM GPIO number → offset 17 directly
                            "direction": "output",
                            "initial": 0,
                            "enabled": True,
                        },
                        "LED2": {
                            "pin": "15",      # physical header pin 15 → GPIO22
                            "direction": "output",
                            "initial": 0,
                            "enabled": True,
                        },
                        "BUTTON1": {
                            "pin": "GPIO27",
                            "direction": "input",
                            "enabled": True,
                        },
                    }

                Pin addressing rules:

                * ``"GPIO<n>"`` — BCM GPIO offset *n* used directly.
                * ``"<n>"`` or integer *n* — physical 40-pin header number,
                  resolved via ``Pi5HeadToGPIOMap``.

                ``"pinmode"`` is accepted as an alias for ``"direction"``.

            chip_name (str): Accepted for API compatibility with
                GPIODController; not used by RPi.GPIO (which operates over
                the BCM kernel layer directly).
        """
        self.chip_name = chip_name          # kept for API compatibility
        self.gpio_config = gpio_config

        # {pin_name: int}  — BCM offset, kept for cleanup
        self._offsets: dict[str, int] = {}
        # {pin_name: GPIOPin}
        self._pins: dict[str, GPIOPin] = {}

        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            # Probe that the library can actually talk to the hardware.
            # RPi.GPIO raises RuntimeError on Pi 5 because it cannot
            # determine the SoC peripheral base address.
            GPIO.gpio_function(0)
        except RuntimeError as e:
            GPIO.cleanup()
            raise RuntimeError(
                "RPi.GPIO failed to initialise — this is most likely because "
                "RPi.GPIO does not support the Raspberry Pi 5 (or newer). "
                "Use GPIODController (gpiod-based) instead, which works on all "
                "Pi models including Pi 5.\n"
                f"Original error: {e}"
            ) from e

        print("[RPiGPIOController] Initialised RPi.GPIO in BCM mode.")
        self._initialize_pins()

    # ------------------------------------------------------------------
    # Internal initialisation
    # ------------------------------------------------------------------

    def _initialize_pins(self):
        for pin_name, pin_info in self.gpio_config.items():
            if not pin_info.get("enabled", True):
                continue
            try:
                pin_id   = pin_info.get("pin", pin_name)
                offset   = _parse_pin_offset(pin_id)
                direction = _parse_direction(
                    pin_info.get("direction", pin_info.get("pinmode", Direction.OUT))
                )
                initial  = _parse_state(pin_info.get("initial", State.LOW))

                if direction == Direction.OUT:
                    GPIO.setup(offset, GPIO.OUT, initial=int(initial))
                else:
                    GPIO.setup(offset, GPIO.IN)

                self._offsets[pin_name] = offset
                pin = GPIOPin(
                    name=pin_name,
                    gpio_number=offset,
                    direction=direction.value,
                    value=initial,
                    enabled=True,
                )
                # Inject live callbacks
                pin._setter = lambda v, _n=pin_name: self.set_value(_n, v)
                pin._getter = lambda _n=pin_name: self.get_value(_n)
                self._pins[pin_name] = pin
                print(
                    f"[RPiGPIOController]   Configured '{pin_name}' "
                    f"offset={offset} dir={direction.value}"
                )

            except Exception:
                print(f"[RPiGPIOController] ERROR configuring pin '{pin_name}':")
                traceback.print_exc()

    # ------------------------------------------------------------------
    # Public API  (mirrors GPIODController exactly)
    # ------------------------------------------------------------------

    def setup_gpio(self):
        """(Re-)initialise all configured pins."""
        self.cleanup()
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        self._initialize_pins()

    def get_pin(self, pin_name: str) -> GPIOPin:
        """
        Return the :class:`GPIOPin` for *pin_name*, with live
        ``.set()`` / ``.get()`` / ``.toggle()`` callbacks already injected::

            led = controller.get_pin("LED1")
            led.set(1)
            state = led.get()   # State.HIGH / State.LOW
            led.toggle()

        Raises:
            KeyError: If *pin_name* has not been configured.
        """
        if pin_name not in self._pins:
            raise KeyError(f"Pin '{pin_name}' is not configured.")
        return self._pins[pin_name]

    def get_all_pins(self) -> dict[str, GPIOPin]:
        """Return all configured :class:`GPIOPin` objects (with live callbacks)."""
        return dict(self._pins)

    def set_value(self, pin_name: str, value: "int | State"):
        """
        Set the output value of a configured output pin.

        Args:
            pin_name (str): Key from the gpio_config dictionary.
            value (int | State): ``State.LOW`` / ``0`` for LOW,
                ``State.HIGH`` / ``1`` for HIGH.

        Raises:
            KeyError: If *pin_name* has not been configured.
            RuntimeError: If the pin is not configured as an output.
        """
        if pin_name not in self._pins:
            raise KeyError(f"Pin '{pin_name}' is not configured.")
        pin = self._pins[pin_name]
        if pin.direction != Direction.OUT.value:
            raise RuntimeError(f"Pin '{pin_name}' is not configured as an output.")

        state = _parse_state(value)
        GPIO.output(self._offsets[pin_name], int(state))
        pin.value = state   # keep GPIOPin in sync

    def get_value(self, pin_name: str) -> State:
        """
        Read the current value of a configured pin.

        Returns:
            State: ``State.LOW`` or ``State.HIGH``.

        Raises:
            KeyError: If *pin_name* has not been configured.
        """
        if pin_name not in self._pins:
            raise KeyError(f"Pin '{pin_name}' is not configured.")

        state = _parse_state(GPIO.input(self._offsets[pin_name]))
        self._pins[pin_name].value = state  # keep GPIOPin in sync
        return state

    def write(self, pin_name: str, value: "int | State"):
        """Alias for :meth:`set_value`."""
        self.set_value(pin_name, value)

    def read(self, pin_name: str) -> State:
        """Alias for :meth:`get_value`."""
        return self.get_value(pin_name)

    def toggle(self, pin_name: str):
        """Toggle the output state of a configured output pin."""
        current = self.get_value(pin_name)
        self.set_value(pin_name, State.LOW if current == State.HIGH else State.HIGH)

    def cleanup(self):
        """Release all configured GPIO channels and reset RPi.GPIO state."""
        offsets = list(self._offsets.values())
        if offsets:
            try:
                GPIO.cleanup(offsets)
            except Exception:
                pass
        self._offsets.clear()
        self._pins.clear()

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        pin_summary = {
            name: f"GPIO{p.gpio_number}({p.direction})"
            for name, p in self._pins.items()
        }
        return f"RPiGPIOController(pins={pin_summary})"


if __name__ == "__main__":
    gpio_config = {
        "LED1": {
            "pin": "GPIO17",
            "direction": "output",
            "initial": 0,
            "enabled": True,
        },
        "LED2": {
            "pin": "15",        # header pin 15 → GPIO22
            "direction": "output",
            "initial": 0,
            "enabled": True,
        },
        "BUTTON1": {
            "pin": "GPIO27",
            "direction": "input",
            "enabled": True,
        },
    }

    controller = RPiGPIOController(gpio_config=gpio_config) 
    # Via controller
    print(f"Initial LED1 : {controller.get_value('LED1')}")
    controller.set_value("LED1", 1)
    print(f"LED1 HIGH    : {controller.get_value('LED1')}")
    controller.toggle("LED1")
    print(f"LED1 toggled : {controller.get_value('LED1')}")

    # Via GPIOPin
    led1 = controller.get_pin("LED1")
    led1.set(State.HIGH)
    print(f"led1.get()   : {led1.get()}")
    led1.toggle()
    print(f"led1 toggled : {led1.get()}  value={led1.value}")

    btn = controller.get_pin("BUTTON1")
    print(f"BUTTON1      : {btn.get()}  (int: {int(btn.get())})")

    controller.cleanup()

    
