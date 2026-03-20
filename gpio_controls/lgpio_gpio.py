"""
This script handles the GPIO controlling for the Raspberry Pi.
It utilizes the lgpio library to interact with the GPIO pins.

lgpio works on all Pi models including Raspberry Pi 5, making it a good
alternative to RPi.GPIO (which does not support Pi 5) when gpiod is not
preferred.

lgpio API used here:
    - lgpio.gpiochip_open(chip_num)             -> handle (int)
    - lgpio.gpio_claim_output(handle, gpio, level) -> 0 on success
    - lgpio.gpio_claim_input(handle, gpio)         -> 0 on success
    - lgpio.gpio_write(handle, gpio, level)        -> 0 on success
    - lgpio.gpio_read(handle, gpio)                -> level (0 or 1)
    - lgpio.gpio_free(handle, gpio)                -> 0 on success
    - lgpio.gpiochip_close(handle)                 -> 0 on success
    - lgpio.error_text(err)                        -> human-readable error

The public API is identical to GPIODController and RPiGPIOController so
all three controllers can be used interchangeably.
"""

import os
import traceback
import time
import lgpio

from gpio_classes.gpio_dataclasses import (
    GPIOPin, Direction, State, GPIOMode,
    Pi5HeadToGPIOMap,
    parse_pin_offset as _parse_pin_offset_base,
    parse_direction as _parse_direction,
    parse_state as _parse_state,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_pin_offset(pin) -> int:
    """Resolve a pin identifier to a BCM GPIO offset using Pi5HeadToGPIOMap."""
    return _parse_pin_offset_base(pin, Pi5HeadToGPIOMap)


def _chip_num_from_name(chip_name: str) -> int:
    """
    Extract the integer chip number from a chip device path.

    Examples::

        "/dev/gpiochip0"  ->  0
        "/dev/gpiochip4"  ->  4
    """
    try:
        return int(chip_name.rstrip("/").split("gpiochip")[-1])
    except (ValueError, IndexError):
        raise ValueError(
            f"Cannot extract chip number from '{chip_name}'. "
            "Expected a path like '/dev/gpiochip0'."
        )


def _check_lgpio_call(ret: int, operation: str):
    """Raise RuntimeError if an lgpio call returned a negative error code."""
    if ret < 0:
        raise RuntimeError(
            f"lgpio error during {operation}: {lgpio.error_text(ret)} (code {ret})"
        )


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class LGPIOController:
    def __init__(self, gpio_config: dict = {}, chip_name: str = "/dev/gpiochip0"):
        """
        Initialise the LGPIOController.

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

            chip_name (str): Path to the gpiochip device
                (default ``/dev/gpiochip0``).  The numeric part is extracted
                and passed to ``lgpio.gpiochip_open()``.
        """
        assert "gpiochip" in chip_name, (
            f"Chip name must contain 'gpiochip'. Provided: {chip_name}"
        )
        assert os.path.exists(chip_name), (
            f"Chip device '{chip_name}' not found."
        )

        self.chip_name = chip_name
        self.gpio_config = gpio_config
        self._handle: int = -1

        # {pin_name: int}  — BCM offset, kept for per-pin cleanup
        self._offsets: dict[str, int] = {}
        # {pin_name: GPIOPin}
        self._pins: dict[str, GPIOPin] = {}

        chip_num = _chip_num_from_name(chip_name)
        self._handle = lgpio.gpiochip_open(chip_num)
        if self._handle < 0:
            raise RuntimeError(
                f"lgpio could not open chip '{chip_name}': "
                f"{lgpio.error_text(self._handle)} (code {self._handle})"
            )

        print(f"[LGPIOController] Opened chip '{chip_name}' (handle {self._handle}).")
        self._initialize_pins()

    # ------------------------------------------------------------------
    # Internal initialisation
    # ------------------------------------------------------------------

    def _initialize_pins(self):
        for pin_name, pin_info in self.gpio_config.items():
            if not pin_info.get("enabled", True):
                continue
            try:
                pin_id    = pin_info.get("pin", pin_name)
                offset    = _parse_pin_offset(pin_id)
                direction = _parse_direction(
                    pin_info.get("direction", pin_info.get("pinmode", Direction.OUT))
                )
                initial   = _parse_state(pin_info.get("initial", State.LOW))

                if direction == Direction.OUT:
                    ret = lgpio.gpio_claim_output(self._handle, offset, int(initial))
                    _check_lgpio_call(ret, f"gpio_claim_output({offset})")
                else:
                    ret = lgpio.gpio_claim_input(self._handle, offset)
                    _check_lgpio_call(ret, f"gpio_claim_input({offset})")

                self._offsets[pin_name] = offset
                pin = GPIOPin(
                    name=pin_name,
                    gpio_number=offset,
                    direction=direction.value,
                    value=initial,
                    enabled=True,
                )
                # Inject live callbacks (default-arg captures pin_name per iteration)
                pin._setter = lambda v, _n=pin_name: self.set_value(_n, v)
                pin._getter = lambda _n=pin_name: self.get_value(_n)
                self._pins[pin_name] = pin
                print(
                    f"[LGPIOController]   Configured '{pin_name}' "
                    f"offset={offset} dir={direction.value}"
                )

            except Exception:
                print(f"[LGPIOController] ERROR configuring pin '{pin_name}':")
                traceback.print_exc()

    # ------------------------------------------------------------------
    # Public API  (mirrors GPIODController / RPiGPIOController exactly)
    # ------------------------------------------------------------------

    def setup_gpio(self):
        """(Re-)initialise all configured pins."""
        self.cleanup()
        chip_num = _chip_num_from_name(self.chip_name)
        self._handle = lgpio.gpiochip_open(chip_num)
        if self._handle < 0:
            raise RuntimeError(
                f"lgpio could not reopen chip '{self.chip_name}': "
                f"{lgpio.error_text(self._handle)} (code {self._handle})"
            )
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
            RuntimeError: If the pin is not configured as an output, or if
                the lgpio call fails.
        """
        if pin_name not in self._pins:
            raise KeyError(f"Pin '{pin_name}' is not configured.")
        pin = self._pins[pin_name]
        if pin.direction != Direction.OUT.value:
            raise RuntimeError(f"Pin '{pin_name}' is not configured as an output.")

        state = _parse_state(value)
        ret = lgpio.gpio_write(self._handle, self._offsets[pin_name], int(state))
        _check_lgpio_call(ret, f"gpio_write({pin_name})")

        # Keep GPIOPin in sync
        pin.value = state

    def get_value(self, pin_name: str) -> State:
        """
        Read the current value of a configured pin.

        Returns:
            State: ``State.LOW`` or ``State.HIGH``.

        Raises:
            KeyError: If *pin_name* has not been configured.
            RuntimeError: If the lgpio call fails.
        """
        if pin_name not in self._pins:
            raise KeyError(f"Pin '{pin_name}' is not configured.")

        ret = lgpio.gpio_read(self._handle, self._offsets[pin_name])
        _check_lgpio_call(ret, f"gpio_read({pin_name})")

        state = _parse_state(ret)
        self._pins[pin_name].value = state  # Keep GPIOPin in sync
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
        """Free all claimed GPIO lines and close the chip handle."""
        for offset in list(self._offsets.values()):
            try:
                lgpio.gpio_free(self._handle, offset)
            except Exception:
                pass
        self._offsets.clear()
        self._pins.clear()

        if self._handle >= 0:
            try:
                lgpio.gpiochip_close(self._handle)
            except Exception:
                pass
            self._handle = -1

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
        return (
            f"LGPIOController(chip='{self.chip_name}', "
            f"handle={self._handle}, pins={pin_summary})"
        )


if __name__ == "__main__":
    gpio_config = {
        "LED1": {
            "pin": "GPIO24",
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
            "pin": "GPIO23",
            "direction": "input",
            "enabled": True,
        },
    }

    with LGPIOController(gpio_config=gpio_config) as controller:
        # Via controller
        print(f"Initial LED1 : {controller.get_value('LED1')}")
        controller.set_value("LED1", 1)
        print(f"LED1 HIGH    : {controller.get_value('LED1')}")
        time.sleep(5)
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
