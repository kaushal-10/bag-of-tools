"""
This script handles the GPIO controlling for the Raspberry Pi.
It utilizes the gpiod library and supports both gpiod v1 and v2 APIs,
auto-detecting which version is available at import time.

gpiod v1 API (< 2.0):
    - chip.get_line(offset) -> gpiod.line
    - line.request({'consumer': ..., 'type': LINE_REQ_DIR_OUT, 'default_vals': [val]})
    - line.set_value(val) / line.get_value()
    - line.release()

gpiod v2 API (>= 2.0):
    - gpiod.Chip(path)
    - gpiod.LineSettings(direction=..., output_value=...)
    - chip.request_lines(consumer=..., config={offset: settings}) -> LineRequest
    - request.set_values([val]) / request.get_values()
    - request.release()
"""


import os
import time
import traceback

import gpiod

from gpio_classes.gpio_dataclasses import (
    GPIOPin, Direction, State, GPIOMode,
    Pi5GPIOToHeadMap, Pi5HeadToGPIOMap,
    parse_pin_offset as _parse_pin_offset_base,
    parse_direction as _parse_direction,
    parse_state as _parse_state,
)

# ---------------------------------------------------------------------------
# Version detection
# ---------------------------------------------------------------------------
try:
    _GPIOD_MAJOR = int(gpiod.__version__.split(".")[0])
except Exception:
    # Older gpiod builds may not expose __version__; assume v1
    _GPIOD_MAJOR = 1

_GPIOD_V2 = _GPIOD_MAJOR >= 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_pin_offset(pin) -> int:
    """Resolve a pin identifier to a BCM GPIO offset using Pi5HeadToGPIOMap."""
    return _parse_pin_offset_base(pin, Pi5HeadToGPIOMap)


def _check_chip_available(chip_name: str) -> bool:
    """
    Return True if *chip_name* looks like a valid gpiochip device.
    Works for both gpiod v1 and v2.
    """
    if not os.path.exists(chip_name):
        return False
    # gpiod v2 exposes a dedicated helper; use it when available
    if hasattr(gpiod, "is_gpiochip_device"):
        return gpiod.is_gpiochip_device(chip_name)
    return True


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class GPIODController:
    def __init__(self, gpio_config: dict = {}, chip_name: str = "/dev/gpiochip0"):
        """
        Initialise the GPIODController.

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
                            "pin": "11",      # physical header pin 11 → GPIO17 via Pi5HeadToGPIOMap
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

                * "GPIO<n>" — uses GPIO offset *n* directly (BCM numbering).
                * "<n>" or integer *n* — treated as a **physical 40-pin header
                  pin number** and resolved to a GPIO offset via
                  :data:~gpio_classes.gpio_dataclasses.Pi5HeadToGPIOMap.

                The key "pinmode" is accepted as an alias for "direction"
                to stay compatible with RPi.GPIO-style configs.

            chip_name (str): Path to the gpiochip device (default /dev/gpiochip0).
        """
        assert "gpiochip" in chip_name, (
            f"Chip name must contain 'gpiochip'. Provided: {chip_name}"
        )
        assert _check_chip_available(chip_name), (
            f"Chip '{chip_name}' not found or is not a valid gpiochip device."
        )

        self.chip_name = chip_name
        self.gpio_config = gpio_config
        self.chip = None

        # v1: {pin_name: gpiod.line}
        # v2: {pin_name: gpiod.LineRequest}
        self._lines: dict = {}

        # Canonical pin metadata, keyed by pin_name
        self._pins: dict[str, GPIOPin] = {}

        print(f"[GPIODController] Detected gpiod v{_GPIOD_MAJOR} API.")
        self._initialize_chip()

    # ------------------------------------------------------------------
    # Internal initialisation
    # ------------------------------------------------------------------

    def _initialize_chip(self):
        """Open the chip and request lines for every enabled pin in the config."""
        if _GPIOD_V2:
            self._initialize_chip_v2()
        else:
            self._initialize_chip_v1()

    def _initialize_chip_v1(self):
        """gpiod v1 initialisation path."""
        self.chip = gpiod.Chip(self.chip_name)
        print(f"[GPIODController] Opened chip (v1): {self.chip_name}")

        for pin_name, pin_info in self.gpio_config.items():
            if not pin_info.get("enabled", True):
                continue
            try:
                pin_id = pin_info.get("pin", pin_name)
                offset = _parse_pin_offset(pin_id)
                direction = _parse_direction(
                    pin_info.get("direction", pin_info.get("pinmode", Direction.OUT))
                )
                initial = _parse_state(pin_info.get("initial", State.LOW))

                line = self.chip.get_line(offset)
                req_type = (
                    gpiod.LINE_REQ_DIR_OUT
                    if direction == Direction.OUT
                    else gpiod.LINE_REQ_DIR_IN
                )
                if direction == Direction.OUT:
                    line.request(
                        consumer="gpio_controller",
                        type=req_type,
                        default_val=initial,   # State is int, no .value needed
                    )
                else:
                    line.request(
                        consumer="gpio_controller",
                        type=req_type,
                    )

                self._lines[pin_name] = line
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
                    f"[GPIODController]   Configured '{pin_name}' "
                    f"offset={offset} dir={direction.value}"
                )

            except Exception:
                print(f"[GPIODController] ERROR configuring pin '{pin_name}':")
                traceback.print_exc()

    def _initialize_chip_v2(self):
        """gpiod v2 initialisation path."""
        self.chip = gpiod.Chip(self.chip_name)
        print(f"[GPIODController] Opened chip (v2): {self.chip_name}")

        for pin_name, pin_info in self.gpio_config.items():
            if not pin_info.get("enabled", True):
                continue
            try:
                pin_id = pin_info.get("pin", pin_name)
                offset = _parse_pin_offset(pin_id)
                direction = _parse_direction(
                    pin_info.get("direction", pin_info.get("pinmode", Direction.OUT))
                )
                initial = _parse_state(pin_info.get("initial", State.LOW))

                settings = gpiod.LineSettings()
                if direction == Direction.OUT:
                    settings.direction = gpiod.line.Direction.OUTPUT
                    settings.output_value = (
                        gpiod.line.Value.ACTIVE
                        if initial == State.HIGH
                        else gpiod.line.Value.INACTIVE
                    )
                else:
                    settings.direction = gpiod.line.Direction.INPUT

                request = self.chip.request_lines(
                    consumer="gpio_controller",
                    config={offset: settings},
                )

                self._lines[pin_name] = request
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
                    f"[GPIODController]   Configured '{pin_name}' "
                    f"offset={offset} dir={direction.value}"
                )

            except Exception:
                print(f"[GPIODController] ERROR configuring pin '{pin_name}':")
                traceback.print_exc()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def setup_gpio(self):
        """(Re-)initialise the chip and request all configured lines."""
        self.cleanup()
        self._initialize_chip()

    def get_pin(self, pin_name: str) -> GPIOPin:
        """
        Return the :class:`GPIOPin` for *pin_name*, with live
        ``.set()`` / ``.get()`` / ``.toggle()`` callbacks already injected::

            led = controller.get_pin("LED1")
            led.set(1)            # drive HIGH
            led.set(State.HIGH)   # same
            state = led.get()     # State.HIGH / State.LOW
            led.toggle()
            print(led.value)      # 0 or 1, synced automatically

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
            value (int | State): State.LOW / 0 for LOW,
                State.HIGH / 1 for HIGH.

        Raises:
            KeyError: If *pin_name* has not been configured.
            RuntimeError: If the pin is not configured as an output.
        """
        if pin_name not in self._lines:
            raise KeyError(f"Pin '{pin_name}' is not configured.")
        pin = self._pins[pin_name]
        if pin.direction != Direction.OUT.value:
            raise RuntimeError(f"Pin '{pin_name}' is not configured as an output.")

        state = _parse_state(value)

        if _GPIOD_V2:
            gpio_val = (
                gpiod.line.Value.ACTIVE if state == State.HIGH else gpiod.line.Value.INACTIVE
            )
            self._lines[pin_name].set_values([gpio_val])
        else:
            self._lines[pin_name].set_value(state)  # State is int

        # Keep GPIOPin in sync
        pin.value = state                            # State is int

    def get_value(self, pin_name: str) -> State:
        """
        Read the current value of a configured pin.

        Args:
            pin_name (str): Key from the gpio_config dictionary.

        Returns:
            State: State.LOW or State.HIGH.

        Raises:
            KeyError: If *pin_name* has not been configured.
        """
        if pin_name not in self._lines:
            raise KeyError(f"Pin '{pin_name}' is not configured.")

        if _GPIOD_V2:
            values = self._lines[pin_name].get_values()
            state = (
                State.HIGH if values[0] == gpiod.line.Value.ACTIVE else State.LOW
            )
        else:
            state = State.HIGH if self._lines[pin_name].get_value() else State.LOW

        # Keep GPIOPin in sync
        self._pins[pin_name].value = state  # State is int
        return state

    def write(self, pin_name: str, value: "int | State"):
        """Alias for :meth:set_value."""
        self.set_value(pin_name, value)

    def read(self, pin_name: str) -> State:
        """Alias for :meth:get_value."""
        return self.get_value(pin_name)

    def toggle(self, pin_name: str):
        """Toggle the output state of a configured output pin."""
        current = self.get_value(pin_name)
        self.set_value(pin_name, State.LOW if current == State.HIGH else State.HIGH)

    def cleanup(self):
        """Release all requested lines and close the chip."""
        for line_or_req in list(self._lines.values()):
            try:
                line_or_req.release()
            except Exception:
                pass
        self._lines.clear()
        self._pins.clear()

        if self.chip is not None:
            try:
                self.chip.close()
            except Exception:
                pass
            self.chip = None

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
            f"GPIODController(chip='{self.chip_name}', "
            f"gpiod_v{_GPIOD_MAJOR}, pins={pin_summary})"
        )

def blink(controller: GPIODController, pin_name: str, times: int = 5, interval: float = 0.5):
    """Example function to blink an output pin."""
    for _ in range(times):
        controller.toggle(pin_name)
        time.sleep(interval)

def monitor(controller: GPIODController, pin_name: str, duration: float = 10.0):
    """Example function to monitor and print the state of an input pin."""
    start_time = time.time()
    if duration <= 0: #infinite monitoring if duration is 0 or negative
        print(f"Monitoring '{pin_name}' indefinitely. Press Ctrl+C to stop.")
        while True:
            state = controller.get_value(pin_name)
            print(f"{pin_name} state: {state}")
            time.sleep(1.0)
    else:
        while time.time() - start_time < duration:
            state = controller.get_value(pin_name)
            print(f"{pin_name} state: {state}")
            time.sleep(1.0)
    
if __name__ == "__main__":
    # Example usage
    gpio_config = {
        "LED1": {
            "pin": "GPIO24",   # direct BCM GPIO offset → line 17
            "direction": "output",
            "initial": 0,
            "enabled": True,
        },
        "LED2": {
            "pin": "15",       # physical header pin 15 → GPIO22 via Pi5HeadToGPIOMap
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
    controller = GPIODController(gpio_config=gpio_config)

    # --- Via controller (original API) ---
    print(f"Initial state of LED1 : {controller.get_value('LED1')}")
    controller.set_value("LED1", 1)
    print(f"LED1 after set HIGH   : {controller.get_value('LED1')}")
    time.sleep(5)
    controller.toggle("LED1")
    print(f"LED1 after toggle     : {controller.get_value('LED1')}")

    # --- Via GPIOPin (pin-level API) ---
    led1 = controller.get_pin("LED1")
    print(f"\nled1 object           : {led1}")
    led1.set(State.HIGH)
    print(f"LED1 after led1.set(State.HIGH): {led1.get()}")
    led1.toggle()
    print(f"LED1 after led1.toggle()       : {led1.get()}")
    print(f"led1.value (synced)            : {led1.value}")

    # Iterate all pins
    for name, pin in controller.get_all_pins().items():
        print(f"{name}: GPIO{pin.gpio_number} dir={pin.direction} value={pin.value}")

    # Read input pin via pin object
    btn = controller.get_pin("BUTTON1")
    print(f"\nBUTTON1 state: {btn.get()}  (int: {int(btn.get())})")

    blink(controller, "LED1", times=10, interval=1.0)

    monitor(controller, "BUTTON1", duration=0)
