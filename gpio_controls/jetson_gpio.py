"""
GPIO controller for NVIDIA Jetson devices using the Jetson.GPIO library.

Jetson.GPIO is used in **BOARD mode** (physical 40-pin header pin numbers).
The controller auto-detects the Jetson model and JetPack version at
startup (via /proc/device-tree/model and /etc/nv_tegra_release),
or they can be supplied explicitly.

Depending on the detected (or supplied) model + JetPack combination, a
per-device pin map is loaded so that the "GPIO<n>" syntax can resolve
Linux sysfs GPIO numbers to header pins.  If no map is available for the
combination, the controller still works — but only header-pin addressing
(plain integers like "12") is supported.

The public API is identical to GPIODController, RPiGPIOController, and
LGPIOController so all four backends can be used interchangeably.

Pin addressing
--------------
* "12" or 12   — physical 40-pin header pin number (BOARD mode).
* "GPIO398"        — Linux GPIO (sysfs) number, reverse-mapped to a
                         header pin via the active per-device pin map.

Jetson.GPIO API surface used
-----------------------------
    GPIO.setmode(GPIO.BOARD)
    GPIO.setwarnings(False)
    GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(pin, GPIO.IN)
    GPIO.output(pin, value)
    GPIO.input(pin)
    GPIO.cleanup([pins])
    GPIO.JETSON_INFO / GPIO.model  — for auto-detection
"""

import os
import re
import subprocess
import traceback

try:
    import Jetson.GPIO as GPIO
    _JETSON_GPIO_AVAILABLE = True
except ImportError:
    _JETSON_GPIO_AVAILABLE = False

from gpio_classes.gpio_dataclasses import (
    GPIOPin, Direction, State, GPIOMode,
    parse_direction as _parse_direction,
    parse_state as _parse_state,
    JNanoHeadToGPIOMap, JNanoGPIOToHeadMap,
    JONanoJP5HeadToGPIOMap, JONanoJP5GPIOToHeadMap,
    JONanoJP6HeadToGPIOMap, JONanoJP6GPIOToHeadMap,
)


# ---------------------------------------------------------------------------
# L4T → JetPack major-version mapping
# ---------------------------------------------------------------------------

_L4T_TO_JETPACK: dict[int, str] = {
    32: "4",   # L4T 32.x → JetPack 4.x
    35: "5",   # L4T 35.x → JetPack 5.x
    36: "6",   # L4T 36.x → JetPack 6.x
}


# ---------------------------------------------------------------------------
# Pin-map registry:  (model, jetpack_major) → (head_to_gpio, gpio_to_head)
# ---------------------------------------------------------------------------

_PIN_MAP_REGISTRY: dict[tuple[str, str], tuple[dict, dict]] = {
    ("JETSON_NANO", "4"):       (JNanoHeadToGPIOMap,       JNanoGPIOToHeadMap),
    ("JETSON_ORIN_NANO", "5"):  (JONanoJP5HeadToGPIOMap,   JONanoJP5GPIOToHeadMap),
    ("JETSON_ORIN_NANO", "6"):  (JONanoJP6HeadToGPIOMap,   JONanoJP6GPIOToHeadMap),
}


def _get_pin_maps(model: "str | None", jp: "str | None"):
    """
    Return (head_to_gpio, gpio_to_head) for the given *model* + *jp*
    combination, or (None, None) if no map is registered or the map
    is empty.
    """
    if model is None or jp is None:
        return None, None
    pair = _PIN_MAP_REGISTRY.get((model, str(jp)))
    if pair is None:
        return None, None
    h2g, g2h = pair
    if not h2g:          # empty map — treat as unavailable
        return None, None
    return h2g, g2h


# ---------------------------------------------------------------------------
# Auto-detection utilities
# ---------------------------------------------------------------------------

def _normalize_model(raw: str) -> str:
    """
    Normalise a free-form device-tree model string to a canonical
    identifier used as registry key.

    Examples::

        "NVIDIA Jetson Nano Developer Kit"      → "JETSON_NANO"
        "NVIDIA Jetson Orin Nano Developer Kit"  → "JETSON_ORIN_NANO"
        "NVIDIA Jetson AGX Orin Developer Kit"   → "JETSON_AGX_ORIN"
    """
    s = raw.upper()

    # Order matters — check more-specific patterns first
    if "ORIN" in s and "NANO" in s:
        return "JETSON_ORIN_NANO"
    if "ORIN" in s and "NX" in s:
        return "JETSON_ORIN_NX"
    if "AGX" in s and "ORIN" in s:
        return "JETSON_AGX_ORIN"
    if "ORIN" in s:
        return "JETSON_ORIN"

    if "XAVIER" in s and "NX" in s:
        return "JETSON_XAVIER_NX"
    if "AGX" in s and "XAVIER" in s:
        return "JETSON_AGX_XAVIER"
    if "XAVIER" in s:
        return "JETSON_XAVIER"

    if "TX2" in s and "NX" in s:
        return "JETSON_TX2_NX"
    if "TX2" in s:
        return "JETSON_TX2"
    if "TX1" in s:
        return "JETSON_TX1"

    if "NANO" in s:
        return "JETSON_NANO"

    # Fallback: stripped original
    return raw.strip()


def _detect_jetson_model() -> "str | None":
    """
    Auto-detect the Jetson model.

    Tries (in order):

    1. Jetson.GPIO model attribute (GPIO.model).
    2. /proc/device-tree/model file.

    Returns:
        Normalised model string (e.g. "JETSON_ORIN_NANO") or None.
    """
    # Method 1: Jetson.GPIO
    if _JETSON_GPIO_AVAILABLE:
        try:
            model = getattr(GPIO, "model", None)
            if model:
                return str(model).strip().upper()
        except Exception:
            pass

    # Method 2: device-tree
    dt = "/proc/device-tree/model"
    if os.path.exists(dt):
        try:
            with open(dt, "r") as f:
                raw = f.read().strip().rstrip("\x00")
            if raw:
                return _normalize_model(raw)
        except Exception:
            pass

    return None


def _detect_l4t_version() -> "tuple[int, int, int] | None":
    """
    Parse /etc/nv_tegra_release and return (major, minor, patch).

    Example first line::

        # R35 (release), REVISION: 3.1, GCID: 32827747, …
        → (35, 3, 1)

    Returns None if the file does not exist or cannot be parsed.
    """
    path = "/etc/nv_tegra_release"
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            line = f.readline().strip()
        m = re.match(
            r"#\s*R(\d+)\s*\(release\),\s*REVISION:\s*(\d+)\.(\d+)", line
        )
        if m:
            return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except Exception:
        pass
    return None


def _detect_jetpack_version() -> "str | None":
    """
    Determine the JetPack **major** version string ("4", "5", …).

    Tries (in order):

    1. L4T version from /etc/nv_tegra_release mapped via
       :data:_L4T_TO_JETPACK.
    2. dpkg -l nvidia-jetpack (parses the installed package version).

    Returns None if detection fails.
    """
    # Method 1: L4T → JetPack
    l4t = _detect_l4t_version()
    if l4t is not None:
        jp = _L4T_TO_JETPACK.get(l4t[0])
        if jp:
            return jp

    # Method 2: dpkg
    try:
        result = subprocess.run(
            ["dpkg", "-l", "nvidia-jetpack"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            for ln in result.stdout.splitlines():
                if "nvidia-jetpack" in ln and ln.startswith("ii"):
                    parts = ln.split()
                    if len(parts) >= 3:
                        return parts[2].split(".")[0]   # e.g. "5.1.2" → "5"
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# Jetson-specific pin parser
# ---------------------------------------------------------------------------

def _parse_jetson_pin(
    pin,
    gpio_to_head: "dict | None",
    head_to_gpio: "dict | None",
) -> int:
    """
    Resolve a pin identifier to a **BOARD header pin number**.

    * "12" or 12   → header pin 12 (validated against map if available).
    * "GPIO398"        → Linux GPIO 398 → look up header pin via
                             *gpio_to_head* map.

    Raises:
        ValueError / TypeError: If the identifier cannot be resolved.
    """
    if isinstance(pin, int):
        if head_to_gpio and pin not in head_to_gpio:
            raise ValueError(
                f"Header pin {pin} is not in the known GPIO pin map. "
                f"Valid header pins: {sorted(head_to_gpio.keys())}"
            )
        return pin

    pin_str = str(pin).strip()
    upper = pin_str.upper()

    if upper.startswith("GPIO"):
        linux_gpio = int(upper[4:])
        if gpio_to_head is None:
            raise ValueError(
                f"Cannot resolve '{pin}' — no GPIO-to-header-pin map is "
                "available for the current Jetson model / JetPack combination. "
                "Use header pin numbers (e.g., '7', '12') instead, or provide "
                "the jetson_model and jetpack_version arguments."
            )
        if linux_gpio not in gpio_to_head:
            raise ValueError(
                f"Linux GPIO {linux_gpio} is not in the known pin map. "
                f"Known Linux GPIO numbers: {sorted(gpio_to_head.keys())}"
            )
        return gpio_to_head[linux_gpio]

    # Plain integer string → header pin
    try:
        header_pin = int(pin_str)
    except ValueError:
        raise TypeError(
            f"Cannot interpret '{pin}' as a Jetson pin identifier. "
            "Use 'GPIO<n>' for a Linux GPIO number or a plain integer "
            "for a 40-pin header pin number."
        )

    if head_to_gpio and header_pin not in head_to_gpio:
        raise ValueError(
            f"Header pin {header_pin} is not in the known GPIO pin map. "
            f"Valid header pins: {sorted(head_to_gpio.keys())}"
        )
    return header_pin


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class JetsonGPIOController:
    """
    GPIO controller for NVIDIA Jetson boards using Jetson.GPIO
    in BOARD mode.

    Interchangeable with GPIODController, RPiGPIOController,
    and LGPIOController — same public API.
    """

    def __init__(
        self,
        gpio_config: dict = {},
        chip_name: str = "/dev/gpiochip0",
        jetson_model: "str | None" = None,
        jetpack_version: "str | int | None" = None,
    ):
        """
        Initialise the JetsonGPIOController.

        Args:
            gpio_config (dict): GPIO pin configuration dictionary.
                Example::

                    {
                        "LED1": {
                            "pin": "GPIO398",  # Linux GPIO → resolved to header pin via map
                            "direction": "output",
                            "initial": 0,
                            "enabled": True,
                        },
                        "LED2": {
                            "pin": "12",       # physical header pin 12 directly
                            "direction": "output",
                            "initial": 0,
                            "enabled": True,
                        },
                        "BUTTON1": {
                            "pin": "29",       # header pin 29
                            "direction": "input",
                            "enabled": True,
                        },
                    }

                Pin addressing rules:

                * "GPIO<n>" — Linux sysfs GPIO number *n*, reverse-mapped
                  to a header pin via the per-device pin map.
                * "<n>" or integer *n* — physical 40-pin header pin number
                  (used directly in BOARD mode).

                "pinmode" is accepted as an alias for "direction".

            chip_name (str): Kept for API compatibility with other controllers.
                Not used by Jetson.GPIO (which talks to the kernel driver
                directly via /dev/gpiochipN internally).

            jetson_model (str | None): Jetson model identifier, e.g.
                "JETSON_NANO", "JETSON_ORIN_NANO",
                "JETSON_AGX_ORIN".
                If None, auto-detected from GPIO.model or
                /proc/device-tree/model.

            jetpack_version (str | int | None): JetPack **major** version,
                e.g. "5" or 5.
                If None, auto-detected from /etc/nv_tegra_release
                (L4T version) or dpkg -l nvidia-jetpack.
        """
        if not _JETSON_GPIO_AVAILABLE:
            raise ImportError(
                "Jetson.GPIO is not installed. Install it with:\n"
                "  pip install Jetson.GPIO\n"
                "or, on JetPack systems, it is typically pre-installed."
            )

        self.chip_name = chip_name          # API compatibility
        self.gpio_config = gpio_config

        # ---- Detection / user-override ----
        self.jetson_model: "str | None" = (
            jetson_model.strip().upper() if jetson_model else _detect_jetson_model()
        )
        self.jetpack_version: "str | None" = (
            str(jetpack_version).strip() if jetpack_version else _detect_jetpack_version()
        )
        self.l4t_version = _detect_l4t_version()

        # ---- Pin maps ----
        self._head_to_gpio: "dict | None"
        self._gpio_to_head: "dict | None"
        self._head_to_gpio, self._gpio_to_head = _get_pin_maps(
            self.jetson_model, self.jetpack_version
        )

        # ---- Diagnostics ----
        l4t_str = (
            f"{self.l4t_version[0]}.{self.l4t_version[1]}.{self.l4t_version[2]}"
            if self.l4t_version else "unknown"
        )
        print(
            f"[JetsonGPIOController] Detected — "
            f"Model: {self.jetson_model or 'unknown'}, "
            f"JetPack: {self.jetpack_version or 'unknown'}, "
            f"L4T: {l4t_str}"
        )

        if self._gpio_to_head is None:
            if self.jetson_model and self.jetpack_version:
                print(
                    f"[JetsonGPIOController] WARNING: No pin map registered for "
                    f"{self.jetson_model} + JP{self.jetpack_version}. "
                    f"'GPIO<n>' syntax is unavailable — use header pin numbers."
                )
            else:
                print(
                    "[JetsonGPIOController] WARNING: Could not determine model "
                    "and/or JetPack version. 'GPIO<n>' syntax is unavailable — "
                    "use header pin numbers or supply jetson_model / "
                    "jetpack_version arguments."
                )
        else:
            print(
                f"[JetsonGPIOController] Pin map loaded: "
                f"{len(self._head_to_gpio)} GPIO-capable header pins."
            )

        # ---- Internal state ----
        # {pin_name: header_pin_number}  — used for Jetson.GPIO calls
        self._offsets: dict[str, int] = {}
        # {pin_name: GPIOPin}
        self._pins: dict[str, GPIOPin] = {}

        # ---- Initialise Jetson.GPIO ----
        try:
            GPIO.setmode(GPIO.BOARD)
            GPIO.setwarnings(False)
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialise Jetson.GPIO in BOARD mode: {e}"
            ) from e

        print("[JetsonGPIOController] Initialised Jetson.GPIO in BOARD mode.")
        self._initialize_pins()

    # ------------------------------------------------------------------
    # Internal initialisation
    # ------------------------------------------------------------------

    def _initialize_pins(self):
        for pin_name, pin_info in self.gpio_config.items():
            if not pin_info.get("enabled", True):
                continue
            try:
                pin_id = pin_info.get("pin", pin_name)
                board_pin = _parse_jetson_pin(
                    pin_id, self._gpio_to_head, self._head_to_gpio
                )
                direction = _parse_direction(
                    pin_info.get("direction", pin_info.get("pinmode", Direction.OUT))
                )
                initial = _parse_state(pin_info.get("initial", State.LOW))

                if direction == Direction.OUT:
                    GPIO.setup(board_pin, GPIO.OUT, initial=int(initial))
                else:
                    GPIO.setup(board_pin, GPIO.IN)

                self._offsets[pin_name] = board_pin

                # Resolve Linux GPIO number for metadata (if map available)
                linux_gpio = (
                    self._head_to_gpio[board_pin]
                    if self._head_to_gpio and board_pin in self._head_to_gpio
                    else board_pin
                )

                pin = GPIOPin(
                    name=pin_name,
                    gpio_number=linux_gpio,
                    direction=direction.value,
                    value=initial,
                    enabled=True,
                )
                # Inject live callbacks (default-arg captures pin_name per iteration)
                pin._setter = lambda v, _n=pin_name: self.set_value(_n, v)
                pin._getter = lambda _n=pin_name: self.get_value(_n)
                self._pins[pin_name] = pin

                print(
                    f"[JetsonGPIOController]   Configured '{pin_name}' "
                    f"header_pin={board_pin} linux_gpio={linux_gpio} "
                    f"dir={direction.value}"
                )

            except Exception:
                print(f"[JetsonGPIOController] ERROR configuring pin '{pin_name}':")
                traceback.print_exc()

    # ------------------------------------------------------------------
    # Public API  (mirrors GPIODController / RPiGPIOController exactly)
    # ------------------------------------------------------------------

    def setup_gpio(self):
        """(Re-)initialise all configured pins."""
        self.cleanup()
        GPIO.setmode(GPIO.BOARD)
        GPIO.setwarnings(False)
        self._initialize_pins()

    def get_pin(self, pin_name: str) -> GPIOPin:
        """
        Return the :class:GPIOPin for *pin_name*, with live
        .set() / .get() / .toggle() callbacks already injected::

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
        """Return all configured :class:GPIOPin objects (with live callbacks)."""
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
            State: State.LOW or State.HIGH.

        Raises:
            KeyError: If *pin_name* has not been configured.
        """
        if pin_name not in self._pins:
            raise KeyError(f"Pin '{pin_name}' is not configured.")

        state = _parse_state(GPIO.input(self._offsets[pin_name]))
        self._pins[pin_name].value = state  # keep GPIOPin in sync
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
        """Release all configured GPIO channels and reset Jetson.GPIO state."""
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
            name: f"pin{self._offsets.get(name, '?')}(gpio{p.gpio_number},{p.direction})"
            for name, p in self._pins.items()
        }
        return (
            f"JetsonGPIOController("
            f"model='{self.jetson_model}', "
            f"jp={self.jetpack_version}, "
            f"pins={pin_summary})"
        )


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Example for Jetson Orin Nano (JetPack 5) using Linux GPIO numbers
    # ------------------------------------------------------------------
    gpio_config = {
        "LED1": {
            "pin": "GPIO398",      # Linux GPIO 398 → header pin 12
            "direction": "output",
            "initial": 0,
            "enabled": True,
        },
        "LED2": {
            "pin": "40",           # header pin 40 directly
            "direction": "output",
            "initial": 0,
            "enabled": True,
        },
        "BUTTON1": {
            "pin": "29",           # header pin 29
            "direction": "input",
            "enabled": True,
        },
    }

    # Auto-detect model + JetPack:
    #   controller = JetsonGPIOController(gpio_config=gpio_config)
    #
    # Or specify explicitly:
    #   controller = JetsonGPIOController(
    #       gpio_config=gpio_config,
    #       jetson_model="JETSON_ORIN_NANO",
    #       jetpack_version="5",
    #   )

    with JetsonGPIOController(gpio_config=gpio_config) as controller:
        print(controller)

        # --- Via controller ---
        print(f"Initial LED1 : {controller.get_value('LED1')}")
        controller.set_value("LED1", 1)
        print(f"LED1 HIGH    : {controller.get_value('LED1')}")
        controller.toggle("LED1")
        print(f"LED1 toggled : {controller.get_value('LED1')}")

        # --- Via GPIOPin ---
        led1 = controller.get_pin("LED1")
        led1.set(State.HIGH)
        print(f"led1.get()   : {led1.get()}")
        led1.toggle()
        print(f"led1 toggled : {led1.get()}  value={led1.value}")

        btn = controller.get_pin("BUTTON1")
        print(f"BUTTON1      : {btn.get()}  (int: {int(btn.get())})")
