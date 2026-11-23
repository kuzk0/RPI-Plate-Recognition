"""Barrier control using Raspberry Pi 5's RP1 hardware PWM."""
import time

from rpi_hardware_pwm import HardwarePWM


class HardwareServo:
    """AngularServo-compatible adapter backed by RP1 hardware PWM."""

    def __init__(
        self,
        pwm_channel=1,
        pwm_chip=2,
        frequency_hz=50,
        min_pulse_width_ms=1.0,
        max_pulse_width_ms=2.0,
        initial_angle=180,
        closed_pulse_width_ms=None,
        opened_pulse_width_ms=None,
    ):
        if frequency_hz <= 0:
            raise ValueError("frequency_hz must be positive")
        if not 0 < min_pulse_width_ms < max_pulse_width_ms:
            raise ValueError("Invalid servo pulse-width range")
        self.frequency_hz = float(frequency_hz)
        self.period_ms = 1000.0 / self.frequency_hz
        if max_pulse_width_ms >= self.period_ms:
            raise ValueError("Servo pulse width must be shorter than the PWM period")

        self.min_pulse_width_ms = float(min_pulse_width_ms)
        self.max_pulse_width_ms = float(max_pulse_width_ms)
        self.closed_pulse_width_ms = closed_pulse_width_ms
        self.opened_pulse_width_ms = opened_pulse_width_ms
        for label, pulse in (("closed", closed_pulse_width_ms), ("opened", opened_pulse_width_ms)):
            if pulse is not None and not 0 < float(pulse) < self.period_ms:
                raise ValueError(f"{label} pulse width must be between 0 and PWM period")
        self._angle = max(0.0, min(180.0, float(initial_angle)))
        self._pwm = HardwarePWM(
            pwm_channel=int(pwm_channel),
            hz=int(frequency_hz),
            chip=int(pwm_chip),
        )
        self._pwm.start(self._angle_to_duty(self._angle))

    def _angle_to_duty(self, angle):
        angle = max(0.0, min(180.0, float(angle)))
        pulse_ms = self.min_pulse_width_ms + (
            self.max_pulse_width_ms - self.min_pulse_width_ms
        ) * angle / 180.0
        return pulse_ms / self.period_ms * 100.0

    def set_pulse_width_ms(self, pulse_width_ms):
        pulse_width_ms = float(pulse_width_ms)
        if not 0 < pulse_width_ms < self.period_ms:
            raise ValueError("Pulse width must be between 0 and PWM period")
        self._pwm.change_duty_cycle(pulse_width_ms / self.period_ms * 100.0)

    @property
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self, value):
        value = max(0.0, min(180.0, float(value)))
        self._pwm.change_duty_cycle(self._angle_to_duty(value))
        self._angle = value

    def close(self):
        self._pwm.stop()


class Barrier:
    def __init__(
        self,
        pwm_channel=1,
        pwm_chip=2,
        frequency_hz=50,
        min_pulse_width_ms=1.0,
        max_pulse_width_ms=2.0,
        closed_angle=180,
        opened_angle=90,
        open_duration=5.0,
        closed_pulse_width_ms=None,
        opened_pulse_width_ms=None,
    ):
        self.closed_angle = closed_angle
        self.opened_angle = opened_angle
        self.open_duration = float(open_duration)
        self.is_open = False
        self.close_at = None

        self.servo = HardwareServo(
            pwm_channel=pwm_channel,
            pwm_chip=pwm_chip,
            frequency_hz=frequency_hz,
            min_pulse_width_ms=min_pulse_width_ms,
            max_pulse_width_ms=max_pulse_width_ms,
            initial_angle=self.closed_angle,
            closed_pulse_width_ms=closed_pulse_width_ms,
            opened_pulse_width_ms=opened_pulse_width_ms,
        )
        self.close()

    def open(self):
        """Открывает шлагбаум и продлевает время его открытого состояния."""
        if self.servo.opened_pulse_width_ms is not None:
            self.servo.set_pulse_width_ms(self.servo.opened_pulse_width_ms)
        else:
            self.servo.angle = self.opened_angle
        self.is_open = True
        self.close_at = time.monotonic() + self.open_duration

    def close(self):
        """Закрывает шлагбаум."""
        if self.servo.closed_pulse_width_ms is not None:
            self.servo.set_pulse_width_ms(self.servo.closed_pulse_width_ms)
        else:
            self.servo.angle = self.closed_angle
        self.is_open = False
        self.close_at = None

    def keep_open(self, duration=None):
        """Продлевает время до автоматического закрытия."""
        duration = self.open_duration if duration is None else float(duration)
        if self.is_open:
            self.close_at = time.monotonic() + duration

    def update(self, vehicle_present=False):
        """Оставляет шлагбаум открытым при наличии автомобиля."""
        if not self.is_open:
            return
        if vehicle_present:
            self.close_at = time.monotonic() + self.open_duration
            return
        if self.close_at is not None and time.monotonic() >= self.close_at:
            self.close()

    def open_for(self, duration=None):
        """Неблокирующее открытие на заданное время."""
        duration = self.open_duration if duration is None else float(duration)
        self.open()
        self.close_at = time.monotonic() + duration

    def stop(self):
        try:
            self.close()
        finally:
            self.servo.close()
