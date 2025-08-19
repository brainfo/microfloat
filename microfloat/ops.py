from __future__ import annotations

from typing import Callable

import numpy as np

from .format import MicroFloatFormat


def _binary_op(fmt: MicroFloatFormat, a_packed: np.ndarray, b_packed: np.ndarray, op: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> np.ndarray:
	a = fmt.decode(a_packed)
	b = fmt.decode(b_packed)
	res = op(a.astype(np.float32), b.astype(np.float32))
	return fmt.encode(res)


def add(fmt: MicroFloatFormat, a_packed: np.ndarray, b_packed: np.ndarray) -> np.ndarray:
	return _binary_op(fmt, a_packed, b_packed, np.add)


def subtract(fmt: MicroFloatFormat, a_packed: np.ndarray, b_packed: np.ndarray) -> np.ndarray:
	return _binary_op(fmt, a_packed, b_packed, np.subtract)


def multiply(fmt: MicroFloatFormat, a_packed: np.ndarray, b_packed: np.ndarray) -> np.ndarray:
	return _binary_op(fmt, a_packed, b_packed, np.multiply)


def divide(fmt: MicroFloatFormat, a_packed: np.ndarray, b_packed: np.ndarray) -> np.ndarray:
	return _binary_op(fmt, a_packed, b_packed, np.divide)


