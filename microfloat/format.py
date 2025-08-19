from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np


def _smallest_uint_dtype_for_bits(total_bits: int) -> np.dtype:
	if total_bits <= 8:
		return np.uint8
	elif total_bits <= 16:
		return np.uint16
	elif total_bits <= 32:
		return np.uint32
	elif total_bits <= 64:
		return np.uint64
	else:
		raise ValueError("Total bits must be <= 64")


@dataclass(frozen=True)
class MicroFloatFormat:
	"""Configurable tiny floating-point format packed into an unsigned integer dtype.

	Layout is [sign | exponent | mantissa] from most-significant to least-significant bits.

	- sign_bits: 0 or 1
	- exponent_bits: >= 1
	- mantissa_bits: >= 1
	- exponent_bias: if None, uses (2^(exponent_bits-1) - 1)
	- support_subnormals: whether to encode subnormals; if False, treat as zero

	The packed storage dtype is selected automatically based on total bits.
	"""

	sign_bits: int
	exponent_bits: int
	mantissa_bits: int
	exponent_bias: int | None = None
	support_subnormals: bool = True

	def __post_init__(self):
		if self.sign_bits not in (0, 1):
			raise ValueError("sign_bits must be 0 or 1")
		if self.exponent_bits < 1:
			raise ValueError("exponent_bits must be >= 1")
		if self.mantissa_bits < 1:
			raise ValueError("mantissa_bits must be >= 1")
		object.__setattr__(
			self,
			"total_bits",
			self.sign_bits + self.exponent_bits + self.mantissa_bits,
		)
		bias = self.exponent_bias
		if bias is None:
			bias = (1 << (self.exponent_bits - 1)) - 1
		object.__setattr__(self, "exponent_bias", bias)
		object.__setattr__(self, "storage_dtype", _smallest_uint_dtype_for_bits(self.total_bits))
		# Masks and shifts
		mantissa_mask = (1 << self.mantissa_bits) - 1
		exponent_mask = (1 << self.exponent_bits) - 1
		sign_shift = self.mantissa_bits + self.exponent_bits
		exponent_shift = self.mantissa_bits
		object.__setattr__(self, "_mantissa_mask", mantissa_mask)
		object.__setattr__(self, "_exponent_mask", exponent_mask)
		object.__setattr__(self, "_sign_shift", sign_shift)
		object.__setattr__(self, "_exponent_shift", exponent_shift)
		object.__setattr__(self, "_sign_mask", 1 << sign_shift if self.sign_bits == 1 else 0)
		object.__setattr__(self, "_exp_all_ones", exponent_mask)

	@property
	def dtype(self) -> np.dtype:
		return self.storage_dtype

	def view_fields(self, packed: np.ndarray) -> np.ndarray:
		"""Return a structured view exposing sign/exponent/mantissa as integer fields."""
		packed = np.asarray(packed, dtype=self.storage_dtype)
		sign = ((packed >> self._sign_shift) & 0x1).astype(self.storage_dtype) if self.sign_bits else np.zeros_like(packed, dtype=self.storage_dtype)
		exponent = ((packed >> self._exponent_shift) & self._exponent_mask).astype(self.storage_dtype)
		mantissa = (packed & self._mantissa_mask).astype(self.storage_dtype)
		dtype = np.dtype([
			("sign", self.storage_dtype),
			("exponent", self.storage_dtype),
			("mantissa", self.storage_dtype),
		])
		out = np.empty(packed.shape, dtype=dtype)
		out["sign"] = sign
		out["exponent"] = exponent
		out["mantissa"] = mantissa
		return out

	def encode(self, values: np.ndarray | float) -> np.ndarray:
		"""Vectorized encode from float32/float64 to packed unsigned integer dtype."""
		floats = np.asarray(values, dtype=np.float32)
		# handle NaN/Inf early
		nan_mask = np.isnan(floats)
		pos_inf_mask = np.isposinf(floats)
		neg_inf_mask = np.isneginf(floats)

		# Decompose float32: sign, exponent, mantissa
		bits = floats.view(np.uint32)
		sign = (bits >> 31) & 0x1
		exp = (bits >> 23) & 0xFF
		mant = bits & 0x7FFFFF

		# IEEE 754 float32 constants
		f32_bias = 127
		f32_mant_bits = 23

		# Convert exponent to unbiased E
		is_zero_like = (exp == 0) & (mant == 0)
		is_subnormal = (exp == 0) & (mant != 0)
		E = exp.astype(np.int32) - f32_bias

		# Normalize subnormals via frexp on magnitude for robust rounding
		mant_norm = mant.copy()
		E_norm = E.copy()
		mask_sub = is_subnormal
		if np.any(mask_sub):
			v = np.abs(floats[mask_sub].astype(np.float64))
			m, e = np.frexp(v)  # v = m * 2**e, m in [0.5, 1)
			E_norm[mask_sub] = e.astype(np.int32) - 1
			frac = (m * 2.0 - 1.0)
			mant_norm[mask_sub] = np.clip(
				np.rint(frac * (1 << f32_mant_bits)),
				0,
				(1 << f32_mant_bits) - 1,
			).astype(np.uint32)

		# Target mantissa rounding: ties-to-even
		target_mant_bits = self.mantissa_bits
		shift = f32_mant_bits - target_mant_bits
		if shift > 0:
			low_mask = (1 << shift) - 1
			q = mant_norm >> shift
			r = mant_norm & low_mask
			half = 1 << (shift - 1)
			should_round_up = (r > half) | ((r == half) & ((q & 1) == 1))
			mant_q = q + should_round_up.astype(np.uint32)
		elif shift == 0:
			mant_q = mant_norm
		else:
			mant_q = mant_norm << (-shift)

		# Handle mantissa overflow on rounding: if mant_q hits 2^p, increment exponent
		mant_overflow = mant_q >> target_mant_bits
		mant_q = mant_q & ((1 << target_mant_bits) - 1)
		E_out = E_norm + mant_overflow.astype(np.int32)

		# Map exponent with bias and handle limits
		exp_bias = self.exponent_bias
		exp_max = (1 << self.exponent_bits) - 2  # reserve all-ones for Inf/NaN
		E_max = exp_max - exp_bias
		E_min = 1 - exp_bias  # smallest normal exponent

		# Determine subnormal in target
		subnorm_mask = E_out < E_min
		underflow_mask = E_out < (E_min - target_mant_bits)  # too small even as subnormal

		mant_sub = np.zeros_like(mant_q, dtype=np.uint32)
		if self.support_subnormals:
			# For subnormals: set exponent to 0 and shift mantissa right by (E_min - 1 - E_out)
			shift_sub = np.clip((E_min - 1 - E_out).astype(np.int32), 0, 63)
			hidden_one = (1 << target_mant_bits)
			extended = (hidden_one | mant_q).astype(np.uint64)
			shift_sub_u = shift_sub.astype(np.uint64)
			mant_sub = (extended >> shift_sub_u) & np.uint64((1 << target_mant_bits) - 1)

		# Compose exponent field
		exp_field = np.clip(E_out + exp_bias, 0, exp_max).astype(np.int32)
		exp_field[subnorm_mask] = 0

		# Zeros
		is_zero = is_zero_like | underflow_mask
		mant_field = mant_q.astype(np.uint32)
		mant_field[subnorm_mask] = mant_sub[subnorm_mask].astype(np.uint32)
		mant_field[is_zero] = 0

		# Infinities and NaNs
		overflow_mask = E_out > E_max
		exp_field[overflow_mask] = self._exp_all_ones
		mant_field[overflow_mask] = 0
		sign_field = sign.astype(np.uint32)
		sign_field[pos_inf_mask | neg_inf_mask] = sign[pos_inf_mask | neg_inf_mask]

		# NaN: set exponent all-ones, mantissa non-zero; preserve sign=0
		if np.any(nan_mask):
			exp_field[nan_mask] = self._exp_all_ones
			mant_field[nan_mask] = 1
			sign_field[nan_mask] = 0

		# Compose packed bits
		sign_component = (((sign_field & 0x1).astype(np.uint64) << self._sign_shift) if self.sign_bits else np.uint64(0))
		packed = (
			sign_component
			| ((exp_field.astype(np.uint64) & self._exponent_mask) << self._exponent_shift)
			| (mant_field.astype(np.uint64) & self._mantissa_mask)
		).astype(self.storage_dtype)

		return packed

	def decode(self, packed: np.ndarray | int) -> np.ndarray:
		"""Vectorized decode from packed unsigned integer dtype to float32."""
		p = np.asarray(packed, dtype=self.storage_dtype)
		sign = ((p >> self._sign_shift) & 0x1).astype(np.int32) if self.sign_bits else np.zeros_like(p, dtype=np.int32)
		exp_field = ((p >> self._exponent_shift) & self._exponent_mask).astype(np.int32)
		mant_field = (p & self._mantissa_mask).astype(np.int32)

		exp_all_ones = self._exp_all_ones
		bias = self.exponent_bias
		mant_bits = self.mantissa_bits

		zero_mask = (exp_field == 0) & (mant_field == 0)
		sub_mask = (exp_field == 0) & (mant_field != 0)
		inf_mask = (exp_field == exp_all_ones) & (mant_field == 0)
		nan_mask = (exp_field == exp_all_ones) & (mant_field != 0)

		E = (exp_field - bias).astype(np.int32)
		frac = mant_field.astype(np.float64) / (1 << mant_bits)
		sig = 1.0 + frac

		values = np.empty(p.shape, dtype=np.float32)
		normal_mask = (exp_field != 0) & (exp_field != exp_all_ones)
		values[normal_mask] = (np.ldexp(sig, E)).astype(np.float32)[normal_mask]
		if np.any(sub_mask):
			values[sub_mask] = np.ldexp(
				mant_field[sub_mask].astype(np.float64) / (1 << mant_bits),
				1 - bias,
			).astype(np.float32)
		values[zero_mask] = 0.0
		values[inf_mask] = np.inf
		values[nan_mask] = np.nan

		if self.sign_bits == 1:
			values = np.copysign(values, 1.0 - 2.0 * sign.astype(np.float32))

		return values.astype(np.float32)

	def storage_info(self) -> Dict[str, int | np.dtype]:
		return {
			"total_bits": self.total_bits,
			"dtype": self.storage_dtype,
			"sign_bits": self.sign_bits,
			"exponent_bits": self.exponent_bits,
			"mantissa_bits": self.mantissa_bits,
			"exponent_bias": self.exponent_bias,
		}


