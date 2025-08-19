## Microfloat: Tiny, customizable floating formats with NumPy

This repo provides a small Python module to define and use tiny floating-point formats ("microfloats") using NumPy. You can:

- Define formats with custom sign/exponent/mantissa bit widths
- Pack/unpack values to a compact integer storage dtype (`uint8`/`uint16`/`uint32`/`uint64`)
- Store, index, slice, and broadcast microfloats as NumPy arrays
- Vectorize encode/decode across arrays
- Do arithmetic by converting to higher precision (e.g., `float32`), operating there, then repacking

NumPy does not support sub-byte bitfields in structured dtypes. We pack all fields into the smallest unsigned integer dtype that fits the total bit width, and provide helpers to view the sign/exponent/mantissa as separate integer arrays when desired.

### Install

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
```

### Quick start

```python
import numpy as np
from microfloat.format import MicroFloatFormat
from microfloat.ops import add, multiply

# 8-bit microfloat: 1 sign, 3 exponent, 4 mantissa (bias=3 by default)
mf8 = MicroFloatFormat(sign_bits=1, exponent_bits=3, mantissa_bits=4)

x = np.array([0.1, 0.25, 0.5, 1.0, -1.0, 10.0, np.inf, np.nan], dtype=np.float32)

packed = mf8.encode(x)          # uint8 array
decoded = mf8.decode(packed)    # float32 array

# Arithmetic: compute in float32, then repack
sum_packed = add(mf8, packed, packed)        # x + x → packed microfloats
prod_packed = multiply(mf8, packed, packed)  # x * x → packed microfloats

# Inspect fields
fields = mf8.view_fields(packed)
# fields is a structured array with integer fields: 'sign', 'exponent', 'mantissa'
```

### Defining formats

```python
# bfloat8-ish (not IEEE): 1/5/2 layout
bf8 = MicroFloatFormat(1, 5, 2)

# custom 12-bit: 1 sign, 5 exponent, 6 mantissa → stored in uint16
mf12 = MicroFloatFormat(1, 5, 6)
```

### Notes

- Arithmetic (+, *, etc.) does not work natively on the packed dtype. Convert to a higher-precision type (`float32`), operate, then repack.
- Rounding is ties-to-even when quantizing mantissa.
- Overflow saturates to signed infinities; NaNs are propagated.
- Subnormals can be enabled/disabled per format (defaults to enabled).

See `examples/demo.py` for a runnable demo.


