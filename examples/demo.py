import numpy as np

from microfloat.format import MicroFloatFormat
from microfloat.ops import add, multiply


def main() -> None:
	mf8 = MicroFloatFormat(1, 3, 4)

	print("Format:", mf8.storage_info())

	x = np.array([0.0, -0.0, 0.1, 0.25, 0.5, 1.0, -1.0, 10.0, 1000.0, np.inf, -np.inf, np.nan], dtype=np.float32)
	packed = mf8.encode(x)
	decoded = mf8.decode(packed)

	print("Original:", x)
	print("Packed (uint):", packed)
	print("Decoded:", decoded)

	fields = mf8.view_fields(packed)
	print("Fields sample (first 5):")
	print(fields[:5])

	sum_packed = add(mf8, packed, packed)
	prod_packed = multiply(mf8, packed, packed)
	print("Sum decoded:", mf8.decode(sum_packed))
	print("Prod decoded:", mf8.decode(prod_packed))


if __name__ == "__main__":
	main()


