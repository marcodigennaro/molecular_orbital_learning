import timeit

timeit.timeit('"-".join(str(n) for n in range(100))', number=10000)

for pkg in ['sys', 'torch', 'numpy', 'pandas']:
    print(pkg, timeit.timeit(f"import {pkg}"))