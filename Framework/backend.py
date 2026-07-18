import os

if os.environ.get("XP_BACKEND", "").lower() == "numpy":
    import numpy as xp
    BACKEND = "numpy"

else:
    try:
        import cupy as cp

        if cp.cuda.runtime.getDeviceCount() > 0:
            xp = cp
            BACKEND = "cupy"
        else:
            raise RuntimeError

    except Exception:
        import numpy as xp
        BACKEND = "numpy"
