export JAX_ENABLE_X64=1  # allow fp64
export JAX_DEFAULT_DTYPE_BITS=64  # ..and enforce it

# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/xla.proto
export XLA_FLAGS="--xla_force_host_platform_device_count=8"  # We don't use TPU-CPU for ML
# export XLA_FLAGS="--xla_step_marker_location=1 $XLA_FLAGS"  # 0 = entry; 1 = outer while
export PYTHONPATH=`pwd`

/usr/bin/env python3 -m pytest "$@"  # for example: `bash test.sh unittests/grad/norm.py`
