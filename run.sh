LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4  # faster malloc
TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=60000000000  # no numpy memory warnings
TF_CPP_MIN_LOG_LEVEL=4   # no dataset warnings

XRT_TPU_CONFIG="localservice;0;localhost:51011"

JAX_ENABLE_X64=1  # allow fp64
JAX_DEFAULT_DTYPE_BITS=32  # ..but don't enforce it

WANDB_WATCH="false"  # workaround to wandb crashing and killing the whole run
WANDB_START_METHOD="thread"

# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/xla.proto
XLA_FLAGS = "--xla_force_host_platform_device_count=1"  # We don't use TPU-CPU for ML
XLA_FLAGS = "--xla_step_marker_location=1 $XLA_FLAGS"  # 0 = entry; 1 = outer while

/usr/bin/env python3 main.py "$@"
