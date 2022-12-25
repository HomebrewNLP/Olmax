export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4  # faster malloc
export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=60000000000  # no numpy memory warnings
export TF_CPP_MIN_LOG_LEVEL=4   # no dataset warnings

export XRT_TPU_CONFIG="localservice;0;localhost:51011"

export JAX_ENABLE_X64=0  # allow fp64
export JAX_DEFAULT_DTYPE_BITS=32  # ..but don't enforce it

export WANDB_WATCH="false"  # workaround to wandb crashing and killing the whole run
export WANDB_START_METHOD="thread"

# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/xla.proto
export XLA_FLAGS="--xla_force_host_platform_device_count=1"  # We don't use TPU-CPU for ML
# export XLA_FLAGS="--xla_step_marker_location=1 $XLA_FLAGS"  # 0 = entry; 1 = outer while

/usr/bin/env python3 main.py "$@"
