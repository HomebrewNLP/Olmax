XLA_FLAGS="--xla_force_host_platform_device_count=48" TF_CPP_MIN_LOG_LEVEL=4 TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=60000000000 python3 model.py