import os

os.environ['LD_PRELOAD'] = "/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "4"
os.environ['XRT_TPU_CONFIG'] = "localservice;0;localhost:51011"
os.environ['XLA_FLAGS'] = "--xla_force_host_platform_device_count=48"
os.environ['TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD'] = "60000000000"

from src.main import main

if __name__ == '__main__':
    main()
