HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_NCCL_HOME=/usr/local/nccl/ horovodrun -np 6 python 3c.train_maskrcnn.py
