source /scratch/cluster/aish/gt_env/bin/activate
export PYTHONPATH=/scratch/cluster/aish/gt_env/lib/python3.5/site-packages
export PATH=/scratch/cluster/aish/cudnn/cuda:/opt/cuda-9.0/:$PATH
export LD_LIBRARY_PATH=/scratch/cluster/aish/cudnn/cuda/lib64:/opt/cuda-9.0/lib64/$LD_LIBRARY_PATH
export CUDA_HOME=/scratch/cluster/aish/cudnn/cuda/:/opt/cuda-9.0/
cd ../../utils
python3 extract_resnet_fcn_features.py \
    --dataset-dir=/scratch/cluster/aish/ReferIt \
    --ckpt-path=/scratch/cluster/aish/tf_slim_models/resnet_v2_101.ckpt \
    --image-list-file=referit_all_imlist.txt \
    --output-file=referit_all_imlist.hdf5 \
    > /scratch/cluster/aish/ReferIt/bash_log/resnet_fcn_features/all.out \
    2> /scratch/cluster/aish/ReferIt/bash_log/resnet_fcn_features/all.err
deactivate
