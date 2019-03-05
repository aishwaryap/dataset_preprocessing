source /scratch/cluster/aish/lisa_env/bin/activate
export PYTHONPATH=/scratch/cluster/aish/lisa_env/lib/python3.5/site-packages
cd ../../utils
python3 extract_resnet_fcn_features_nocompress.py \
    --dataset-dir=/scratch/cluster/aish/ReferIt \
    --ckpt-path=/scratch/cluster/aish/tf_slim_models/resnet_v2_101.ckpt \
    --image-list-file=referit_trainval_imlist.txt \
    --output-file=referit_trainval_imlist_nocompress.hdf5 
deactivate
