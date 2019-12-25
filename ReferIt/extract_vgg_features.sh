universe = vanilla
Initialdir = /u/aish/Documents/Research/Code/dataset_preprocessing/utils
Executable = /lusr/bin/python
Arguments = extract_vgg_features.py \
                --image-list-file=/scratch/cluster/aish/ReferIt/image_lists/referit_all_imlist.txt \
                --output-file=/scratch/cluster/aish/ReferIt/vgg_features/referit_all_imlist.csv \
                --prototxt-file=/scratch/cluster/aish/CaffeModels/vgg7k.prototxt \
                --caffemodel-file=/scratch/cluster/aish/CaffeModels/vgg7k.caffemodel \
                --restart-log=/scratch/cluster/aish/ReferIt/condor_log/restart.txt
+Group   = "GRAD"
+Project = "AI_ROBOTICS"
+ProjectDescription = "Extract VGG features"
JobBatchName = "ReferIt VGG feature extraction"
Requirements = TARGET.GPUSlot
getenv = True
request_GPUs = 1
+GPUJob = true
Log = /scratch/cluster/aish/ReferIt/condor_log/extract_vgg_features/extract_vgg_features.log
Error = /scratch/cluster/aish/ReferIt/condor_log/extract_vgg_features/extract_vgg_features.err
Output = /scratch/cluster/aish/ReferIt/condor_log/extract_vgg_features/extract_vgg_features.out
Notification = complete
Notify_user = aish@cs.utexas.edu
Queue 1