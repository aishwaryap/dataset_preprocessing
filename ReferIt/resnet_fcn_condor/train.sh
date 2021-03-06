universe = vanilla
Initialdir = /u/aish/Documents/Research/Code/dataset_preprocessing/ReferIt/resnet_fcn_bash
Executable = /lusr/bin/bash
Arguments = train.sh
+Group   = "GRAD"
+Project = "AI_ROBOTICS"
+ProjectDescription = "Extract ResNet FCN Features for ReferIt"
JobBatchName = "Extract ResNet FCN Features for ReferIt"
Requirements = TARGET.GPUSlot
getenv = True
request_GPUs = 1
+GPUJob = true
Log = /scratch/cluster/aish/ReferIt/condor_log/resnet_fcn/train.log
Error = /scratch/cluster/aish/ReferIt/condor_log/resnet_fcn/train.err
Output = /scratch/cluster/aish/ReferIt/condor_log/resnet_fcn/train.out
Notification = complete
Notify_user = aish@cs.utexas.edu
Queue 1