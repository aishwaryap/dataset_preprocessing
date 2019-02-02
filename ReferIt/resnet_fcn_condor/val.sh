universe = vanilla
Initialdir = /u/aish/Documents/Research/Code/dataset_preprocessing/ReferIt/resnet_fcn_bash
Executable = /lusr/bin/bash
Arguments = val.sh
+Group   = "GRAD"
+Project = "AI_ROBOTICS"
+ProjectDescription = "Extract ResNet FCN Features for ReferIt"
JobBatchName = "Extract ResNet FCN Features for ReferIt"
Requirements = TARGET.GPUSlot
getenv = True
request_GPUs = 1
+GPUJob = true
Log = /scratch/cluster/aish/ReferIt/condor_log/resnet_fcn/val.log
Error = /scratch/cluster/aish/ReferIt/condor_log/resnet_fcn/val.err
Output = /scratch/cluster/aish/ReferIt/condor_log/resnet_fcn/val.out
Notification = complete
Notify_user = aish@cs.utexas.edu
Queue 1