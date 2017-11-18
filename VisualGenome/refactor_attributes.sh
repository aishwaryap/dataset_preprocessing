python organize_attributes.py \
    --dataset-dir=/scratch/cluster/aish/VisualGenome/ \
    --make-region-contents-unique \
    --make-contents-list-unique

python stats.py \
    --dataset-dir=/scratch/cluster/aish/VisualGenome/ \
    --num-regions-per-content \
    --above-threshold \
    --regions-with-common-content

python create_classifier_data.py \
    --dataset-dir=/scratch/cluster/aish/VisualGenome/ \
    --verbose \
    --organize-labels-and-regions

python create_condor_scripts.py \
    --dataset-dir=/scratch/cluster/aish/VisualGenome/ \
    --condor-dir=write_multilabels \
    --write-multilabels

python create_condor_scripts.py \
    --dataset-dir=/scratch/cluster/aish/VisualGenome/ \
    --condor-dir=write_individual_labels \
    --write-individual-labels

python create_condor_scripts.py \
    --dataset-dir=/scratch/cluster/aish/VisualGenome/ \
    --condor-dir=write_features \
    --write-features