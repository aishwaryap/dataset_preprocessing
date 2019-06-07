# Preprocessing VisualGenome dataset for various experiments

Much of this code has been written assuming that preprocessing will be distributed over multiple systems,
and you will need to write scripts to collect all steps of each preprocessing operation. For the HTCondor
system as used by UTCS, these scripts can be generated using create_condor_scripts.py
The documentation mentions where such scripts will be needed. I will try to automate this eventually.
Currently this only lists the preprocessing needed for the paper Padmakumar et. al., [Learning a Policy for Opportunistic Active Learning](http://www.cs.utexas.edu/users/ml/papers/padmakumar.emnlp18.pdf)
accepted at EMNLP 2018.

## Download dataset
```bash
DATASET_DIR=<path_to_store_dataset>
./download.sh $DATASET_DIR
```

## Create image lists for extracting VGG features
```bash
python create_image_lists.py \
    --dataset-dir=$DATASET_DIR
```
Image lists are created in ```$DATASET_DIR/image_lists```
Optionally adjust the number of images per list using ```--batch-size```

## Extract VGG features using a CaffeModel
For each image list file created in the previous step, run the following, with output files of the form
$DATASET_DIR/regions_vgg_features/<image list num>.csv
```bash
cd ../utils
python extract_vgg_features.py \
    --image-list-file=<image list file> \
    --output-file=<output file> \
    --prototxt-file=<model prototxt file> \
    --caffemodel-file=<CaffeModel file> \
    --restart-log=<a log file>
```

## Several serial preprocessing steps
```bash
python organize_region_graphs.py \
    --dataset-dir=$DATASET_DIR
python preprocess_descriptions.py \
    --dataset-dir=$DATASET_DIR \
    --preprocess-descriptions
python organize_attributes.py \
    --dataset-dir=$DATASET_DIR
    --create-contents-list \
    --make-region-contents-unique \
    --make-contents-list-unique
python filter_by_hypernyms.py \
    --dataset-dir=$DATASET_DIR \
    --get-selected-synsets
python filter_by_hypernyms.py \
    --dataset-dir=$DATASET_DIR \
    --filter-region-contents \
    --contents-file=region_objects_unique.csv
python filter_by_hypernyms.py \
    --dataset-dir=$DATASET_DIR \
    --filter-region-contents \
    --contents-file=region_attributes_unique.csv
python filter_by_hypernyms.py \
    --dataset-dir=$DATASET_DIR \
    --filter-region-contents \
    --contents-file=region_synsets_unique.csv
python filter_by_hypernyms.py \
    --dataset-dir=$DATASET_DIR \
    --filter-region-contents \
    --contents-file=region_descriptions.csv
python stats.py \
    --dataset-dir=$DATASET_DIR \
    --num-regions-per-content
mkdir $DATASET_DIR/classifiers
mkdir $DATASET_DIR/classifiers/data
mkdir $DATASET_DIR/classifiers/data/features
python create_classifier_data.py \
    --dataset-dir=$DATASET_DIR \
    --organize-labels-and-regions
```

## Organize data for active learning
The lists of train and test regions will now be available in ```$DATASET/classifiers/data/train_regions.txt```
and ```$DATASET/classifiers/data/train_regions.txt``` respectively. Choose a batch size of regions (65536 by default)
and compute the number of batches present in the train and test set.
For each batch (numbered starting at 0) in the train set, run
```bash
python create_classifier_data.py \
    --dataset-dir=$DATASET_DIR \
    --write-features \
    --in-train-set \
    --batch-num=<batch num> \
    [--batch-size=<batch size if not 65536> ]
```
For each batch (numbered starting at 0) in the test set, run
```bash
python create_classifier_data.py \
    --dataset-dir=$DATASET_DIR \
    --write-features \
    --batch-num=<batch num>
    [--batch-size=<batch size if not 65536> ]
```

Computing densities of each object is the messiest part because this needed too much memory. First we compute pairwise
cosine similarities of objects in batches. I needed to divide this into smaller sub-batches of size 8192
to actually complete. So the following needs to be iterated for each unique pair of batches i and j including i=j, and
for each sub batch (numbered from 0).
```bash
python compute_densities_and_neighbours.py \
    --dataset-dir=$DATASET_DIR \
    --process-batch-pair \
    --batch-num-i=<i> \
    --batch-num-j=<j> \
    [--batch-size=<batch size if not 65536> ] \
    [--sub-batch-size=<sub batch size if not 8192> ] \
    --sub-batch-num=<sub batch num>
    [--in-train-set when this is being computed for batches in the train set] \
    --num-nbrs=<Number of nearest neighbours needed per object>
```

Aggregate densities computed in the previous step once for each train and test batch
```bash
python compute_densities_and_neighbours.py \
    --dataset-dir=$DATASET_DIR \
    --aggregate-densities \
    --batch-num-i=<batch num> \
    [--batch-size=<batch size if not 65536> ] \
    [--in-train-set when this is being computed for batches in the train set]
```

Aggregate neighbours computed earlier once for each train and test batch
```bash
python compute_densities_and_neighbours.py \
    --dataset-dir=$DATASET_DIR \
    --aggregate-nbrs \
    --batch-num-i=<batch num> \
    [--batch-size=<batch size if not 65536> ] \
    [--in-train-set when this is being computed for batches in the train set]
    --num-nbrs=<Number of nearest neighbours needed per object>
```


