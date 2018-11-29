# Preprocessing ReferIt dataset for various experiments

## Download dataset
```bash
DATASET_DIR=<path_to_download>
./download.sh $DATASET_DIR
```

## Preprocessing

Modify annotations to a format similar to the Kitchen dataset
```bash
python preprocess_dataset.py --dataset-dir=$DATASET_DIR
```

Create image lists to extract VGG features
```bash
python create_image_lists.py --dataset-dir=$DATASET_DIR
```

Extract VGG features
```bash
MODEL_DIR=<path to store caffe models>
cd ../utils
./download_caffe_models.sh $MODEL_DIR

mkdir $DATASET_DIR/vgg_features
python extract_vgg_features.py \
    --image-list-file=$DATASET_DIR/image_lists/referit_all_imlist.txt \
    --output-file=$DATASET_DIR/vgg_features/referit_all_imlist.csv \
    --prototxt-file=$MODEL_DIR/vgg7k.prototxt \
    --caffemodel-file=$MODEL_DIR/vgg7k.caffemodel \
    --restart-log=/dev/null

python extract_vgg_features.py \
    --image-list-file=$DATASET_DIR/image_lists/referit_edgeboxes_imlist.csv \
    --output-file=$DATASET_DIR/vgg_features/referit_edgeboxes_imlist.csv \
    --prototxt-file=$MODEL_DIR/vgg7k.prototxt \
    --caffemodel-file=$MODEL_DIR/vgg7k.caffemodel \
    --restart-log=/dev/null

cd ../ReferIt
```

## Download word2vec and GloVe vectors
Download pretrained word2vec vectors from [here](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing) into ```$MODEL_DIR/word2vec```
Run the following script to download GloVe vectors, and extract both.
```bash
cd ../utils
./download_gensim_word_vectors.sh
cd ../ReferIt
```