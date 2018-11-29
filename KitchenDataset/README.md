# Preprocessing Kitchen Dataset for various experiments

## Download the dataset
```bash
DATASET_DIR=<path_to_download>
./download.sh $DATASET_DIR
```

## Extract VGG features
```bash
python create_image_list.py \
    --images-dir=$DATASET_DIR/Kitchen/images/Kitchen
    --image-list-file=$DATASET_DIR/image_list.txt

python create_image_list.py \
    --images-dir=$DATASET_DIR/Kitchen/images/ImageNET \
    --image-list-file=$DATASET_DIR/imagenet_image_list.txt

MODEL_DIR=<path to store caffe models>
cd ../utils
./download_caffe_models.sh $MODEL_DIR

python extract_vgg_features.py \
    --image-list-file=$DATASET_DIR/image_list.txt \
    --output-file=$DATASET_DIR/vgg_features.csv \
    --prototxt-file=$MODEL_DIR/vgg7k.prototxt \
    --caffemodel-file=$MODEL_DIR/vgg7k.caffemodel \
    --restart-log=/dev/null

python extract_vgg_features.py \
    --image-list-file=$DATASET_DIR/imagenet_image_list.txt \
    --output-file=$DATASET_DIR/imagenet_vgg_features.csv \
    --prototxt-file=$MODEL_DIR/vgg7k.prototxt \
    --caffemodel-file=$MODEL_DIR/vgg7k.caffemodel \
    --restart-log=/dev/null

cd ../KitchenDataset
```

## Download word2vec vectors
Download pretrained word2vec vectors from [here](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing) into ```$MODEL_DIR/word2vec```
Run the following script to download Glove vectors, and extract both.
```bash
cd ../utils
./download_gensim_word_vectors.sh
cd ../KitchenDataset
```