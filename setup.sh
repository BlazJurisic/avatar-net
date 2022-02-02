pip install -r requirements.txt
gdown --folder https://drive.google.com/drive/folders/1_7x93xwZMhCL-kLrz4B2iZ01Y8Q7SlTX
wget http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz -O vgg_weights.tar.gz
tar xzvf vgg_weights.tar.gz
mkdir data/eval