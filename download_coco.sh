
export COCO_PATH="/mnt/data/itr_dataset/dataset/coco/images"
mkdir $COCO_PATH
cd $COCO_PATH
echo $COCO_PATH
wget http://images.cocodataset.org/zips/train2014.zip 
wget http://images.cocodataset.org/zips/test2014.zip 
wget http://images.cocodataset.org/zips/val2014.zip 

unzip train2014.zip 
unzip val2014.zip 
unzip test2014.zip 

rm train2014.zip
rm val2014.zip
rm test2014.zip