mkdir -p train/JPEGImages
mkdir -p train/Annotations
mv AIZOO/train/*.jpg ./train/JPEGImages
mv AIZOO/train/*.xml ./train/Annotations
mkdir -p val/JPEGImages
mkdir -p val/Annotations
mv AIZOO/val/*.jpg ./val/JPEGImages
mv AIZOO/val/*.xml ./val/Annotations
mkdir -p test/JPEGImages
mkdir -p test/Annotations
cd val/JPEGImages
mv `ls | head -1000` ../../test/JPEGImages
cd ../Annotations
mv `ls | head -1000` ../../test/Annotations
ls test/JPEGImages > test/test.txt
ls train/JPEGImages > train/train.txt
ls val/JPEGImages > val/val.txt
