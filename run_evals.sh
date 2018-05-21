python test_net.py --dataset pascal_voc --net vgg16 --checksession 1 --checkepoch 6 --checkpoint 10021 --cuda --vis --load_dir ~/gripper2/outputs/orig_vgg16_pascal_voc/ --out_dir ~/gripper2/outputs/orig_vgg16_pascal_voc/eval

python test_new_net.py --dataset pascal_voc --net vgg16 --checksession 1 --checkepoch 6 --checkpoint 1 --cuda --vis --load_dir ~/gripper2/outputs/p100_vgg16_pascal_voc/ --out_dir ~/gripper2/outputs/p100_vgg16_pascal_voc/eval --cag



