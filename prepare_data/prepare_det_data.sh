cp -r /bkai/data/trainset/images/ /bkai/data/det_train_data/
cp -r /bkai/data/trainset/labels/ /bkai/data/det_train_data/

python3 augment_det/augment_color.py \
--input_data_folder /bkai/data/trainset/ \
--output_data_folder /bkai/data/det_train_data/ \
--num_aug_imgs 5

python3 augment_det/augment_crop.py \
--input_data_folder /bkai/data/trainset/ \
--output_data_folder /bkai/data/det_train_data/ \
--num_aug_imgs 5

python3 augment_det/augment_cutpaste.py \
--input_cropped_bboxes_folder /bkai/data/rec_train_data/ \
--input_background_img_folder /bkai/data/synthtext/ \
--output_data_folder /bkai/data/det_train_data/ \
--num_aug_imgs 5

python3 generate_abcnet_data/bezier_generator.py \
--input_data_folder /bkai/data/det_train_data/ \
--output_data_folder /bkai/data/ABCnetV2/

python3 generate_abcnet_data/generate_abcnet_json.py \
--input_data_folder /bkai/data/det_train_data/ \
--output_data_folder /bkai/data/ABCnetV2/