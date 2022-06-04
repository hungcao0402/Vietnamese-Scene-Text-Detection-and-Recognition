cd data && 
gdown --id 1UUQhNvzgpZy7zXBFQp0Qox-BBjunZ0ml -O vintext/vietnamese_original.zip && gdown --folder --id 1WNOk3EMSgawdbrHeLiO7N_WcapgmbdmV -O bkai && 
unzip -q bkai/train_gt.zip -d bkai && 
unzip -q bkai/train_imgs.zip -d bkai &&
unzip -q vintext/vietnamese_original.zip -d vintext &&

# synthtext
gdown 1-0eyK_tjmbU3lRLIR5cf9iZXso4gnKat -O synthtext/bg_img.zip  &&
unzip -q synthtext/bg_img.zip -d synthtext &&

cd .. &&
# synthtiger
gdown --id 11D_C3P18XKHgfw2GTmV4EJtEYQMACRh- -O prepare_data/augment_rec/synthtiger/resources.zip &&
unzip -q prepare_data/augment_rec/synthtiger/resources.zip -d prepare_data/augment_rec/synthtiger/ 

# abcnetv2 pretrained
gdown --id 1ROXLQVyMK0wYNcIhHhbwNTYIo9uQZdEN -O ABCnetV2/text_pretraining/model_v2_totaltext.pth

## our checkpoints