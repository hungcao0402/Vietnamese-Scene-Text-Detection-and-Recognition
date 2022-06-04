python3 demo.py --config configs/train_iternet.yaml --cuda 0  \
      --checkpoint "./workdir/train-iternet-final/best-train-iternet-final.pth"  \
      --input /bkai/ABCnetV2/output/prediction  \
      --test_img /bkai/data/public_test_imgs && zip -jrm sub.zip submissions/predicted/*