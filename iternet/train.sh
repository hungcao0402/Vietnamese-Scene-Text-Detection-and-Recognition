CUDA_VISIBLE_DEVICES=0,1 python3 main.py --config=configs/pretrain_vm.yaml &&
CUDA_VISIBLE_DEVICES=0,1 python3 main.py --config=configs/pretrain_language_model.yaml &&
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 main.py --config=configs/train_iternet.yaml &&
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 main.py --config=configs/train.yaml --checkpoint ./workdir/train-iternet/best-train-iternet.pth