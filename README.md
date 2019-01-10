I do a simple re-implement of siamese network structure. Some functions I referenced to [fangping's implementation](https://github.com/fangpin/siamese-pytorch)

### Requirement:
Pytorch 0.4.xx, Torchvision, numpy, PIL, Python 3.5,3.6 tested

### How to use:
```sh
usage: train.py [-h] [--train_path TRAIN_PATH] [--test_path TEST_PATH]
                [--way WAY] [--times TIMES] [--workers WORKERS]
                [--batch_size BATCH_SIZE] [--lr LR] [--max_iter MAX_ITER]
                [--save_path SAVE_PATH]

PyTorch One shot siamese training

optional arguments:
  -h, --help            show this help message and exit
  --train_path TRAIN_PATH
                        training folder
  --test_path TEST_PATH
                        path of testing folder
  --way WAY             how much way one-shot learning
  --times TIMES         number of samples to test accuracy
  --workers WORKERS     number of dataLoader workers
  --batch_size BATCH_SIZE
                        number of batch size
  --lr LR               learning rate
  --max_iter MAX_ITER   number of iterations before stopping
  --save_path SAVE_PATH
                        path to save model
```

