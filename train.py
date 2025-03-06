import os
import sys
import numpy as np
import torch
import utils
import logging
import argparse
from pathlib import Path

import torch.utils
import torch.backends.cudnn as cudnn

from PIL import Image
from torch.autograd import Variable
from model import Network
from multi_read_data import MemoryFriendlyLoader

parser = argparse.ArgumentParser("ruas")
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--seed', type=int, default=2, help='random seed')
args = parser.parse_args()

# EXP_path = r'./EXP\train/'
# if not os.path.isdir(EXP_path):
#     os.mkdir(EXP_path)
# model_path = EXP_path + '/model/'
# if not os.path.isdir(model_path):
#     os.mkdir(model_path)

# EXP_path = Path('EXP/train')
# EXP_path = Path('DarkFace_EXP/train')
EXP_path = Path('Exdark_EXP/train')
EXP_path.mkdir(parents=True, exist_ok=True)

# model_path = Path('RUAS_LOD')
# model_path = Path('RUAS_DarkFace')
model_path = Path('RUAS_Exdark')
model_path.mkdir(parents=True, exist_ok=True)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(EXP_path, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

torch.autograd.set_detect_anomaly(True)

def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    model = Network()
    model = model.cuda()

    # prepare DataLoader
    # train_low_data_names = r'D:\ZJA\data\LOL\OR\trainA/*.png'
    # train_low_data_names = 'LOD_train/*'
    # train_low_data_names = 'DarkFace_Train/*'
    train_low_data_names = 'Exdark_Train/*'
    # train_low_data_names = r'H:\image-enhance\UPE500\OR\trainA/*.png'
    TrainDataset = MemoryFriendlyLoader(img_dir=train_low_data_names, task='train')

    train_queue = torch.utils.data.DataLoader(
        TrainDataset, batch_size=args.batch_size,
        pin_memory=True, num_workers=0)
    total_step = 0
    while (total_step < 1000):
        input = next(iter(train_queue))
        total_step = total_step + 1
        model.train()
        # input = Variable(input, requires_grad=False).cuda()
        input = input.cuda(non_blocking=True)
        loss1, loss2, _ = model.optimizer(input, input, total_step)

        if total_step % args.report_freq == 0 and total_step != 0:
            logging.info('train %03d %f %f', total_step, loss1, loss2)
            utils.save(model, os.path.join(model_path, 'weights.pt'))


if __name__ == '__main__':
    main()
