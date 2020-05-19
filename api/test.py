import os
import cv2
import sys
import numpy as np

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import torch
import torch.utils.data as data
import torch.backends.cudnn as cudnn

from torchvision import transforms
from core.data import get_segmentation_dataset
from core.model import get_segmentation_model
from core.utils.metric import SegmentationMetric
from core.utils.visualize import get_color_pallete
from core.utils.logger import setup_logger
from core.utils.distributed import synchronize, get_rank, make_data_sampler, make_batch_data_sampler

from scripts.train import parse_args


class Evaluator(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda")

        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

        # dataset and dataloader
        val_dataset = get_segmentation_dataset(root=args.dataset, split='val', mode='val', transform=input_transform)
        val_sampler = make_data_sampler(val_dataset, False, args.distributed)
        val_batch_sampler = make_batch_data_sampler(val_sampler, images_per_batch=1)
        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=args.workers,
                                          pin_memory=True)

        # create network
        print(args.model)
        print(args.classnum)
        print(args.aux)
        self.model = get_segmentation_model(model=args.model, classnum=args.classnum,
                                            aux=args.aux, pretrained=True, pretrained_base=False)
        if args.distributed:
            self.model = self.model.module
        self.model.to(self.device)

        #self.metric = SegmentationMetric(args.classnum)

    def eval(self):
        #self.metric.reset()
        self.model.eval()
        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model
        logger.info("Start validation, Total sample: {:d}".format(len(self.val_loader)))
        counter = 0
        for i, (image, target) in enumerate(self.val_loader):
            oimage = cv2.imread("../dataset/rgb/train/1.png", 1)
            oimage = cv2.cvtColor(oimage, cv2.COLOR_BGR2RGB)
            oimage = np.transpose(oimage, (2, 0, 1))
            oimage = oimage.astype(np.float32) / 255.
            oimage = oimage[np.newaxis, ...]
            oimage = torch.tensor(oimage).cuda()

            image = image.to(self.device)
            #print(image)
            #print(type(image), image.shape)
            #target = target.to(self.device)

            with torch.no_grad():
                import time
                time_start = time.time()
                for i in range(100):
                    outputs = model(oimage)
                print((time.time() - time_start) / 100.)

            #self.metric.update(outputs[0], target)
            #pixAcc, mIoU = self.metric.get()
            #logger.info("Sample: {:d}, validation pixAcc: {:.3f}, mIoU: {:.3f}".format(
            #    i + 1, pixAcc * 100, mIoU * 100))

            if self.args.save_pred:
                pred = torch.argmax(outputs[0], 1)
                pred = pred.cpu().data.numpy()
                predict = pred.squeeze(0)
                #predict = cv2.resize(predict, (oimage.shape[1], oimage.shape[0]), interpolation = cv2.INTER_NEAREST)
                mask = get_color_pallete(predict, self.args.dataset)
                #cv2.imshow("image", np.array(mask, dtype=np.uint8))
                #cv2.waitKey(2000)
                cv2.imwrite(os.path.join(outdir, 'test_mask_' + str(counter) + '.png'), np.array(predict * 20, dtype=np.uint8))
                mask.save(os.path.join(outdir, 'test_' + str(counter) + '.png'))
                counter += 1
        synchronize()


if __name__ == '__main__':
    args = parse_args()
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    if not args.no_cuda and torch.cuda.is_available():
        cudnn.benchmark = True
        args.device = "cuda"
    else:
        args.distributed = False
        args.device = "cpu"
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    # TODO: optim code
    args.save_pred = True
    if args.save_pred:
        outdir = '../runs/pred_pic/{}'.format(args.model)
        if not os.path.exists(outdir):
            os.makedirs(outdir)

    logger = setup_logger("semantic_segmentation", args.log_dir, get_rank(),
                          filename='{}_log.txt'.format(args.model), mode='a+')

    evaluator = Evaluator(args)
    evaluator.eval()
