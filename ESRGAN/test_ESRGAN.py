from esrgan import GeneratorRRDB
from loader import *
from torch.utils.data import DataLoader
import torch
from torch.autograd import Variable
import argparse
import os
from torchvision.utils import save_image
from torcheval.metrics.functional import peak_signal_noise_ratio
import numpy

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_root", type=str, default="../datasets/validation", required=False, help="Path to image")
parser.add_argument("--dataset_txt_path", type=str, default="../datasets/val_all.txt", required=False, help="Path to image")
parser.add_argument("--output_path", type=str, default="../result/esrgan/images/test/shaken_model", required=False, help="Path to image")
parser.add_argument("--checkpoint_model", type=str, default="../result/esrgan/saved_model/shaken_model/generator_199.pth", required=False, help="Path to checkpoint model")
parser.add_argument("--channels", type=int, default=3, help="Number of image channels", required=False)
parser.add_argument("--residual_blocks", type=int, default=23, help="Number of residual blocks in G", required=False)
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation", required=False)
parser.add_argument("--hr_height", type=int, default=256, help="high res. image height", required=False)
parser.add_argument("--hr_width", type=int, default=256, help="high res. image width", required=False)
opt = parser.parse_args()

os.makedirs(opt.output_path, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model and load model checkpoint
generator = GeneratorRRDB(opt.channels, filters=64, num_res_blocks=opt.residual_blocks).to(device)
generator.load_state_dict(torch.load(opt.checkpoint_model))
generator.eval()

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

dataloader = DataLoader(
    ImageDataset(root=opt.dataset_root, text_file_path = opt.dataset_txt_path, shape=(opt.hr_height, opt.hr_width)),
    batch_size=1,
    shuffle=True,
    num_workers=opt.n_cpu,
)

psnr_val = []

# Upsample image
with torch.no_grad():
    for i, imgs in enumerate(dataloader):

        batches_done = i

        # Configure model input
        imgs_lr = Variable(imgs["lr"].type(Tensor))
        imgs_hr = Variable(imgs["gt"].type(Tensor))

        # Generate a high resolution image from low resolution input
        gen_hr = generator(imgs_lr)

        psnr=peak_signal_noise_ratio(imgs_hr, gen_hr)
        psnr_val.append(psnr)
        print(
            # "[Batch %d/%d] [PSNR: %f] [SSIM: %f]"
            "[Batch %d/%d] [PSNR: %f] %s"
            % (
                i,
                len(dataloader),
                psnr,
                imgs["file_name"][0]
            )
        )
        # Save image
        save_image(denormalize(gen_hr).cpu(), opt.output_path+f"/sr-"+imgs["file_name"][0])

print("Average PSNR:",numpy.mean(psnr_val))