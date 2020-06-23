import gc
import sys
import math
from data_modules import *
import torch
import torch.optim as optim
import torch.nn as nn


def sigma_255_to_torch(sigma_255):
    return (sigma_255 / 255) / 0.5


def calc_padding(arch_opt):
    patch_w = 5 if arch_opt.rgb else 7
    bilinear_pad = 1
    averaging_pad = (patch_w - 1) // 2
    patch_w_scale_1 = 2 * patch_w - 1
    find_nn_pad = (patch_w_scale_1 - 1) // 2
    total_pad0 = patch_w + 13
    total_pad = averaging_pad + bilinear_pad + find_nn_pad + 14 * 2
    offs = total_pad - total_pad0

    return offs, total_pad


def tensor_to_ndarray_uint8(im_tensor):
    im_nparray = (im_tensor.permute(0, 2, 3, 1).clamp(-1, 1).squeeze(-1).detach().cpu().numpy() * 0.5 + 0.5) * 255.0
    im_nparray = (im_nparray + 0.5).astype(np.uint8)
    return im_nparray


def add_noise_to_image(image_c, sigma):
    image_n = image_c + sigma_255_to_torch(sigma) * torch.randn_like(image_c)
    return image_n


def get_image_params(image, patch_w, neigh_pad):
    im_params = dict()
    im_params['batches'] = image.shape[0]
    im_params['pixels_h'] = image.shape[2] - 2 * neigh_pad
    im_params['pixels_w'] = image.shape[3] - 2 * neigh_pad
    im_params['patches_h'] = im_params['pixels_h'] - (patch_w - 1)
    im_params['patches_w'] = im_params['pixels_w'] - (patch_w - 1)
    im_params['patches'] = im_params['patches_h'] * im_params['patches_w']
    im_params['pad_patches_h'] = image.shape[2] - (patch_w - 1)
    im_params['pad_patches_w'] = image.shape[3] - (patch_w - 1)
    im_params['pad_patches'] = im_params['pad_patches_h'] * im_params['pad_patches_w']

    return im_params


def crop_offset(in_image, row_offs, col_offs):
    if len(row_offs) == 1:
        row_offs += row_offs
    if len(col_offs) == 1:
        col_offs += col_offs
    if row_offs[1] > 0 and col_offs[1] > 0:
        out_image = in_image[..., row_offs[0]:-row_offs[1], col_offs[0]:-col_offs[-1]]
    elif row_offs[1] > 0 and col_offs[1] == 0:
        out_image = in_image[..., row_offs[0]:-row_offs[1], :]
    elif 0 == row_offs[1] and col_offs[1] > 0:
        out_image = in_image[..., :, col_offs[0]:-col_offs[1]]
    else:
        out_image = in_image
    return out_image


def apply_on_chuncks(func, x, max_chunk, out):
    for si in range(0, x.shape[1], max_chunk):
        ei = min(si + max_chunk, x.shape[1])
        out[:, si:ei, :, :] = func(x[:, si:ei, :, :])


def process_image(nl_denoiser, image_n, max_chunk):
    with torch.no_grad():
        image_dn = nl_denoiser(image_n, train=False, save_memory=True, max_chunk=max_chunk)
        image_dn = image_dn.clamp(-1, 1)
    return image_dn


def adapt_net(nl_denoiser, opt, total_pad, train_image_c):

    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(nl_denoiser.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)

    transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomVerticalFlip(),
                                    RandomTranspose(),
                                    transforms.ToTensor(),
                                    ShiftImageValues(),
                                    ])
    block_w_pad = opt.block_w + 2 * total_pad
    train_set = ImageDataSet(block_w=block_w_pad, images=tensor_to_ndarray_uint8(train_image_c),
                             transform=transform, stride=opt.dset_stride)
    train_set_loader = data.DataLoader(train_set, batch_size=opt.train_batch_size, shuffle=True, num_workers=0)
    batch_last_it = train_set_loader.dataset.__len__() // train_set_loader.batch_size - 1
    train_image_n = add_noise_to_image(train_image_c, opt.sigma)

    for epoch in range(opt.epoch_num):
        print('Training epoch {} of {}'.format(epoch + 1, opt.epoch_num))
        sys.stdout.flush()
        gc.collect()
        torch.cuda.empty_cache()
        if opt.cuda_retrain:
            if torch.cuda.is_available():
                nl_denoiser.cuda()
        else:
            nl_denoiser.cpu()
        device = next(nl_denoiser.parameters()).device
        train_it = enumerate(train_set_loader)
        for i, image_c in train_it:

            image_c = image_c.to(device=device)
            image_n = image_c + sigma_255_to_torch(opt.sigma) * torch.randn_like(image_c)

            optimizer.zero_grad()
            image_dn = nl_denoiser(image_n, train=True)

            total_pad = (image_c.shape[-1] - image_dn.shape[-1]) // 2
            image_ref = crop_offset(image_c, (total_pad,), (total_pad,))
            loss = torch.log10(criterion(image_dn, image_ref))
            assert not np.isnan(loss.item())
            loss.backward()
            optimizer.step()

            if i == batch_last_it and (epoch + 1) % opt.epochs_between_check == 0:
                gc.collect()
                torch.cuda.empty_cache()

                if opt.cuda_denoise:
                    if torch.cuda.is_available():
                        nl_denoiser.cuda()
                else:
                    nl_denoiser.cpu()
                device = next(nl_denoiser.parameters()).device
                train_image_dn = process_image(nl_denoiser, train_image_n.to(device), opt.max_chunk)
                train_image_dn = train_image_dn.clamp(-1, 1).cpu()
                train_psnr = -10 * math.log10(criterion(train_image_dn / 2, train_image_c / 2).item())

                print('Epoch {} of {} done, training PSNR = {:.2f}'.format(epoch + 1, opt.epoch_num, train_psnr))
                sys.stdout.flush()

                break

    return nl_denoiser
