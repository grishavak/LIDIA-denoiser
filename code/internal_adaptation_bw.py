from modules import *
from utils import *
import warnings
import os.path
import argparse
import matplotlib.pyplot as plt


def parse_input():
    parser = argparse.ArgumentParser()

    # noise parameters
    parser.add_argument('--sigma', type=int, default=50, help='noise sigma: 15, 25, 50')
    parser.add_argument('--seed', type=int, default=8, help='random seed')

    # path
    parser.add_argument('--in_path', type=str, default='images/brick_house/gray/test1_bw.png')
    parser.add_argument('--out_path', type=str, default='output/')
    parser.add_argument('--save', action='store_true', help='save output image')

    # memory consumption
    parser.add_argument('--max_chunk', type=int, default=40000)

    # gpu
    parser.add_argument('--cuda_retrain', action='store_true', help='use CUDA during retraining')
    parser.add_argument('--cuda_denoise', action='store_true', help='use CUDA during inference')

    # additional parameters
    parser.add_argument('--plot', action='store_true', help='plot the processed image')

    # training parameters
    parser.add_argument('--block_w', type=int, default=64)
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--epoch_num', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dset_stride', type=int, default=32)
    parser.add_argument('--epochs_between_check', type=int, default=5)

    opt = parser.parse_args()

    assert 15 == opt.sigma or 25 == opt.sigma or 50 == opt.sigma, "supported sigma values: 15, 25, 50"

    return opt


def internal_adaptation_bw_func():
    arch_opt = ArchitectureOptions(rgb=False, small_network=False)
    opt = parse_input()
    sys.stdout = Logger('output/log.txt')
    pad_offs, total_pad = calc_padding(arch_opt)
    nl_denoiser = NonLocalDenoiser(pad_offs, arch_opt)
    criterion = nn.MSELoss(reduction='mean')

    if opt.cuda_denoise:
        if torch.cuda.is_available():
            nl_denoiser.cuda()
        else:
            warnings.warn("CUDA isn't supported")
    device = next(nl_denoiser.parameters()).device

    state_file_name0 = 'models/model_state_sigma_{}_bw.pt'.format(opt.sigma)
    assert os.path.isfile(state_file_name0)
    model_state0 = torch.load(state_file_name0)
    nl_denoiser.patch_denoise_net.load_state_dict(model_state0['state_dict'])

    torch.manual_seed(opt.seed)
    test_image_c = load_image_from_file(opt.in_path)
    test_image_n = add_noise_to_image(test_image_c, opt.sigma)
    test_image_dn1 = process_image(nl_denoiser, test_image_n.to(device), opt.max_chunk)

    test_image_dn1 = test_image_dn1.clamp(-1, 1).cpu()
    psnr_dn1 = -10 * math.log10(criterion(test_image_dn1 / 2, test_image_c / 2).item())

    print('Denoising a grayscale image with sigma = {} done.'
          ' Before adaptation output PSNR = {:.2f}'.format(opt.sigma, psnr_dn1))
    sys.stdout.flush()

    print('Adapting the network, please wait, this may take a while')
    sys.stdout.flush()
    nl_denoiser = adapt_net(nl_denoiser, opt, total_pad, test_image_dn1)

    if opt.cuda_denoise:
        if torch.cuda.is_available():
            nl_denoiser.cuda()
    else:
        nl_denoiser.cpu()
    device = next(nl_denoiser.parameters()).device
    test_image_dn2 = process_image(nl_denoiser, test_image_n.to(device), opt.max_chunk)

    test_image_dn2 = test_image_dn2.clamp(-1, 1).cpu()
    psnr_dn2 = -10 * math.log10(criterion(test_image_dn2 / 2, test_image_c / 2).item())
    print('Denoising a grayscale image with sigma = {} done.'
          ' After adaptation output PSNR = {:.2f}'.format(opt.sigma, psnr_dn2))
    sys.stdout.flush()

    if opt.plot:
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].imshow(tensor_to_ndarray_uint8(test_image_c).squeeze(), cmap='gray', vmin=0, vmax=255)
        axs[0, 0].set_title('Clean')
        axs[0, 1].imshow(tensor_to_ndarray_uint8(test_image_n).squeeze(), cmap='gray', vmin=0, vmax=255)
        axs[0, 1].set_title(r'Noisy with $\sigma$ = {}'.format(opt.sigma))
        axs[1, 0].imshow(tensor_to_ndarray_uint8(test_image_dn1).squeeze(), cmap='gray', vmin=0, vmax=255)
        axs[1, 0].set_title('Before Adaptation, PSNR = {:.2f}'.format(psnr_dn1))
        axs[1, 1].imshow(tensor_to_ndarray_uint8(test_image_dn2).squeeze(), cmap='gray', vmin=0, vmax=255)
        axs[1, 1].set_title('After Adaptation, PSNR = {:.2f}'.format(psnr_dn2))
        plt.draw()
        plt.pause(1)

    if opt.save:
        imageio.imwrite(opt.out_path + 'before_adaptation_bw.png', tensor_to_ndarray_uint8(test_image_dn1).squeeze())
        imageio.imwrite(opt.out_path + 'after_adaptation_bw.png', tensor_to_ndarray_uint8(test_image_dn2).squeeze())
        model_state = {'state_dict': nl_denoiser.patch_denoise_net.state_dict()}
        torch.save(model_state, opt.out_path + 'model_state_after_adaptation_sigma_{}_bw.pt'.format(opt.sigma))

    if opt.plot:
        plt.show()


if __name__ == '__main__':
    internal_adaptation_bw_func()
