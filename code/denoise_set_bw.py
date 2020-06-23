from modules import *
from utils import *
import warnings
import os.path
import argparse
import matplotlib.pyplot as plt


def parse_input():
    parser = argparse.ArgumentParser()

    # noise parameters
    parser.add_argument('--sigma', type=int, default=15, help='noise sigma: 15, 25, 50')
    parser.add_argument('--seed', type=int, default=8, help='random seed')

    # path
    parser.add_argument('--in_path', type=str, default='images/BSD68/gray/')
    parser.add_argument('--out_path', type=str, default='output/')
    parser.add_argument('--save', action='store_true', help='save output image')

    # memory consumption
    parser.add_argument('--max_chunk', type=int, default=40000)

    # gpu
    parser.add_argument('--cuda', action='store_true', help='use CUDA during inference')

    # additional parameters
    parser.add_argument('--plot', action='store_true', help='plot the processed image')

    opt = parser.parse_args()

    assert 15 == opt.sigma or 25 == opt.sigma or 50 == opt.sigma, "supported sigma values: 15, 25, 50"

    return opt


def denoise_set_bw_func():
    arch_opt = ArchitectureOptions(rgb=False, small_network=False)
    opt = parse_input()
    sys.stdout = Logger('output/log.txt')
    pad_offs, _ = calc_padding(arch_opt)
    nl_denoiser = NonLocalDenoiser(pad_offs, arch_opt)
    criterion = nn.MSELoss(reduction='mean')

    if opt.cuda:
        if torch.cuda.is_available():
            nl_denoiser.cuda()
        else:
            warnings.warn("CUDA isn't supported")
    device = next(nl_denoiser.parameters()).device
    state_file_name0 = 'models/model_state_sigma_{}_bw.pt'.format(opt.sigma)
    assert os.path.isfile(state_file_name0)
    model_state0 = torch.load(state_file_name0)
    nl_denoiser.patch_denoise_net.load_state_dict(model_state0['state_dict'])

    axs = None
    if opt.plot:
        fig, axs = plt.subplots(1, 3)

    torch.manual_seed(opt.seed)
    image_names = sorted(os.listdir(opt.in_path))
    avg_psnr = 0
    for im_name in image_names:
        test_image_c = load_image_from_file(opt.in_path + im_name)
        test_image_n = add_noise_to_image(test_image_c, opt.sigma)
        test_image_dn = process_image(nl_denoiser, test_image_n.to(device), opt.max_chunk)

        test_image_dn = test_image_dn.clamp(-1, 1).cpu()
        psnr_dn = -10 * math.log10(criterion(test_image_dn / 2, test_image_c / 2).item())
        avg_psnr = avg_psnr + psnr_dn

        print('Denoising grayscale {} with sigma = {} done, output PSNR = {:.2f}'.format(im_name, opt.sigma, psnr_dn))
        sys.stdout.flush()

        if opt.plot:
            axs[0].imshow(tensor_to_ndarray_uint8(test_image_c).squeeze(), cmap='gray', vmin=0, vmax=255)
            axs[0].set_title('Clean {}'.format(im_name))
            axs[1].imshow(tensor_to_ndarray_uint8(test_image_n).squeeze(), cmap='gray', vmin=0, vmax=255)
            axs[1].set_title(r'Noisy with $\sigma$ = {}'.format(opt.sigma))
            axs[2].imshow(tensor_to_ndarray_uint8(test_image_dn).squeeze(), cmap='gray', vmin=0, vmax=255)
            axs[2].set_title('Denoised, PSNR = {:.2f}'.format(psnr_dn))
            plt.draw()
            plt.pause(1)

        if opt.save:
            out_name = opt.out_path + im_name[:-4] + '_s{}_out.png'.format(opt.sigma)
            imageio.imwrite(out_name, tensor_to_ndarray_uint8(test_image_dn).squeeze())

    avg_psnr = avg_psnr / len(image_names)
    print('\nDenoising a grayscale set with sigma = {} done, average PSNR = {:.2f}'.format(opt.sigma, avg_psnr))
    sys.stdout.flush()

    if opt.plot:
        plt.show()


if __name__ == '__main__':
    denoise_set_bw_func()
