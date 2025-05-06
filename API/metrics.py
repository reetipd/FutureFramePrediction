import numpy as np
from skimage.metrics import structural_similarity as ssim

def MAE(pred, true):
    return np.mean(np.abs(pred-true),axis=(0,1)).sum()

def MSE(pred, true):
    return np.mean((pred-true)**2,axis=(0,1)).sum()

# cite the `PSNR` code from E3d-LSTM, Thanks!
# https://github.com/google/e3d_lstm/blob/master/src/trainer.py line 39-40
def PSNR(pred, true):
    mse = np.mean((np.uint8(pred * 255)-np.uint8(true * 255))**2)
    return 20 * np.log10(255) - 10 * np.log10(mse)

# def metric(pred, true, mean, std, return_ssim_psnr=False, clip_range=[0, 1]):
#     pred = pred*std + mean
#     true = true*std + mean
#     mae = MAE(pred, true)
#     mse = MSE(pred, true)

#     if return_ssim_psnr:
#         pred = np.maximum(pred, clip_range[0])
#         pred = np.minimum(pred, clip_range[1])
#         ssim, psnr = 0, 0
#         for b in range(pred.shape[0]):
#             for f in range(pred.shape[1]):
#                 ssim += cal_ssim(pred[b, f].swapaxes(0, 2), true[b, f].swapaxes(0, 2), multichannel=True)
#                 psnr += PSNR(pred[b, f], true[b, f])
#         ssim = ssim / (pred.shape[0] * pred.shape[1])
#         psnr = psnr / (pred.shape[0] * pred.shape[1])
#         return mse, mae, ssim, psnr
#     else:
#         return mse, mae


def metric(preds, trues, mean, std, verbose=False):
    """
    Compute metrics for validation: MSE, MAE, SSIM, PSNR.
    Args:
        preds: predicted frames
        trues: true frames
        mean: dataset mean
        std: dataset std
        verbose: whether to print verbose output
    Returns:
        mse, mae, ssim, psnr: calculated metrics
    """
    mse = np.mean((preds - trues) ** 2)
    mae = np.mean(np.abs(preds - trues))
    ssim_val = 0
    psnr_val = 0

    # Loop through batch and frames
    for b in range(len(preds)):
        for f in range(preds.shape[1]):
            pred = preds[b, f]
            true = trues[b, f]

            # Adaptive window size for SSIM
            win_size = min(pred.shape[0], pred.shape[1], 7) if min(pred.shape[0], pred.shape[1]) >= 7 else None

            ssim_val += ssim(pred.swapaxes(0, 2), true.swapaxes(0, 2), 
                             channel_axis=-1, win_size=win_size, data_range=1.0)

            # PSNR Calculation
            mse_value = np.mean((pred - true) ** 2)
            psnr_val += 20 * np.log10(1.0 / np.sqrt(mse_value + 1e-10))  # Avoid log(0)

    # Compute average SSIM and PSNR
    ssim_val /= (len(preds) * preds.shape[1])
    psnr_val /= (len(preds) * preds.shape[1])

    if verbose:
        print(f"SSIM: {ssim_val:.4f}, PSNR: {psnr_val:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}")

    return mse, mae, ssim_val, psnr_val