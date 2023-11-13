import numpy as np
from scipy.optimize import curve_fit

def gauss_guess(x, y, sig_guess):
    norm = np.median(np.percentile(y, 50))
    w = np.mean(x)
    N_guess = np.max(y) - np.min(y)
    p0 = [norm, N_guess, w, sig_guess]
    return p0


def gaussian(x, *p):
    # A gaussian peak with:
    #   Constant Background          : p[0]
    #   Peak height above background : p[1]
    #   Central value                : p[2]
    #   Standard deviation           : p[3]
    return p[0] + p[1] * np.exp(-1. * (x - p[2]) ** 2 / (2. * p[3] ** 2))
    

def line_residuals(wave, flux, ivar, mask, line_list, noff=5, nfit_min=25, diff_err_max=0.1, los_err_max=0.1, diff_guess=0.0, los_guess=0.5, plot=False):
    # DETERMINE GAUSSIAN CENTER + WIDTHS OF ISOLATED LINES
    dwave, diff, diff_err, los, los_err = [], [], [], [], []
    for line in line_list['Wave']:
        wline = [line - noff, line + noff]
        mw = (wave - diff_guess > wline[0]) & (wave - diff_guess < wline[1]) & (flux > 0) & (mask > 0)
        n_badpix = np.sum(~mask[(wave - diff_guess > wline[0]) & (wave - diff_guess < wline[1])])
        if (np.sum(mw) > nfit_min) & (n_badpix < 1):
            p0 = gauss_guess(wave[mw] - diff_guess, flux[mw], los_guess)
            if len(np.array(diff)[(np.array(diff_err) < 0.1) & (np.array(diff_err) > 0)]) > 5:
                p0[2] += np.median(np.array(diff)[np.array(diff_err) < 0.1])
            try:
                p, pcov = curve_fit(
                    gaussian,
                    wave[mw] - diff_guess,
                    flux[mw],
                    sigma=1. / np.sqrt(ivar[mw]),
                    p0=p0,
                    bounds=(
                        [np.min(flux), 0, wline[0], 0.2],
                        [np.max(flux), 2 * np.max(flux), wline[1], 5],
                    ),
                )
                perr = np.sqrt(np.diag(pcov))
                d = p[2] - line
            except:
                p = p0
                p[2] = -99
                perr = p0
                d = -99
            if np.abs(d) == 5:
                p = p0
                p[2] = -99
                perr = p0
                d = -99
            if ~np.isfinite(perr[2]):
                perr[2] = 1000.0
            if plot:
                gfit = gaussian(wave[mw], *p)
                plt.figure(figsize=(8, 3))
                plt.plot(wave[mw] - diff_guess, gfit, 'g')
                plt.plot(wave[mw] - diff_guess, flux[mw])
                plt.title(f'lambda={line:0.2f} sigma={p[3]:0.2f} diff={d:0.3f} differr={perr[2]:0.3f}')
                plt.show()
            dwave = np.append(dwave, line)
            diff = np.append(diff, d)
            diff_err = np.append(diff_err, perr[2])
            los = np.append(los, p[3])
            los_err = np.append(los_err, perr[3])
    m = (diff_err < diff_err_max) & (diff_err > 0.0) & (los_err < los_err_max) & (los_err > 0.0)
    return dwave[m], diff[m] + diff_guess, diff_err[m], los[m], los_err[m]


def fit_line_poly(wlines, wdiff, wdiff_err, max_iter=10, deg=4):
    iteration = 0
    converged = False
    sigma_clip = np.zeros_like(wlines, dtype=bool)
    p = np.zeros(2)
    while not converged and (iteration < max_iter):
        z = np.polyfit(wlines[~sigma_clip], wdiff[~sigma_clip], w=1. / wdiff_err[~sigma_clip], deg=deg, cov=False)
        p = np.poly1d(z)
        resid = wdiff - p(wlines)
        std = np.std(resid[~sigma_clip])
        sigma_clip_new = np.abs(resid) > 3 * np.sqrt(std ** 2 + wdiff_err ** 2)
        converged = np.all(sigma_clip_new == sigma_clip)
        sigma_clip = sigma_clip_new
        iteration += 1
    if iteration == max_iter:
        raise RuntimeError("Max Iter Reached")
    return p, ~sigma_clip