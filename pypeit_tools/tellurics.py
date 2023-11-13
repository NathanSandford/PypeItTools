from tqdm import tqdm
import numpy as np
import scipy.ndimage as scipynd
import astropy.units as u
from astropy.io import fits
from pypeit.core.wave import airtovac
import matplotlib as mpl
import matplotlib.pyplot as plt


def load_telluric_templates(telluric_dir):
    tfiles = sorted(list(telluric_dir.glob("telluric_0.02A*.fits")))
    n_files = len(tfiles)
    n_tell_pix = fits.open(tfiles[0])[1].data.shape[0]
    h2o = np.zeros(n_files)
    o2 = np.zeros(n_files)
    tell_wave = np.zeros((n_files, n_tell_pix))
    tell_flux = np.zeros((n_files, n_tell_pix))
    for i, tfile in tqdm(enumerate(tfiles), total=len(tfiles)):
        spl = tfile.stem.split("_")
        h2o[i] = float(spl[3])
        o2[i] = float(spl[5])
        hdu = fits.open(tfile)
        data = hdu[1].data
        tell_wave[i] = airtovac(np.array(data["wave"]).flatten() * u.AA).value
        tell_flux[i] = np.array(data["flux"]).flatten()
    return tfiles, h2o, o2, tell_wave, tell_flux


def create_tell_masks(data_wave):
    # USE THIS FOR TELLURIC FITTING
    b = [6855, 7167, 7580, 8160, 8925, 9400]
    r = [6912, 7320, 7690, 8300, 9120, 9550]
    data_mask1 = (data_wave > b[0]) & (data_wave < r[0])
    data_mask2 = (data_wave > b[1]) & (data_wave < r[1])
    data_mask3 = (data_wave > b[2]) & (data_wave < r[2])
    data_mask4 = (data_wave > b[3]) & (data_wave < r[3])
    data_mask5 = (data_wave > b[4]) & (data_wave < r[4])
    data_mask6 = (data_wave > b[5]) & (data_wave < r[5])
    tell_mask = (
        data_mask1 | data_mask2 | data_mask3 | data_mask4 | data_mask5 | data_mask6
    )
    return tell_mask


def create_cont_masks(data_wave):
    # USE THIS FOR CONTINUUM FITTING
    cmask1 = (data_wave > 6555) & (data_wave < 6567)
    cmask2 = (data_wave > 7590) & (data_wave < 7680)
    cmask3 = (data_wave > 8470) & (data_wave < 8660)
    cmaski = cmask1 | cmask2 | cmask3
    continuum_mask = np.invert(cmaski)
    return continuum_mask


def fit_syn_continuum_telluric(data_wave, data_flux, data_ivar, cmask, synth_flux, npoly):
    # FIT CONTINUUM -- for weights use 1/sigma
    ivar = data_ivar / synth_flux ** 2
    p = np.polyfit(
        data_wave[cmask],
        data_flux[cmask] / synth_flux[cmask],
        npoly,
        w=np.sqrt(ivar[cmask]),
    )
    fit = np.poly1d(p)
    d = data_flux / fit(data_wave)
    cmask2 = (d > np.percentile(d, 15)) & (d < np.percentile(d, 99))
    p = np.polyfit(
        data_wave[cmask2],
        data_flux[cmask2] / synth_flux[cmask2],
        npoly,
        w=np.sqrt(ivar[cmask2]),
    )
    fit = np.poly1d(p)
    # ADD CONTNUMM TO SYNTHETIC SPECTRUM
    continuum_syn_flux = fit(data_wave) * synth_flux
    return p, continuum_syn_flux


def telluric_chi2_min(
    twave, conv_tell, data_wave, data_flux, data_ivar, chi2_mask, cfit, wshift, show_plots=False,
):
    shift_wave = twave[:, np.newaxis] + wshift * 0.02
    conv_shift_tell = np.apply_along_axis(lambda x: np.interp(data_wave, x, conv_tell), axis=0, arr=shift_wave)
    # FIT CONTINUUM
    final_fit = np.polyval(cfit, data_wave[:, np.newaxis]) * conv_shift_tell
    # CALCUALTE CHI2
    chi2 = np.sum(
        (data_flux[chi2_mask, np.newaxis] - final_fit[chi2_mask, :])**2 * data_ivar[chi2_mask, np.newaxis], axis=0
    ) / (np.sum(chi2_mask) - 3)
    # NOW CALCULATE BEST CHI2
    n = np.argmin(chi2)
    min_w = wshift[n]
    min_w_chi2 = chi2[n]
    if show_plots:
        plt.scatter(wshift, chi2)
        plt.axvline(min_w)
        plt.xlabel('w [km/s]')
        plt.ylabel('Chi2')
        plt.show()
    return min_w, min_w_chi2


def telluric_marginalize_w(
    tell_wave, tell_flux, data_wave, data_flux, data_ivar, wmin, wmax, vstep, losvd_pix, cont_mask, chi2_mask, npoly=5, show_plots=False,
):
    # TRIM MODEL TO REDUCE COMPUTATION TIME
    mt = (tell_wave > wmin - 10) & (tell_wave < wmax + 10)
    tell_wave_trim = tell_wave[mt]
    tell_flux_trim = tell_flux[mt]
    conv_tell = scipynd.gaussian_filter1d(tell_flux_trim, losvd_pix, truncate=3)
    # FIT CONTINUUM OUTSIDE LOOP TO SAVE TIME
    tmp_flux = np.interp(data_wave, tell_wave_trim, conv_tell)
    cont_fit, _ = fit_syn_continuum_telluric(
        data_wave, data_flux, data_ivar, cont_mask, tmp_flux, npoly,
    )
    # SEARCH OVER TELLURIC SHIFT RANGE
    wshift = np.arange(-500, 500, vstep)
    min_w, min_w_chi2 = telluric_chi2_min(
            tell_wave_trim,
            conv_tell,
            data_wave,
            data_flux,
            data_ivar,
            chi2_mask,
            cont_fit,
            wshift,
            show_plots=show_plots,
        )
    return min_w, min_w_chi2


def solve_for_y(poly_coeffs, y):
    pc = poly_coeffs.copy()
    pc[-1] -= y
    return np.roots(pc)


def chi_interp_1d(chi, x, y):
    best_chi_idx = np.argmin(chi)
    best_chi = chi[best_chi_idx]
    best_y = y[best_chi_idx]
    # FIND X MIN in a 2D array
    matching_y = y == best_y
    x_matching_y = x[matching_y]
    chi_matching_y = chi[matching_y]
    sorted_idx = np.argsort(chi_matching_y)
    sorted_x = x_matching_y[sorted_idx]
    sorted_chi = chi_matching_y[sorted_idx]
    p_x = np.polyfit(sorted_x[:3], sorted_chi[:3], 2)
    r_x = np.roots(p_x)
    err_x = -99
    r = 0
    if 0 <= best_chi < 30:
        err = solve_for_y(p_x, best_chi + 0.1)
        err_x = (err[0] - err[1]) / 2.0
        r = r_x.real[0]
    return r, err_x


def generate_single_telluric(tell_wave, tell_flux, data_wave, data_flux, data_ivar, w, lsp, cont_mask, npoly=5):
    losvd_pix = lsp / 0.02
    conv_tell = scipynd.gaussian_filter1d(tell_flux, losvd_pix)
    shift_wave = tell_wave + w * 0.02
    conv_shift_tell = np.interp(data_wave, shift_wave, conv_tell)
    _, model = fit_syn_continuum_telluric(
        data_wave, data_flux, data_ivar, cont_mask, conv_shift_tell, npoly,
    )
    return model


def projected_chi_plot(x, y, z):
    unq_a = np.unique(x)
    unq_b = np.unique(y)
    aa, bb, cc = [], [], []
    for a in unq_a:
        for b in unq_b:
            m = (x == a) & (y == b)
            if np.sum(m) > 0:
                cc = np.append(cc, np.min(z[m]))
                aa = np.append(aa, a)
                bb = np.append(bb, b)
    return aa, bb, cc


def qa_telluric_plot1(model, specobj, tell_mask, o2, h2o, tmp_chi):
    wave=specobj.OPT_WAVE
    flux=specobj.OPT_COUNTS
    # PLOT CHI2 GRID AND BEST FIT
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(20, 5), gridspec_kw={"width_ratios": [1, 3]}
    )
    n = np.argmin(tmp_chi)
    vmn = np.log(np.min(tmp_chi))
    vmx = np.log(vmn + np.percentile(tmp_chi, 25))
    # CATCH PROBLEM CHI2 values
    if vmx <= vmn:
        vmx = vmn + 0.5
    aa, bb, cc = projected_chi_plot(o2, h2o, tmp_chi)
    ax1.scatter(aa, bb, c=np.log(cc), vmin=vmn, vmax=vmx, cmap="cool")
    ax1.plot(o2[n], h2o[n], "o", mfc="none", mec="k", ms=15)
    ax1.set_ylabel("H2O")
    ax1.set_xlabel("O2")
    ax1.set_title(f"H2O={h2o[n]:0.1f}, O2={o2[n]:0.1f}")
    # MAKE COLORBAR
    v1 = np.linspace(vmn, vmx, 8, endpoint=True)
    cax, _ = mpl.colorbar.make_axes(ax1, ticks=v1)
    normalize = mpl.colors.Normalize(vmin=vmn, vmax=vmx)
    cbar = mpl.colorbar.ColorbarBase(
        cax, norm=normalize, cmap=mpl.cm.cool
    )
    positions = v1
    labels = [f"{np.exp(i):0.1f}" for i in v1]
    cbar.ax.yaxis.set_major_locator(mpl.ticker.FixedLocator(positions))
    cbar.ax.yaxis.set_major_formatter(mpl.ticker.FixedFormatter(labels))
    cbar.ax.set_ylabel("chi2")
    ax2.plot(wave, flux, linewidth=0.9)
    ax2.plot(wave, model, "k", label="model", linewidth=0.8)
    ax2.plot(
        wave[tell_mask],
        model[tell_mask],
        "r",
        label="telluric fitted region",
        linewidth=0.8,
        alpha=0.8,
    )
    ax2.set_xlim(
        np.min(specobj.OPT_WAVE[specobj.OPT_MASK]),
        np.max(specobj.OPT_WAVE[specobj.OPT_MASK])
    )
    ax2.set_ylim(
        np.max([0, np.min(flux)-50]),
        1.1 * np.quantile(flux, 0.95)
    )
    plt.show()


def calc_exp_tell(bo2, berr_o2, bh2o, berr_h2o, bchi2, bw):
    # MASK OUT BAD DATA
    # *****************
    m1 = (
        #(bchi2 < 35)
        #& (bchi2 > 0)
        (bh2o > 1)
        & (bh2o < 120)
        & (bo2 > 0.6)
        & (bo2 < 3.0)
        & (berr_h2o > 0.01)
        & (berr_o2 > 0.01)
        #& (np.abs(bw) < 50)
    )
    # REJECT OUTLIERS (MUST BE A BETTER WAY TO DO THIS!)
    std = np.std(bh2o[m1])
    md = np.median(bh2o[m1])
    std2 = np.std(bo2[m1])
    md2 = np.median(bo2[m1])
    m = (
        (bchi2 < 35)
        & (bchi2 > 0)
        & (bh2o > 1)
        & (bh2o < 120)
        & (bo2 > 0.6)
        & (bo2 < 3.0)
        & (berr_h2o > 0.01)
        & (berr_o2 > 0.01)
        #& (np.abs(bw) < 50)
        & (bh2o > md - 3 * std)
        & (bh2o < md + 3 * std)
        & (bo2 > md2 - 3 * std2)
        & (bo2 < md2 + 3 * std2)
    )
    # FIX PROBLEM WITH SMALL H2O VALUES
    mh20 = bh2o < 10
    berr_h2o[mh20] = (
        berr_h2o[mh20] + 10
    )
    good_h2o = bh2o[m1]
    good_eh2o = berr_h2o[m1]
    good_o2 = bo2[m1]
    good_eo2 = berr_o2[m1]
    # REMOVE OUTLIERS
    mh, mo = 0, 0
    if np.size(good_h2o) > 2:
        mh = (good_h2o > np.percentile(good_h2o, 10)) & (
            good_h2o < np.percentile(good_h2o, 90)
        )
    if np.size(good_o2) > 2:
        mo = (good_o2 > np.percentile(good_o2, 10)) & (
            good_o2 < np.percentile(good_o2, 90)
        )
    # DETERMINE FINAL VALUES BASED ON WEIGHTED MEANS
    final_h2o = np.average(good_h2o[mh], weights=1.0 / good_eh2o[mh] ** 2)
    final_o2 = np.average(good_o2[mo], weights=1.0 / good_eo2[mo] ** 2)
    if final_h2o > 100:
        final_h2o = 100
    # ROUND TO FINEST GRID
    round_h2o = 2.0 * round(final_h2o / 2)
    round_o2 = 0.05 * round(final_o2 / 0.05)
    tfine = f"telluric_0.02A_h2o_{int(round_h2o)}_o2_{round_o2:2.2f}.fits"
    return round_o2, round_h2o, m, tfine


def get_o2_nodata(airmass):
    # USING FIT TO ALL O2 DATA, DETERMINE BASED ON AIRMASS
    # FIT IS DONE IN NOTEBOOK:  dmost/figure_telluric_all_O2fit
    m = 0.928209438
    b = 0.07763335
    o2 = m * airmass + b
    return o2


def qa_telluric_plot2(s2n, final_o2, bo2, berr_o2, final_h2o, bh2o, berr_h2o, m, airmass, show_plots=True):
    fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(2, 2, figsize=(16, 9))
    # PLOT H20-- zoom in and all data
    ax1.plot(s2n, bh2o, ".")
    ax1.plot(s2n[m], bh2o[m], "r.")
    ax1.errorbar(
        s2n[m],
        bh2o[m],
        yerr=berr_h2o[m],
        fmt=".r",
        ecolor="grey",
    )
    ax1.axhline(final_h2o)
    ax1.set_title(f"H2O = {final_h2o:0.3f}")
    ax1.set_ylim(0, 105)
    ax3.plot(s2n, bh2o, ".")
    ax3.plot(s2n[m], bh2o[m], "r.")
    ax3.errorbar(
        s2n[m],
        bh2o[m],
        yerr=berr_h2o[m],
        fmt=".r",
        ecolor="grey",
    )
    ax3.axhline(final_h2o)
    ax3.set_title(f"Full data range: H2O = {final_h2o:0.3f}")
    # PLOT OXYGEN-- zoom in on all data
    ax2.plot(s2n, bo2, ".")
    ax2.errorbar(
        s2n[m],
        bo2[m],
        yerr=np.abs(berr_o2[m]),
        fmt=".r",
        ecolor="grey",
    )
    airmass_fit = f"Airmass fit value {get_o2_nodata(np.mean(airmass)):0.3f}"
    ax2.set_title(f"O2 = {final_o2:0.3f}")
    ax2.axhline(final_o2, label="Final value")
    ax2.axhline(
        get_o2_nodata(np.mean(airmass)), label=airmass_fit, c="k", ls="--", lw=0.8
    )
    ax2.set_ylim(0.6, 2.05)
    ax2.legend(loc=4)
    ax4.plot(s2n, bo2, ".")
    ax4.errorbar(
        s2n[m],
        bo2[m],
        yerr=berr_o2[m],
        fmt=".r",
        ecolor="grey",
    )
    ax4.set_title(f"Full data range: O2 = {final_o2:0.3f}")
    ax4.axhline(final_o2)
    ax1.set_xlabel("SN")
    ax2.set_xlabel("SN")
    ax1.set_ylabel("H2O")
    ax2.set_ylabel("O2")
    if show_plots:
        plt.show()


def fit_exp_telluric(spec_objs, arc_sigma_mean, airmass, telluric_data, show_plots=False):
    tfiles, h2o, o2, tell_wave, tell_flux = telluric_data
    bo2 = np.nan*np.ones(len(spec_objs))
    berr_o2 = np.nan*np.ones(len(spec_objs))
    bh2o = np.nan*np.ones(len(spec_objs))
    berr_h2o = np.nan*np.ones(len(spec_objs))
    bw = np.nan*np.ones(len(spec_objs))
    bchi2 = np.inf*np.ones(len(spec_objs))
    s2n = np.nan*np.ones(len(spec_objs))
    for i, specobj in enumerate(spec_objs):
        mask_id = specobj['MASKDEF_ID']
        if specobj.S2N < 2:
            continue
        s2n[i] = specobj.S2N
        # CREATE DATA MASKS
        tell_mask = create_tell_masks(specobj.OPT_WAVE)
        cont_mask = create_cont_masks(specobj.OPT_WAVE)
        # Fit telluric, marginalizing over w
        tmp_chi = np.zeros(len(tfiles))
        tmp_w = np.zeros(len(tfiles))
        for j in tqdm(range(len(tfiles))):
            min_w, min_chi2 = telluric_marginalize_w(
                tell_wave[j],
                tell_flux[j],
                specobj.OPT_WAVE,
                specobj.OPT_COUNTS,
                specobj.OPT_COUNTS_IVAR,
                np.max([6800, np.min(specobj.OPT_WAVE[specobj.OPT_MASK])]),
                np.max(specobj.OPT_WAVE[specobj.OPT_MASK]),
                10, 
                arc_sigma_mean[mask_id]/0.02,
                cont_mask,
                tell_mask,
                npoly=5,
                show_plots=False,
            )
            tmp_chi[j] = min_chi2
            tmp_w[j] = min_w
        n = np.argmin(tmp_chi)
        bo2[i], berr_o2[i] = chi_interp_1d(tmp_chi, o2, h2o)
        bh2o[i], berr_h2o[i] = chi_interp_1d(tmp_chi, h2o, o2)
        bchi2[i] = tmp_chi[n]
        bw[i] = tmp_w[n]
        # GENERATE THE BEST MODEL
        model = generate_single_telluric(
            tell_wave[n],
            tell_flux[n],
            specobj.OPT_WAVE,
            specobj.OPT_COUNTS,
            specobj.OPT_COUNTS_IVAR,
            tmp_w[n],
            arc_sigma_mean[mask_id],
            cont_mask,
            npoly=8,
        )
        # PLOT
        if show_plots:
            qa_telluric_plot1(model, specobj, tell_mask, o2, h2o, tmp_chi)
    # CALCULATE FINAL EXPOSURE VALUE
    final_o2, final_h2o, m, tfine = calc_exp_tell(bo2, berr_o2, bh2o, berr_h2o, bchi2, bw)
    if show_plots:
        qa_telluric_plot2(
            s2n,
            final_o2, bo2, berr_o2,
            final_h2o, bh2o, berr_h2o,
            m,
            airmass,
            show_plots=show_plots,
        )
    return final_o2, final_h2o, bw, tfine


def fit_coadd_telluric(spec_objs, arc_sigma_mean, airmass, telluric_data, show_plots=False):
    tfiles, h2o, o2, tell_wave, tell_flux = telluric_data
    bo2 = np.nan*np.ones(len(spec_objs))
    berr_o2 = np.nan*np.ones(len(spec_objs))
    bh2o = np.nan*np.ones(len(spec_objs))
    berr_h2o = np.nan*np.ones(len(spec_objs))
    bw = np.nan*np.ones(len(spec_objs))
    bchi2 = np.inf*np.ones(len(spec_objs))
    s2n = np.nan*np.ones(len(spec_objs))
    for i, specobj in enumerate(spec_objs):
        if specobj.S2N < 2:
            continue
        mask_id = specobj['MASKDEF_ID']
        s2n[i] = specobj.S2N
        # CREATE DATA MASKS
        tell_mask = create_tell_masks(specobj.OPT_WAVE)
        cont_mask = create_cont_masks(specobj.OPT_WAVE)
        # Fit telluric, marginalizing over w
        tmp_chi = np.zeros(len(tfiles))
        tmp_w = np.zeros(len(tfiles))
        for j in tqdm(range(len(tfiles))):
            min_w, min_chi2 = telluric_marginalize_w(
                tell_wave[j],
                tell_flux[j],
                specobj.OPT_WAVE,
                specobj.OPT_COUNTS,
                specobj.OPT_COUNTS_IVAR,
                np.max([6800, np.min(specobj.OPT_WAVE[specobj.OPT_MASK])]),
                np.max(specobj.OPT_WAVE[specobj.OPT_MASK]),
                20, 
                arc_sigma_mean[mask_id]/0.02,
                cont_mask,
                tell_mask,
                npoly=5,
                show_plots=False,
            )
            tmp_chi[j] = min_chi2
            tmp_w[j] = min_w
        n = np.argmin(tmp_chi)
        bo2[i], berr_o2[i] = chi_interp_1d(tmp_chi, o2, h2o)
        bh2o[i], berr_h2o[i] = chi_interp_1d(tmp_chi, h2o, o2)
        bchi2[i] = tmp_chi[n]
        bw[i] = tmp_w[n]
        # GENERATE THE BEST MODEL
        model = generate_single_telluric(
            tell_wave[n],
            tell_flux[n],
            specobj.OPT_WAVE,
            specobj.OPT_COUNTS,
            specobj.OPT_COUNTS_IVAR,
            tmp_w[n],
            arc_sigma_mean[mask_id],
            cont_mask,
            npoly=8,
        )
        # PLOT
        if show_plots:
            qa_telluric_plot1(model, specobj, tell_mask, o2, h2o, tmp_chi)
    # CALCULATE FINAL EXPOSURE VALUE
    final_o2, final_h2o, m, tfine = calc_exp_tell(bo2, berr_o2, bh2o, berr_h2o, bchi2, bw)
    if show_plots:
        qa_telluric_plot2(
            s2n,
            final_o2, bo2, berr_o2,
            final_h2o, bh2o, berr_h2o,
            m,
            airmass,
            show_plots=show_plots,
        )
    return final_o2, final_h2o, bw, tfine
