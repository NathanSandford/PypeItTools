from tqdm import tqdm
import numpy as np
import pandas as pd
import scipy.ndimage as scipynd
import astropy.units as u
from astropy.io import fits
from pypeit.core.wave import airtovac
import matplotlib as mpl
import matplotlib.pyplot as plt

def load_stellar_templates(stellar_template_dir):
    temp_files = sorted(list(stellar_template_dir.glob('dmost*.fits')))
    temp_wave = airtovac(fits.open(temp_files[0])[1].data['wave'].flatten() * u.AA).value
    temp_flux = np.zeros((len(temp_files), len(temp_wave)))
    temp_meta = pd.DataFrame(index=[file.name for file in temp_files], columns=['teff', 'logg', 'feh', 'grid1', 'grid2', 'grid3'])
    for i, file in tqdm(enumerate(temp_files), total=len(temp_files)):
        hdu = fits.open(file)
        data = hdu[1].data
        temp_flux[i] = np.array(data['flux']).flatten()
        temp_meta.loc[file.name, 'teff'] = data['teff']
        temp_meta.loc[file.name, 'logg'] = data['logg']
        temp_meta.loc[file.name, 'feh'] = data['feh']
        temp_meta.loc[file.name, 'grid1'] = stellar_template_dir.joinpath('grid1').joinpath(file.name).exists()
        temp_meta.loc[file.name, 'grid2'] = stellar_template_dir.joinpath('grid2').joinpath(file.name).exists()
        temp_meta.loc[file.name, 'grid3'] = stellar_template_dir.joinpath('grid3').joinpath(file.name).exists()
    return temp_wave, temp_flux, temp_meta


def create_template_masks(data_wave):
    # DETERMINE CHI2 NEAR STRONG LINES
    #cmask1 = (data_wave > 6200) & (data_wave < 6750)
    #cmask2 = (data_wave > 6950) & (data_wave < 7140)
    #cmask3 = (data_wave > 7350) & (data_wave < 7550)
    #cmask4 = (data_wave > 7800) & (data_wave < 8125)
    #cmask5 = (data_wave > 8170) & (data_wave < 8210)
    #cmask6 = (data_wave > 8350) & (data_wave < 8875)
    #chi2_mask = cmask1 | cmask2 | cmask3 | cmask4 | cmask5 | cmask6
    chi2_mask = (data_wave > 3000) & (data_wave < 7500)
    # USE THIS FOR CONTINUUM FITTING
    # EXCLUDE FOR CONTINUUM FITTING
    #cmask1 = (data_wave > 6554) & (data_wave < 6567)
    #cmask2 = (data_wave > 6855) & (data_wave < 6912)
    #cmask3 = (data_wave > 7167) & (data_wave < 7320)
    #cmask4 = (data_wave > 7590) & (data_wave < 7680)
    #cmask5 = (data_wave > 8160) & (data_wave < 8300)
    #cmask6 = (data_wave > 8970) & (data_wave < 9030)
    #cmaski = cmask1 | cmask2 | cmask3 | cmask4 | cmask5 | cmask6
    #continuum_mask = np.invert(cmaski)
    continuum_mask = (data_wave > 3000) & (data_wave < 7500)
    return continuum_mask, chi2_mask
    
    
def fit_continuum_template(data_wave, data_flux, data_ivar, cmask, synth_flux, npoly):
    # This is a straight copy of dmost.core.telluric.fit_syn_continuum_telluric
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


def template_chi2_min(
    pwave, conv_spec, data_wave, data_flux, data_ivar, chi2_mask, cfit, vrange, show_plots=False,
):
    shift_wave = pwave[:, np.newaxis] * np.e ** (vrange / 2.997924e5)
    conv_shift_spec = np.apply_along_axis(lambda x: np.interp(data_wave, x, conv_spec), axis=0, arr=shift_wave)
    # FIT CONTINUUM
    final_fit = np.polyval(cfit, data_wave[:, np.newaxis]) * conv_shift_spec
    # CALCUALTE CHI2
    chi2 = np.sum(
        (data_flux[chi2_mask, np.newaxis] - final_fit[chi2_mask, :])**2 * data_ivar[chi2_mask, np.newaxis], axis=0
    ) / (np.sum(chi2_mask) - 3)
    # NOW CALCULATE BEST CHI2
    n = np.argmin(chi2)
    min_v = vrange[n]
    min_v_chi2 = chi2[n]
    if show_plots:
        plt.scatter(vrange, chi2)
        plt.axvline(min_v)
        plt.xlabel('v [km/s]')
        plt.ylabel('Chi2')
        plt.show()
    return min_v, min_v_chi2


def template_marginalize_v(phx_flux, phx_wave, data_wave, data_flux, data_ivar, losvd_pix, vbounds, vstep, cont_mask, chi2_mask, npoly, show_plots=False):
    # TRIM MODEL TO REDUCE COMPUTATION TIME
    mp = (phx_wave > np.min(data_wave) - 20) & (phx_wave < np.max(data_wave) + 20)
    phx_flux_trim = phx_flux[mp]
    phx_wave_trim = phx_wave[mp]
    conv_spec = scipynd.gaussian_filter1d(phx_flux_trim, losvd_pix, truncate=3)
    # FIT CONTINUUM OUTSIDE LOOP TO SAVE TIME
    tmp_flux = np.interp(data_wave, phx_wave_trim, conv_spec)
    cont_fit, _ = fit_continuum_template(data_wave, data_flux, data_ivar, cont_mask, tmp_flux, npoly)
    # SEARCH OVER VELOCITU SHIFT RANGE
    vrange = np.arange(vbounds[0], vbounds[1], vstep)
    min_v, min_v_chi2 = template_chi2_min(
            phx_wave_trim,
            conv_spec,
            data_wave,
            data_flux,
            data_ivar,
            chi2_mask,
            cont_fit,
            vrange,
            show_plots,
        )
    return min_v, min_v_chi2


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


def single_stellar_template(phx_wave, phx_flux, data_wave, data_flux, data_ivar, losvd_pix, vbest, npoly):
    # CREATE MODEL
    conv_spec = scipynd.gaussian_filter1d(phx_flux, losvd_pix, truncate=3)
    # MASK TELLURIC REGIONS
    cmask, chi2_mask = create_template_masks(data_wave)
    # Velocity shift star
    phx_logwave = np.log(phx_wave)
    shifted_logwave = phx_logwave + vbest/2.997924e5
    conv_int_flux   = np.interp(data_wave, np.exp(shifted_logwave), conv_spec)
    # FIT CONTINUUM
    p, final_model = fit_continuum_template(data_wave, data_flux, data_ivar, cmask, conv_int_flux, npoly)
    return final_model


def qa_template_plot(f, data_wave, data_flux, data_ivar, losvd_pix, temp_wave, temp_flux, best_chi, best_teff, best_logg, best_feh, n, chi2_mask, npoly, show_plots=False):
    vmn = np.log(np.min(best_chi))
    tmp = np.percentile(best_chi, 25)
    if vmn <= 0:
        tmp = 1.
        vmn = 0
    vmx = np.log(vmn + tmp)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
    plt.rcParams.update({'font.size': 14})
    aa, bb, cc = projected_chi_plot(best_teff, best_feh, best_chi)
    ax1.scatter(aa, bb, c=np.log(cc), vmin=vmn, vmax=vmx, cmap='cool')
    ax1.plot(best_teff[n], best_feh[n], 'o', mfc='none', mec='k', ms=15)
    ax1.set_xlabel('Teff')
    ax1.set_ylabel('[Fe/H]')
    ax1.set_xlim(2400, 8100)
    ax1.set_ylim(-4.2, 0.2)
    aa, bb, cc = projected_chi_plot(best_teff, best_logg, best_chi)
    ax2.scatter(aa, bb, c=np.log(cc), vmin=vmn, vmax=vmx, cmap='cool')
    ax2.plot(best_teff[n], best_logg[n], 'o', mfc='none', mec='k', ms=15)
    ax2.set_xlabel('Teff')
    ax2.set_ylabel('Logg')
    ax2.set_xlim(2400, 8100)
    ax2.set_ylim(0.5, 5.5)
    aa, bb, cc = projected_chi_plot(best_feh, best_logg, best_chi)
    ax3.scatter(aa, bb, c=np.log(cc), vmin=vmn, vmax=vmx, cmap='cool')
    ax3.plot(best_feh[n], best_logg[n], 'o', mfc='none', mec='k', ms=15)
    xlabel = ax3.set_xlabel('[Fe/H]')
    ax3.set_ylabel('Logg')
    ax3.set_ylim(-4.2, 0.2)
    ax3.set_ylim(0.5, 5.5)
    # MAKE COLORBAR
    v1 = np.linspace(vmn, vmx, 8, endpoint=True)
    cax, _ = mpl.colorbar.make_axes(ax3, ticks=v1)
    normalize = mpl.colors.Normalize(vmin=vmn, vmax=vmx)
    cbar = mpl.colorbar.ColorbarBase(cax, norm=normalize, cmap=mpl.cm.cool)
    positions = v1
    labels = ['{:0.1f}'.format(np.exp(i)) for i in v1]
    cbar.ax.yaxis.set_major_locator(mpl.ticker.FixedLocator(positions))
    cbar.ax.yaxis.set_major_formatter(mpl.ticker.FixedFormatter(labels))
    cbar.ax.set_ylabel('chi2')
    if show_plots:
        plt.show()
    fig, ax = plt.subplots(figsize=(20, 5))
    plt.rcParams.update({'font.size': 14})
    plt.plot(data_wave, data_flux, 'k', label='full spectrum', linewidth=0.9)
    plt.plot(data_wave[chi2_mask], data_flux[chi2_mask], 'grey', label='fitted region', linewidth=0.8)
    model = single_stellar_template(temp_wave, temp_flux[n], data_wave, data_flux,
                                    data_ivar, losvd_pix, f['chi2_v'], npoly)
    plt.plot(data_wave, model, 'r', label='Model', linewidth=0.8, alpha=0.8)
    plt.xlim(3900, 4100)
    plt.ylim(0, 1.2*np.quantile(data_flux[chi2_mask], 0.95))
    if show_plots:
        plt.show()
    plt.close(fig)
        
        
def fit_coadd_phoenix(stellar_template_data, data_wave, data_flux, data_ivar, losvd_pix, s2n, show_plots=False):
    # CONTINUUM POLYNOMIAL SET BY SN LIMITS
    if (s2n) > 100:
        npoly = 7
    else:
        npoly = 5
    cont_mask, chi2_mask = create_template_masks(data_wave)
    # DETERMINE APPROPRIATE GRID BASED ON S/N
    if s2n >= 25:
        grid = 'grid1'
    elif 10 < s2n < 25:
        grid = 'grid2'
    else:
        grid = 'grid3'
    # PARSE STELLAR TEMPLATE DATA
    temp_wave, temp_flux, temp_meta = stellar_template_data
    temp_grid_flux = temp_flux[np.argwhere(temp_meta[grid].values).flatten()]
    temp_grid_meta = temp_meta[temp_meta[grid]]
    # LOOP THROUGH ALL TEMPLATES
    tmp_v_chi2 = np.zeros(len(temp_grid_meta))
    tmp_v = np.zeros(len(temp_grid_meta))
    for j, file in enumerate(tqdm(temp_grid_meta.index, total=len(temp_grid_meta.index))):
        # RUN ONE STELLAR TEMPLATE
        min_v, min_v_chi2 = template_marginalize_v(
            temp_grid_flux[j],
            temp_wave,
            data_wave,
            data_flux,
            data_ivar,
            losvd_pix,
            [-500, 500],
            10,
            cont_mask,
            chi2_mask,
            npoly,
            show_plots=False,
        )
        tmp_v_chi2[j] = min_v_chi2
        tmp_v[j] = min_v
    m = tmp_v_chi2 > 0.1
    n = np.argmin(tmp_v_chi2[m])
    best_temp = temp_grid_meta.index[m][n]
    # REFINE V W/ BEST TEMPLATE
    min_v_final, min_v_final_chi2 = template_marginalize_v(
        temp_grid_flux[m][n],
        temp_wave,
        data_wave,
        data_flux,
        data_ivar,
        losvd_pix,
        [tmp_v[m][n]-100, tmp_v[m][n]+100],
        1,
        cont_mask,
        chi2_mask,
        npoly,
        show_plots=show_plots,
    )
    f = {}
    f['chi2_tfile'] = best_temp
    f['chi2_tgrid'] = grid
    f['chi2_tchi2'] = min_v_final_chi2
    f['chi2_v'] = min_v_final
    f['chi2_teff'] = temp_grid_meta.loc[best_temp]['teff']
    f['chi2_logg'] = temp_grid_meta.loc[best_temp]['logg']
    f['chi2_feh'] = temp_grid_meta.loc[best_temp]['feh']
    qa_template_plot(
        f,
        data_wave,
        data_flux,
        data_ivar,
        losvd_pix,
        temp_wave,
        temp_grid_flux[m],
        tmp_v_chi2[m],
        temp_grid_meta.loc[:, 'teff'].astype(float).values[m],
        temp_grid_meta.loc[:, 'logg'].astype(float).values[m],
        temp_grid_meta.loc[:, 'feh'].astype(float).values[m],
        n,
        chi2_mask,
        npoly,
        show_plots=show_plots,
    )
    return best_temp, f
    