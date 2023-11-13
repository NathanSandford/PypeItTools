import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
from astropy import units as u
from astroplan import  Observer
from pypeit.specobjs import SpecObjs
from pypeit.wavecalib import WaveCalib
from pypeit_tools import io
from pypeit_tools.line_fitting import line_residuals, fit_line_poly
from pypeit_tools.tellurics import fit_exp_telluric, fit_coadd_telluric
from pypeit_tools.phoenix import fit_coadd_phoenix
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


def keck_helio(mjd, ra, dec):
    '''
    Heliocentric velocity correction for Keck 
    
    Parameters
    ----------
    mjd: float
        mjd date of observatiom
    ra: float
        right ascension of observation
    dec: float
        declination of observation
    
    Returns
    -------
    vhelio
        heliocentric velocity correction
        This should be ADDED to the measured velocity
    '''
    t = Time(mjd, format="mjd")
    sc = SkyCoord(ra, dec, unit=(u.deg, u.deg))
    keck = EarthLocation.of_site("Keck Observatory")
    heliocorr = sc.radial_velocity_correction("heliocentric", obstime=t, location=keck)
    vhelio = heliocorr.to(u.km / u.s) * (u.s / u.km)
    return vhelio


def load_mask(filename):
    tmp = pd.read_hdf(filename, key='init')
    rdx_dir = tmp['rdx_dir'].iloc[0]
    mask_name = tmp['mask_name'].iloc[0]
    pypeit_rdx_names = tmp['pypeit_rdx_names'].to_dict()
    mask = LRISMask(rdx_dir, mask_name, pypeit_rdx_names)
    mask.exp_data = dict(
        LRISb=pd.read_hdf(filename, key='exp_data_b'),
        LRISr=pd.read_hdf(filename, key='exp_data_r'),
    )
    mask.exp_summary = dict(
        LRISb=pd.read_hdf(filename, key='exp_summary_b'),
        LRISr=pd.read_hdf(filename, key='exp_summary_r'),
    )
    mask.coadd_summary = pd.read_hdf(filename, key='coadd_summary')
    return mask


class LRISMask:
    def __init__(self, rdx_dir, mask_name, pypeit_rdx_names):
        self.mask_name = mask_name
        self.pypeit_rdx_names = pypeit_rdx_names
        self.rdx_dir = rdx_dir
        self.sci_dir = self.rdx_dir.joinpath('Science')
        self.coadd_dir = self.rdx_dir.joinpath('Science_coadd')
        self.cal_dir = self.rdx_dir.joinpath('Calibrations')

        # Load Individual Exposures
        self.exp_text_files = dict(
            LRISb=sorted(list(self.sci_dir.glob(f'spec1d*LB*{mask_name}*.txt'))),
            LRISr=sorted(list(self.sci_dir.glob(f'spec1d*LR*{mask_name}*.txt')))
        )
        self.exp_fits_files = dict(
            LRISb=sorted(list(self.sci_dir.glob(f'spec1d*LB*{mask_name}*.fits'))),
            LRISr=sorted(list(self.sci_dir.glob(f'spec1d*LR*{mask_name}*.fits'))),
        )
        self.nexp = dict(
            LRISb=len(self.exp_text_files['LRISb']),
            LRISr=len(self.exp_text_files['LRISr'])
        )
        self.load_exp_summary(sn_thresh=1, fwhm_thresh=1.5)
        self.load_exp_spec()
        self.read_exp_headers()

        # Load Coadded Spectra
        tmp = sorted(list(self.coadd_dir.glob(f'spec1d*{mask_name}*.txt')))
        self.coadd_text_file = dict(
            LRISb=tmp[0],
            LRISr=tmp[1]
        )
        tmp = sorted(list(self.coadd_dir.glob(f'spec1d*{mask_name}*.fits')))
        self.coadd_fits_file = dict(
            LRISb=tmp[0],
            LRISr=tmp[1]
        )
        self.load_coadd_summary(sn_thresh=1, fwhm_thresh=1.5)
        self.objnames = self.coadd_summary.index.values
        self.load_coadd_spec()

        self.load_slit_widths()
        
    def save_mask(self, filename):
        self.exp_data['LRISb'].to_hdf(filename, key='exp_data_b')
        self.exp_data['LRISr'].to_hdf(filename, key='exp_data_r')
        self.exp_summary['LRISb'].to_hdf(filename, key='exp_summary_b')
        self.exp_summary['LRISr'].to_hdf(filename, key='exp_summary_r')
        self.coadd_summary.to_hdf(filename, key='coadd_summary')
        pd.DataFrame(dict(
            rdx_dir=self.rdx_dir,
            mask_name=self.mask_name,
            pypeit_rdx_names=self.pypeit_rdx_names,
        )). to_hdf(filename, key='init')

    def load_exp_summary(self, sn_thresh=1, fwhm_thresh=1.5):
        df_b_list = []
        for i in range(self.nexp['LRISb']):
            spec1d_summary_file = self.exp_text_files['LRISb'][i]
            df_b = io.read_spec1d_summary(spec1d_summary_file)
            df_b['det'] = [name.split('-')[-1] for name in df_b['pypeit_name']]
            df_b['slit_width'] = np.nan
            df_b['lsf_mean'] = np.nan
            df_b['lsf_coeff_0'] = np.nan
            df_b['lsf_coeff_1'] = np.nan
            df_b['telluric_offset'] = np.nan
            df_b['stellar_template'] = np.nan
            df_b['stellar_template_rv'] = np.nan
            df_b['spec1d_file'] = spec1d_summary_file.stem
            df_b['raw_file'] = [
                f"{spec1d_file.split('spec1d_')[1].split('-')[0]}.fits.gz" for spec1d_file in df_b['spec1d_file']
            ]
            df_b['good'] = (
                (df_b['s2n'].values > sn_thresh) & (df_b['fwhm'].values < fwhm_thresh) & 
                ['SERENDIP' not in objname for objname in df_b['objname']] &
                ['SPURIOUS' not in objname for objname in df_b['objname']]
            )
            df_b_list.append(df_b)
        df_r_list = []
        for i in range(self.nexp['LRISr']):
            spec1d_summary_file = self.exp_text_files['LRISr'][i]
            df_r = io.read_spec1d_summary(spec1d_summary_file)
            df_r['det'] = [name.split('-')[-1] for name in df_r['pypeit_name']]
            df_r['slit_width'] = np.nan
            df_r['lsf_mean'] = np.nan
            df_r['lsf_coeff_0'] = np.nan
            df_r['lsf_coeff_1'] = np.nan
            df_r['telluric_offset'] = np.nan
            df_r['telluric_offset'] = np.nan
            df_r['stellar_template'] = np.nan
            df_r['stellar_template_rv'] = np.nan
            df_r['spec1d_file'] = spec1d_summary_file.stem
            df_r['raw_file'] = [
                f"{spec1d_file.split('spec1d_')[1].split('-')[0]}.fits.gz" for spec1d_file in df_r['spec1d_file']
            ]
            df_r['good'] = (
                (df_r['s2n'].values > sn_thresh) & (df_r['fwhm'].values < fwhm_thresh) & 
                ['SERENDIP' not in objname for objname in df_r['objname']] &
                ['SPURIOUS' not in objname for objname in df_r['objname']]
            )
            df_r_list.append(df_r)
        self.exp_summary = dict(
            LRISb=pd.concat(df_b_list).sort_values('mask_id'),
            LRISr=pd.concat(df_r_list).sort_values('mask_id')
        )
        
    def read_exp_headers(self):
        exp_data_b = pd.DataFrame(
            index=[filename.stem for filename in self.exp_text_files['LRISb']],
            columns=['MASK_RA', 'MASK_DEC', 'DICHROIC', 'GRISNAME', 'BINNING', 'AIRMASS', 'MJD', 'EXPTIME', 'FILENAME', 'VHELIO', 'telluric_o2', 'telluric_h2o', 'telluric_template'],
        )
        for exp_label, specobjs in self.exp_specobjs['LRISb'].items():
            header = specobjs.header
            exp_data_b.loc[exp_label, 'MASK_RA'] = header['RA']
            exp_data_b.loc[exp_label, 'MASK_DEC'] = header['DEC']
            exp_data_b.loc[exp_label, 'DICHROIC'] = header['DICHROIC']
            exp_data_b.loc[exp_label, 'GRISNAME'] = header['GRISNAME']
            exp_data_b.loc[exp_label, 'BINNING'] = header['BINNING']
            exp_data_b.loc[exp_label, 'AIRMASS'] = header['AIRMASS']
            exp_data_b.loc[exp_label, 'MJD'] = header['MJD']
            exp_data_b.loc[exp_label, 'EXPTIME'] = header['EXPTIME']
            exp_data_b.loc[exp_label, 'FILENAME'] = header['FILENAME']
            exp_data_b.loc[exp_label, 'VHELIO'] = keck_helio(
                header['MJD'],
                header['RA'],
                header['DEC']
            ).value
        airmass_coadd = np.average(exp_data_b['AIRMASS'], weights=exp_data_b['EXPTIME'])
        mjd_coadd = np.average(exp_data_b['MJD'], weights=exp_data_b['EXPTIME'])
        exp_data_b.loc['coadd', 'MASK_RA'] = exp_data_b['MASK_RA'].iloc[0]
        exp_data_b.loc['coadd', 'MASK_DEC'] = exp_data_b['MASK_DEC'].iloc[0]
        exp_data_b.loc['coadd', 'DICHROIC'] = exp_data_b['DICHROIC'].iloc[0]
        exp_data_b.loc['coadd', 'GRISNAME'] = exp_data_b['GRISNAME'].iloc[0]
        exp_data_b.loc['coadd', 'BINNING'] = exp_data_b['BINNING'].iloc[0]
        exp_data_b.loc['coadd', 'AIRMASS'] = airmass_coadd
        exp_data_b.loc['coadd', 'MJD'] = mjd_coadd
        exp_data_b.loc['coadd', 'EXPTIME'] = np.sum(exp_data_b['EXPTIME'])
        exp_data_b.loc['coadd', 'FILENAME'] = exp_data_b['FILENAME'].iloc[0]
        exp_data_b.loc['coadd', 'VHELIO'] = keck_helio(
            exp_data_b.loc['coadd', 'MJD'],
            exp_data_b.loc['coadd', 'MASK_RA'],
            exp_data_b.loc['coadd', 'MASK_DEC']
        ).value
        exp_data_r = pd.DataFrame(
            index=[filename.stem for filename in self.exp_text_files['LRISr']],
            columns=['MASK_RA', 'MASK_DEC', 'DICHROIC', 'GRANAME', 'GRANGLE', 'CENWAVE', 'BINNING', 'AIRMASS', 'MJD', 'EXPTIME', 'FILENAME', 'VHELIO', 'telluric_o2', 'telluric_h2o', 'telluric_template'],
        )
        for exp_label, specobjs in self.exp_specobjs['LRISr'].items():
            header = specobjs.header
            exp_data_r.loc[exp_label, 'MASK_RA'] = header['RA']
            exp_data_r.loc[exp_label, 'MASK_DEC'] = header['DEC']
            exp_data_r.loc[exp_label, 'DICHROIC'] = header['DICHROIC']
            exp_data_r.loc[exp_label, 'GRANAME'] = header['GRANAME']
            exp_data_r.loc[exp_label, 'GRANGLE'] = header['GRANGLE']
            exp_data_r.loc[exp_label, 'CENWAVE'] = header['CENWAVE']
            exp_data_r.loc[exp_label, 'BINNING'] = header['BINNING']
            exp_data_r.loc[exp_label, 'AIRMASS'] = header['AIRMASS']
            exp_data_r.loc[exp_label, 'MJD'] = header['MJD']
            exp_data_r.loc[exp_label, 'EXPTIME'] = header['EXPTIME']
            exp_data_r.loc[exp_label, 'FILENAME'] = header['FILENAME']
            exp_data_r.loc[exp_label, 'VHELIO'] = keck_helio(
                header['MJD'],
                header['RA'],
                header['DEC']
            ).value
        airmass_coadd = np.average(exp_data_r['AIRMASS'], weights=exp_data_r['EXPTIME'])
        mjd_coadd = np.average(exp_data_r['MJD'], weights=exp_data_r['EXPTIME'])
        exp_data_r.loc['coadd', 'MASK_RA'] = exp_data_r['MASK_RA'].iloc[0]
        exp_data_r.loc['coadd', 'MASK_DEC'] = exp_data_r['MASK_DEC'].iloc[0]
        exp_data_r.loc['coadd', 'DICHROIC'] = exp_data_r['DICHROIC'].iloc[0]
        exp_data_r.loc['coadd', 'GRANAME'] = exp_data_r['GRANAME'].iloc[0]
        exp_data_r.loc['coadd', 'GRANGLE'] = exp_data_r['GRANGLE'].iloc[0]
        exp_data_r.loc['coadd', 'CENWAVE'] = exp_data_r['CENWAVE'].iloc[0]
        exp_data_r.loc['coadd', 'BINNING'] = exp_data_r['BINNING'].iloc[0]
        exp_data_r.loc['coadd', 'AIRMASS'] = airmass_coadd
        exp_data_r.loc['coadd', 'MJD'] = mjd_coadd
        exp_data_r.loc['coadd', 'EXPTIME'] = np.sum(exp_data_r['EXPTIME'])
        exp_data_r.loc['coadd', 'FILENAME'] = exp_data_r['FILENAME'].iloc[0]
        exp_data_r.loc['coadd', 'VHELIO'] = keck_helio(
            exp_data_r.loc['coadd', 'MJD'],
            exp_data_r.loc['coadd', 'MASK_RA'],
            exp_data_r.loc['coadd', 'MASK_DEC']
        ).value
        self.exp_data = dict(LRISb=exp_data_b, LRISr=exp_data_r)
        
    def load_exp_spec(self):
        LRISb = {}
        for spec1d_file in self.exp_fits_files['LRISb']:
            spec_objs = SpecObjs.from_fitsfile(spec1d_file, chk_version=False)
            for spec_obj in spec_objs:
                spec_obj['MASKDEF_OBJNAME'] = self.exp_summary['LRISb'].loc[
                    (self.exp_summary['LRISb']['spec1d_file'] == spec1d_file.stem)
                    & (self.exp_summary['LRISb']['pypeit_name'] == spec_obj['NAME']),
                    'objname'
                ].values[0]
            LRISb[spec1d_file.stem] = spec_objs
        LRISr = {}
        for spec1d_file in self.exp_fits_files['LRISr']:
            spec_objs = SpecObjs.from_fitsfile(spec1d_file, chk_version=False)
            for spec_obj in spec_objs:
                spec_obj['MASKDEF_OBJNAME'] = self.exp_summary['LRISr'].loc[
                    (self.exp_summary['LRISr']['spec1d_file'] == spec1d_file.stem)
                    & (self.exp_summary['LRISr']['pypeit_name'] == spec_obj['NAME']),
                    'objname'
                ].values[0]
            LRISr[spec1d_file.stem] = spec_objs
        self.exp_specobjs = dict(LRISb=LRISb, LRISr=LRISr)
    
    def load_coadd_summary(self, sn_thresh=2, fwhm_thresh=1.5, drop_serendip=True):
        df_b = io.read_stack1d_summary(self.coadd_text_file['LRISb'])
        df_r = io.read_stack1d_summary(self.coadd_text_file['LRISr'])
        df_b['det'] = ([name.split('-')[-1] for name in df_b['pypeit_name']])
        df_r['det'] = ([name.split('-')[-1] for name in df_r['pypeit_name']])
        df_b['good'] = (
            (df_b['s2n'].values > sn_thresh) & (df_b['fwhm'].values < fwhm_thresh) & 
            ['SERENDIP' not in objname for objname in df_b['objname']] & 
            ['SPURIOUS' not in objname for objname in df_b['objname']]
        )
        df_r['good'] = (
            (df_r['s2n'].values > sn_thresh) & (df_r['fwhm'].values < fwhm_thresh) & 
            ['SERENDIP' not in objname for objname in df_r['objname']] & 
            ['SPURIOUS' not in objname for objname in df_r['objname']]
        )
        self._coadd_summary = dict(LRISb=df_b, LRISr=df_r)
        self.collate_coadd_summary()
        
    def collate_coadd_summary(self):
        all_mask_names = np.unique(
            list(self._coadd_summary['LRISb']['objname']) + list(self._coadd_summary['LRISr']['objname'])
        )
        df = pd.DataFrame(
            index=all_mask_names,
            columns=[
                'mask_id', 'objra', 'objdec', 
                'slit_b', 'slit_r',
                'pypeit_name_b', 'pypeit_name_r',
                'fwhm_b', 'fwhm_r',
                's2n_b', 's2n_r',
                'wv_rms_b', 'wv_rms_r',
                'det', 'slit_width',
                'lsf_mean_b', 'lsf_coeff_0_b', 'lsf_coeff_1_b',
                'lsf_mean_r', 'lsf_coeff_0_r', 'lsf_coeff_1_r',
                'telluric_offset_b', 'telluric_offset_r',
                'stellar_template_b', 'stellar_template_r', 
                'stellar_template_rv_b', 'stellar_template_rv_r', 
                'good_b', 'good_r',
            ]
        )
        df.index.name = 'objname'
        for name in all_mask_names:
            try:
                lris_b_idx = np.argwhere(self._coadd_summary['LRISb']['objname'] == name)[0][0]
                df.loc[name, 'mask_id'] = self._coadd_summary['LRISb'].index[lris_b_idx]
                df.loc[name, 'objra'] = self._coadd_summary['LRISb'].iloc[lris_b_idx]['objra']
                df.loc[name, 'objdec'] = self._coadd_summary['LRISb'].iloc[lris_b_idx]['objdec']
                df.loc[name, 'slit_b'] = self._coadd_summary['LRISb'].iloc[lris_b_idx]['slit']
                df.loc[name, 'pypeit_name_b'] = self._coadd_summary['LRISb'].iloc[lris_b_idx]['pypeit_name']
                df.loc[name, 'fwhm_b'] = self._coadd_summary['LRISb'].iloc[lris_b_idx]['fwhm']
                df.loc[name, 's2n_b'] = self._coadd_summary['LRISb'].iloc[lris_b_idx]['s2n']
                df.loc[name, 'det'] = self._coadd_summary['LRISb'].iloc[lris_b_idx]['det']
                df.loc[name, 'good_b'] = self._coadd_summary['LRISb'].iloc[lris_b_idx]['good']
            except IndexError:
                lris_r_idx = np.argwhere(self._coadd_summary['LRISr']['objname'] == name)[0][0]
                print(f"LRISb data not found for objname={name}")
                df.loc[name, 'mask_id'] = self._coadd_summary['LRISr'].index[lris_r_idx]
                df.loc[name, 'objra'] = self._coadd_summary['LRISr'].iloc[lris_r_idx]['objra']
                df.loc[name, 'objdec'] = self._coadd_summary['LRISr'].iloc[lris_r_idx]['objdec']
            try:
                lris_r_idx = np.argwhere(self._coadd_summary['LRISr']['objname'] == name)[0][0]
                df.loc[name, 'slit_r'] = self._coadd_summary['LRISr'].iloc[lris_r_idx]['slit']
                df.loc[name, 'pypeit_name_r'] = self._coadd_summary['LRISr'].iloc[lris_r_idx]['pypeit_name']
                df.loc[name, 'fwhm_r'] = self._coadd_summary['LRISr'].iloc[lris_r_idx]['fwhm']
                df.loc[name, 's2n_r'] = self._coadd_summary['LRISr'].iloc[lris_r_idx]['s2n']
                df.loc[name, 'det'] = self._coadd_summary['LRISr'].iloc[lris_r_idx]['det']
                df.loc[name, 'good_r'] = self._coadd_summary['LRISr'].iloc[lris_r_idx]['good']
            except IndexError:
                print(f"LRISr data not found for objname={name}")
            try:
                df.loc[name, 'wv_rms_b'] = self.exp_summary['LRISb'].loc[
                    self.exp_summary['LRISb'].index == df.loc[name, 'mask_id'], 
                    'wv_rms'
                ].mean()
            except KeyError:
                print(f"LRISb data not found in any individual exposure for objname={name}")
            try:
                df.loc[name, 'wv_rms_r'] = self.exp_summary['LRISr'].loc[
                    self.exp_summary['LRISr'].index == df.loc[name, 'mask_id'], 
                    'wv_rms'
                ].mean()
            except KeyError:
                print(f"LRISr data not found in any individual exposure for mask_id={name}")
        self.coadd_summary = df.sort_values('mask_id')
        
    def load_coadd_spec(self):
        specobjs_b = SpecObjs.from_fitsfile(self.coadd_fits_file['LRISb'], chk_version=False)
        specobjs_r = SpecObjs.from_fitsfile(self.coadd_fits_file['LRISr'], chk_version=False)
        for spec_obj in specobjs_b:
            spec_obj['MASKDEF_OBJNAME'] = self.coadd_summary.loc[
                (self.coadd_summary['pypeit_name_b'] == spec_obj['NAME'])
            ].index[0]
            spec_obj['MASKDEF_ID'] = self.coadd_summary.loc[
                (self.coadd_summary['pypeit_name_b'] == spec_obj['NAME']),
                'mask_id'
            ].values[0]
        for spec_obj in specobjs_r:
            spec_obj['MASKDEF_OBJNAME'] = self.coadd_summary.loc[
                (self.coadd_summary['pypeit_name_r'] == spec_obj['NAME']),
            ].index[0]
            spec_obj['MASKDEF_ID'] = self.coadd_summary.loc[
                (self.coadd_summary['pypeit_name_r'] == spec_obj['NAME']),
                'mask_id'
            ].values[0]
        self.coadd_specobjs = dict(
            LRISb=specobjs_b,
            LRISr=specobjs_r
        )

    def load_slit_widths(self):
        self.coadd_summary.loc[:, 'slit_width'] = np.nan
        for det in ['DET01', 'DET02']:
            slit_file_b = self.cal_dir.joinpath(f"Slits_{self.pypeit_rdx_names['LRISb']}_0_{det}.fits.gz")
            slit_file_r = self.cal_dir.joinpath(f"Slits_{self.pypeit_rdx_names['LRISr']}_0_{det}.fits.gz")
            hudl_b = fits.open(slit_file_b)
            hudl_r = fits.open(slit_file_r)
            for i, id in enumerate(hudl_b[2].data['MASKDEF_ID']):
                if (hudl_b[2].data['ALIGN'][i] == 1) or (id not in self.exp_summary['LRISb'].index):
                    continue
                self.exp_summary['LRISb'].loc[id, 'slit_width'] = hudl_b[2].data['SLITWID'][i].round(2)
                self.coadd_summary.loc[self.coadd_summary['mask_id'] == id, 'slit_width'] = hudl_b[2].data['SLITWID'][i].round(2)
            for i, id in enumerate(hudl_r[2].data['MASKDEF_ID']):
                if (hudl_r[2].data['ALIGN'][i] == 1) or (id not in self.exp_summary['LRISr'].index):
                    continue
                self.exp_summary['LRISr'].loc[id, 'slit_width'] = hudl_r[2].data['SLITWID'][i].round(2)
                self.coadd_summary.loc[self.coadd_summary['mask_id'] == id, 'slit_width'] = hudl_r[2].data['SLITWID'][i].round(2)

    def get_exp_specobj(self, objname=None):
        if objname is not None:
            spec1d_files_b = self.exp_summary['LRISb'].loc[
                self.exp_summary['LRISb']['objname'] == objname,
                'spec1d_file'
            ].values
            pypeit_names_b = self.exp_summary['LRISb'].loc[
                self.exp_summary['LRISb']['objname'] == objname,
                'pypeit_name'
            ].values
            spec1d_files_r = self.exp_summary['LRISr'].loc[
                self.exp_summary['LRISr']['objname'] == objname,
                'spec1d_file'
            ].values
            pypeit_names_r = self.exp_summary['LRISr'].loc[
                self.exp_summary['LRISr']['objname'] == objname,
                'pypeit_name'
            ].values
        else:
            raise RuntimeError('Must specify an objname')
        # LRISb
        specobjs_b = {}
        for i, exp in enumerate(spec1d_files_b):
            specobjs = self.exp_specobjs['LRISb'][exp]
            specobjs_b[exp] = specobjs[specobjs.NAME == pypeit_names_b[i]][0]
        # LRISr
        specobjs_r = {}
        for i, exp in enumerate(spec1d_files_r):
            specobjs = self.exp_specobjs['LRISr'][exp]
            specobjs_r[exp] = specobjs[specobjs.NAME == pypeit_names_r[i]][0]
        return dict(LRISb=specobjs_b, LRISr=specobjs_r)
    
    def get_coadd_specobj(self, objname=None):
        if objname is not None:
            pypeit_name_b, pypeit_name_r = self.coadd_summary.loc[
                self.coadd_summary.index == objname,
                ['pypeit_name_b', 'pypeit_name_r']
            ].values[0]
        else:
            raise RuntimeError('Must specify either objname')
        try:
            specobj_b = self.coadd_specobjs['LRISb'][self.coadd_specobjs['LRISb'].NAME == pypeit_name_b][0]
        except IndexError:
            print(f'Could not retrieve LRISb specobj for {objname}')
            specobj_b = None
        try:
            specobj_r = self.coadd_specobjs['LRISr'][self.coadd_specobjs['LRISr'].NAME == pypeit_name_r][0]
        except IndexError:
            print(f'Could not retrieve LRISr specobj for {objname}')
            specobj_r = None
        return dict(LRISb=specobj_b, LRISr=specobj_r)


    def plot_obj_spec(self, objname=None):
        if objname is not None:
            coadd_specobj_b = self.get_coadd_specobj(objname=objname)['LRISb']
            coadd_specobj_r = self.get_coadd_specobj(objname=objname)['LRISr']
            exp_specobjs_b = self.get_exp_specobj(objname=objname)['LRISb']
            exp_specobjs_r = self.get_exp_specobj(objname=objname)['LRISr']
            #mask_id = self.coadd_summary.loc[
            #    self.coadd_summary['objname'] == objname
            #].index[0]
        else:
            raise RuntimeError('Must specify either objname')
        fig = plt.figure(figsize=(100, 20))
        gs = GridSpec(4, 1)
        gs.update(hspace=0.0)
        ax1 = plt.subplot(gs[:3, 0])
        ax2 = plt.subplot(gs[3, 0], sharex=ax1)
        ax1.set_title(f'{objname}', fontsize=86, pad=30)
        ymax1 = 0
        ymax2 = 0
        for specobj in exp_specobjs_b:
            ax1.scatter(
                specobj.OPT_WAVE[specobj.OPT_MASK],
                specobj.OPT_COUNTS[specobj.OPT_MASK],
                marker='.',
                alpha=0.2
            )
            ax2.scatter(
                specobj.OPT_WAVE[specobj.OPT_MASK],
                (specobj.OPT_COUNTS*np.sqrt(specobj.OPT_COUNTS_IVAR))[specobj.OPT_MASK],
                marker='.',
                alpha=0.2,
            )
        for specobj in exp_specobjs_r:
            ax1.scatter(
                specobj.OPT_WAVE[specobj.OPT_MASK],
                specobj.OPT_COUNTS[specobj.OPT_MASK],
                marker='.',
                alpha=0.2
            )
            ax2.scatter(
                specobj.OPT_WAVE[specobj.OPT_MASK],
                (specobj.OPT_COUNTS*np.sqrt(specobj.OPT_COUNTS_IVAR))[specobj.OPT_MASK],
                marker='.',
                alpha=0.2,
            )
        if coadd_specobj_b is not None:
            ax1.plot(
                coadd_specobj_b.OPT_WAVE[coadd_specobj_b.OPT_MASK],
                coadd_specobj_b.OPT_COUNTS[coadd_specobj_b.OPT_MASK],
                c='k'
            )
            ax2.plot(
                coadd_specobj_b.OPT_WAVE[coadd_specobj_b.OPT_MASK],
                (coadd_specobj_b.OPT_COUNTS*np.sqrt(coadd_specobj_b.OPT_COUNTS_IVAR))[coadd_specobj_b.OPT_MASK],
                c='k',
            )
            pypeit_name_b = self.coadd_summary.loc[objname, 'pypeit_name_b']
            ax1.text(
                coadd_specobj_b.OPT_WAVE[coadd_specobj_b.OPT_MASK].mean(),
                -50,
                pypeit_name_b,
                fontsize=64,
                horizontalalignment='center',
            )
            ymax1 = np.max(
                [
                    ymax1,
                    1.5*np.quantile(coadd_specobj_b.OPT_COUNTS[coadd_specobj_b.OPT_MASK], 0.95), 
                ]
            )
            ymax2 = np.max(
                [
                    ymax2,
                    1.5*np.quantile((coadd_specobj_b.OPT_COUNTS*np.sqrt(coadd_specobj_b.OPT_COUNTS_IVAR))[coadd_specobj_b.OPT_MASK], 0.95)
                ]
            )
        if coadd_specobj_r is not None:
            ax1.plot(
                coadd_specobj_r.OPT_WAVE[coadd_specobj_r.OPT_MASK], 
                coadd_specobj_r.OPT_COUNTS[coadd_specobj_r.OPT_MASK],
                c='k'
            )
            ax2.plot(
                coadd_specobj_r.OPT_WAVE[coadd_specobj_r.OPT_MASK],
                (coadd_specobj_r.OPT_COUNTS*np.sqrt(coadd_specobj_r.OPT_COUNTS_IVAR))[coadd_specobj_r.OPT_MASK],
                c='k',
            )
            pypeit_name_r = self.coadd_summary.loc[objname, 'pypeit_name_r']
            ax1.text(
                coadd_specobj_r.OPT_WAVE[coadd_specobj_r.OPT_MASK].mean(),
                -50,
                pypeit_name_r,
                fontsize=64,
                horizontalalignment='center',
            )
            ymax1 = np.max(
                [
                    ymax1,
                    1.5*np.quantile(coadd_specobj_r.OPT_COUNTS[coadd_specobj_r.OPT_MASK], 0.95), 
                ]
            )
            ymax2 = np.max(
                [
                    ymax2,
                    1.5*np.quantile((coadd_specobj_r.OPT_COUNTS*np.sqrt(coadd_specobj_r.OPT_COUNTS_IVAR))[coadd_specobj_r.OPT_MASK], 0.95)
                ]
            )
        ax2.set_xlabel('Wavelength [AA]', fontsize=64)
        ax1.set_ylabel('Flux [counts]', fontsize=64)
        ax2.set_ylabel('S/N', fontsize=64)
        ax1.tick_params('x', labelsize=0)
        ax1.tick_params('y', labelsize=48, pad=15)
        ax2.tick_params(labelsize=48, pad=15)
        ax1.set_xlim(3000, 9000)
        ax1.set_ylim(-75, ymax1)
        ax2.set_ylim(-10, ymax2)
        ax1.grid(True)
        ax2.grid(True)
        plt.setp(ax1.spines.values(), lw=10)
        plt.setp(ax2.spines.values(), lw=10)
        plt.show()
        
    def plot_all_obj_spec(self, include_serendip=False, include_spurious=False):
        for objname in self.objnames:
            if 'SERENDIP' in objname and not include_serendip:
                print(f'Skipping {objname}')
            elif 'SPURIOUS' in objname and not include_spurious:
                print(f'Skipping {objname}')
            else:
                self.plot_obj_spec(objname=objname)

    def plot_rms_seeing(self):
        fig = plt.figure(figsize=(10, 5))
        gs = GridSpec(1, 2)
        gs.update(wspace=0.0)
        ax1 = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[0, 1], sharey=ax1)
        fig.suptitle(f'{self.mask_name}', y=0.92)
        
        ax1.hist(self.coadd_summary['wv_rms_b'], histtype='step', color='b', bins=np.arange(0, 0.6, 0.01))
        ax1.hist(self.coadd_summary['wv_rms_r'], histtype='step', color='r', bins=np.arange(0, 0.6, 0.01))
        ax1.set_xlim(0, 0.49)
        ax1.set_xlabel('Wave RMS [pixels]')
        
        ax2.hist(self.coadd_summary['fwhm_b'], histtype='step', color='b', bins=np.arange(0.6, 1.5, 0.05))
        ax2.hist(self.coadd_summary['fwhm_r'], histtype='step', color='r', bins=np.arange(0.6, 1.5, 0.05))
        ax2.set_xlim(0.6, 1.5)
        ax2.set_xlabel('Seeing [Arcsec]')
        
        ax2.tick_params('y', labelsize=0)
        plt.show()

    def calc_resolution(self, show_plots=False):
        for det in ['DET01', 'DET02']:
            wavecal_file_b = self.cal_dir.joinpath(f"WaveCalib_{self.pypeit_rdx_names['LRISb']}_0_{det}.fits")
            wavecal_file_r = self.cal_dir.joinpath(f"WaveCalib_{self.pypeit_rdx_names['LRISr']}_0_{det}.fits")
            wavecal_b, _, _, _ = WaveCalib._parse(fits.open(wavecal_file_b))
            wavecal_r, _, _, _ = WaveCalib._parse(fits.open(wavecal_file_r))
            for i, wv_fit in enumerate(wavecal_b['wv_fits']):
                spat_id = wv_fit['spat_id']
                if spat_id not in self.exp_summary['LRISb']['slit'].values:
                    print(spat_id)
                    continue
                mask_id = self.exp_summary['LRISb'][
                    (self.exp_summary['LRISb']['slit'] == spat_id) & (self.exp_summary['LRISb']['det'] == det)
                ].index[0]
                arc_b = dict(Wave=wv_fit['wave_fit'])
                arc_lines, arc_diff, arc_ediff, arc_los, arc_elos = line_residuals(
                    wv_fit['wave_soln'],
                    wavecal_b['arc_spectra'][:, i],
                    np.ones_like(wv_fit['wave_soln']),
                    np.ones_like(wv_fit['wave_soln']).astype(bool),
                    arc_b,
                    noff=5,
                    nfit_min=10,
                    diff_err_max=0.2,
                    los_err_max=0.2,
                    diff_guess=0.0,
                    los_guess=1.2,
                )
                fitted_line, good_lines = fit_line_poly(arc_lines, arc_los, arc_elos, deg=1)
                self.exp_summary['LRISb'].loc[mask_id, 'lsf_mean'] = np.average(arc_los[good_lines], weights=1/arc_elos[good_lines])
                self.exp_summary['LRISb'].loc[mask_id, 'lsf_coeff_0'] = fitted_line[0]
                self.exp_summary['LRISb'].loc[mask_id, 'lsf_coeff_1'] = fitted_line[1]
                self.coadd_summary.loc[self.coadd_summary['mask_id'] == mask_id, 'lsf_mean_b'] = np.average(arc_los[good_lines], weights=1/arc_elos[good_lines])
                self.coadd_summary.loc[self.coadd_summary['mask_id'] == mask_id, 'lsf_coeff_0_b'] = fitted_line[0]
                self.coadd_summary.loc[self.coadd_summary['mask_id'] == mask_id, 'lsf_coeff_1_b'] = fitted_line[1]
                if show_plots:
                    plt.errorbar(arc_lines[good_lines], arc_los[good_lines], yerr=arc_elos[good_lines], fmt='o', color='k')
                    plt.errorbar(arc_lines[~good_lines], arc_los[~good_lines], yerr=arc_elos[~good_lines], fmt='o', color='r')
                    plt.plot(wv_fit['wave_soln'], fitted_line(wv_fit['wave_soln']), c='k', ls='--')
                    plt.axhline(np.average(arc_los[good_lines], weights=1/arc_elos[good_lines]), c='k', ls=':')
                    plt.show()
            for i, wv_fit in enumerate(wavecal_r['wv_fits']):
                spat_id = wv_fit['spat_id']
                if spat_id not in self.exp_summary['LRISr']['slit'].values:
                    continue
                mask_id = self.exp_summary['LRISr'][
                    (self.exp_summary['LRISr']['slit'] == spat_id) & (self.exp_summary['LRISr']['det'] == det)
                ].index[0]
                arc_r = dict(Wave=wv_fit['wave_fit'])
                arc_lines, arc_diff, arc_ediff, arc_los, arc_elos = line_residuals(
                    wv_fit['wave_soln'],
                    wavecal_r['arc_spectra'][:, i],
                    np.ones_like(wv_fit['wave_soln']),
                    np.ones_like(wv_fit['wave_soln']).astype(bool),
                    arc_r,
                    noff=5,
                    nfit_min=10,
                    diff_err_max=0.1,
                    los_err_max=0.1,
                    diff_guess=0.0,
                    los_guess=0.6,
                )
                fitted_line, good_lines = fit_line_poly(arc_lines, arc_los, arc_elos, deg=1)
                self.exp_summary['LRISr'].loc[mask_id, 'lsf_mean'] = np.average(arc_los[good_lines], weights=1/arc_elos[good_lines])
                self.exp_summary['LRISr'].loc[mask_id, 'lsf_coeff_0'] = fitted_line[0]
                self.exp_summary['LRISr'].loc[mask_id, 'lsf_coeff_1'] = fitted_line[1]
                self.coadd_summary.loc[self.coadd_summary['mask_id'] == mask_id, 'lsf_mean_r'] = np.average(arc_los[good_lines], weights=1/arc_elos[good_lines])
                self.coadd_summary.loc[self.coadd_summary['mask_id'] == mask_id, 'lsf_coeff_0_r'] = fitted_line[0]
                self.coadd_summary.loc[self.coadd_summary['mask_id'] == mask_id, 'lsf_coeff_1_r'] = fitted_line[1]
                if show_plots:
                    plt.errorbar(arc_lines[good_lines], arc_los[good_lines], yerr=arc_elos[good_lines], fmt='o', color='k')
                    plt.errorbar(arc_lines[~good_lines], arc_los[~good_lines], yerr=arc_elos[~good_lines], fmt='o', color='r')
                    plt.plot(wv_fit['wave_soln'], fitted_line(wv_fit['wave_soln']), c='k', ls='--')
                    plt.axhline(np.average(arc_los[good_lines], weights=1/arc_elos[good_lines]), c='k', ls=':')
                    plt.show()
                    
    def fit_telluric_template(self, telluric_data, show_plots=True):
        self.telluric_bw = dict(LRISb={}, LRISr={})
        for exp in self.exp_specobjs['LRISr'].keys():
            tmp = fit_exp_telluric(
                spec_objs=self.exp_specobjs['LRISr'][exp],
                arc_sigma_mean=self.exp_summary['LRISr']['lsf_mean'].drop_duplicates(),
                airmass=self.exp_data['LRISr'].loc[exp, 'AIRMASS'],
                telluric_data=telluric_data,
                show_plots=show_plots,
            )
            self.exp_data['LRISr'].loc[exp, 'telluric_o2'] = tmp[0]
            self.exp_data['LRISr'].loc[exp, 'telluric_h2o'] = tmp[1]
            self.exp_data['LRISr'].loc[exp, 'telluric_template'] = tmp[3]
            self.telluric_bw['LRISr'][exp] = tmp[2]
            #self.exp_summary['LRISr'].loc[self.exp_summary['LRISr']['spec1d_file'] == exp, 'telluric_offset'] = tmp[2]
        tmp = fit_coadd_telluric(
            spec_objs=self.coadd_specobjs['LRISr'],
            arc_sigma_mean=self.exp_summary['LRISr']['lsf_mean'].drop_duplicates(),
            airmass=self.exp_data['LRISr'].loc['coadd', 'AIRMASS'],
            telluric_data=telluric_data,
            show_plots=show_plots,
        )
        self.exp_data['LRISr'].loc['coadd', 'telluric_o2'] = tmp[0]
        self.exp_data['LRISr'].loc['coadd', 'telluric_h2o'] = tmp[1]
        self.exp_data['LRISr'].loc['coadd', 'telluric_template'] = tmp[3]
        self.telluric_bw['LRISr']['coadd'] = tmp[2]
        #self.coadd_summary['LRISr'].loc[self.exp_summary['LRISr']['spec1d_file'] == exp, 'telluric_offset'] = tmp[2]
                    
    def fit_stellar_template(self, stellar_template_data, show_plots=True):
        for objname in self.objnames:
            s2n = self.coadd_summary.loc[objname, 's2n_b']
            if s2n < 2:
                print(f"Skipping {objname}, S/N too low ({s2n})")
                continue
            print(f"{objname}: S/N = {s2n}")
            specobj_pair = self.get_coadd_specobj(objname=objname)
            specobj = specobj_pair['LRISb']
            try:
                mask_id = specobj['MASKDEF_ID']
            except TypeError:
                print(f"Skipping {objname}, Could not load spectrum")
                continue
            arc_sigma_mean = self.coadd_summary.loc[objname, 'lsf_mean_b']
            tfile, f = fit_coadd_phoenix(
                stellar_template_data,
                specobj.OPT_WAVE,
                specobj.OPT_COUNTS,
                specobj.OPT_COUNTS_IVAR,
                arc_sigma_mean / 0.02,
                s2n,
                show_plots=show_plots,
            )
            self.coadd_summary.loc[objname, 'stellar_template_b'] = tfile
            self.coadd_summary.loc[objname, 'stellar_template_rv_b'] = f['chi2_v']
            