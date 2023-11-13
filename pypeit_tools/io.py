import numpy as np
import pandas as pd


def strip(text):
    try:
        return text.strip()
    except AttributeError:
        return text


def make_int(text):
    try:
        return int(text.strip('" '))
    except ValueError:
        return np.nan


def make_float(text):
    try:
        return float(text.strip('" '))
    except ValueError:
        return np.nan


def read_spec1d_summary(spec1d_text_file, sn_thresh=2):
    df = pd.read_csv(
        spec1d_text_file,
        sep='|',
        usecols=[1,2,3,4,5,6,10,11,13],
        names=['slit', 'pypeit_name', 'mask_id', 'objname', 'objra', 'objdec', 'fwhm', 's2n', 'wv_rms'],
        skiprows=1,
        converters = {
            'slit': make_int,
            'pypeit_name': strip,
            'mask_id': make_int,
            'objname': strip,
            'objra': make_float,
            'objdec': make_float,
            'fwhm': make_float,
            's2n': make_float,
            'wv_rms': make_float,
        },
        index_col = 'mask_id'
    )
    return df


def read_stack1d_summary(stack1d_text_file):
    df = pd.read_csv(
        stack1d_text_file,
        sep='|',
        usecols=[1,2,3,4,5,6,10,11],
        names=['slit', 'pypeit_name', 'mask_id', 'objname', 'objra', 'objdec', 'fwhm', 's2n'],
        skiprows=1,
        converters = {
            'slit': make_int,
            'pypeit_name': strip,
            'mask_id': make_int,
            'objname': strip,
            'objra': make_float,
            'objdec': make_float,
            'fwhm': make_float,
            's2n': make_float,
        },
        index_col = 'mask_id'
    )
    return df
    

