#!/usr/bin/env python
from astropy.coordinates import SkyCoord
from astropy.table import Table, hstack
from astropy.nddata import CCDData
from astropy.stats import mad_std
from astropy.io import fits
from astropy.visualization import imshow_norm, ZScaleInterval
from scipy.optimize import minimize
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from photutils import Background2D, SkyCircularAperture, SkyCircularAnnulus, aperture_photometry
import os
import logging
import argparse

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

target_coords = SkyCoord('19:18:45.580 +49:37:56.03', unit=(u.hourangle, u.deg))
ps1_catalog_path = 'catalogs/spikey_catalog.csv'
apass_catalog_path = 'catalogs/spikey_apass_catalog.csv'
image_dir = 'plots'
lc_file = 'lc.txt'
plot_colors = {'rp': 'r', 'V': 'g'}
plot_markers = {'1m0-06': 'h', '1m0-08': '8', '1.3m McGraw-Hill': '^', '2.4m Hiltner': 's'}
saturation = 50000.


def load_catalog(catalog_path):
    catalog = Table.read(catalog_path, format='ascii.csv', fill_values=[('-999.0', '0'), ('', '0')])
    ras = catalog['raMean'] if 'raMean' in catalog.colnames else catalog['RAJ2000']
    decs = catalog['decMean'] if 'decMean' in catalog.colnames else catalog['DEJ2000']
    catalog_coords = SkyCoord(ras, decs, unit=u.deg)
    target = catalog_coords.separation(target_coords).arcsec < 1.
    if target.sum() == 0:
        ras = np.append(catalog_coords.ra, target_coords.ra)
        decs = np.append(catalog_coords.dec, target_coords.dec)
        catalog_coords = SkyCoord(ras, decs)
        target = np.append(target, True)
    elif target.sum() > 1:
        logging.warning('catalog has multiple sources within 1" of the target')
    return catalog, catalog_coords, target


def update_wcs(wcs, p):
    wcs.wcs.crval += p[:2]
    c, s = np.cos(p[2]), np.sin(p[2])
    if wcs.wcs.has_cd():
        wcs.wcs.cd = wcs.wcs.cd @ np.array([[c, -s], [s, c]]) * p[3]
    else:
        wcs.wcs.pc = wcs.wcs.pc @ np.array([[c, -s], [s, c]]) * p[3]


def wcs_offset(p, radec, xy, origwcs):
    wcs = origwcs.deepcopy()
    update_wcs(wcs, p)
    test_xy = wcs.all_world2pix(radec, 0)
    rms = (np.sum((test_xy - xy) ** 2) / len(radec)) ** 0.5
    return rms


def refine_wcs(wcs, xy, radec):
    res = minimize(wcs_offset, [0., 0., 0., 1.], args=(radec, xy, wcs),
                   bounds=[(-0.01, 0.01), (-0.01, 0.01), (-0.01, 0.01), (0.9, 1.1)])

    orig_rms = wcs_offset([0., 0., 0., 1.], radec, xy, wcs)
    print(' orig_fun: {}'.format(orig_rms))
    print(res)
    update_wcs(wcs, res.x)


def read_and_refine_wcs(filepath, catalog_coords, use_astrometry_net=False, show=True):
    if use_astrometry_net:
        if filepath.endswith('.fz'):
            os.system(f'funpack {filepath}')
            filepath = filepath[:-3]
        os.system(f'solve-field -p --temp-axy -S none -M none -R none -W none -B none -O -U indx.xyls {filepath}'
                  ' && rm indx.xyls')
        os.system(f'mv {filepath.replace(".fits", ".new")} {filepath}')

    ccddata = CCDData.read(filepath, unit='adu', hdu='SCI')
    sources = fits.getdata(filepath, extname='CAT')
    ccddata.mask = fits.getdata(filepath, extname='BPM')
    ra, dec = ccddata.wcs.all_pix2world(sources['x'], sources['y'], 0)
    source_coords = SkyCoord(ra, dec, unit=u.deg)
    i, sep, _ = source_coords.match_to_catalog_sky(catalog_coords)
    n_hist, bins = np.histogram(sep.arcsec)
    i_peak = np.argmax(n_hist)
    match = (sep.arcsec > bins[i_peak]) & (sep.arcsec < bins[i_peak + 1])
    xy = np.array([sources['x'][match], sources['y'][match]]).T
    radec = np.array([catalog_coords.ra.deg[i[match]], catalog_coords.dec.deg[i[match]]]).T
    refine_wcs(ccddata.wcs, xy, radec)

    if show:
        plt.figure(figsize=(6., 6.))
        imshow_norm(ccddata.data, interval=ZScaleInterval(),
                    origin='lower' if ccddata.wcs.wcs.cd[1, 1] > 0. else 'upper')
        plt.axis('off')
        plt.axis('tight')
        plt.tight_layout(pad=0.)
        x, y = ccddata.wcs.all_world2pix(radec, 0).T
        plt.plot(x, y, ls='none', marker='o', mec='r', mfc='none')
        image_filename = os.path.basename(filepath).replace('.fz', '').replace('.fits', '.png')
        plt.savefig(os.path.join(image_dir, image_filename), overwrite=True)
        plt.savefig('latest_image.png', overwrite=True)
        plt.close()

    return ccddata


def extract_photometry(ccddata, catalog, catalog_coords, target, image_path=None, aperture_radius=2.*u.arcsec,
                       bg_radius_in=None, bg_radius_out=None):
    apertures = SkyCircularAperture(catalog_coords, aperture_radius)
    if bg_radius_in is not None and bg_radius_out is not None:
        apertures = [apertures, SkyCircularAnnulus(catalog_coords, bg_radius_in, bg_radius_out)]
    photometry = aperture_photometry(ccddata, apertures)
    if 'aperture_sum_1' in photometry.colnames:
        flux = photometry['aperture_sum_0'] - photometry['aperture_sum_1']
        dflux = (photometry['aperture_sum_err_0'] ** 2. + photometry['aperture_sum_err_1'] ** 2.) ** 0.5
    else:
        flux = photometry['aperture_sum']
        dflux = photometry['aperture_sum_err']
    photometry['aperture_mag'] = u.Magnitude(flux / ccddata.meta['exptime'])
    photometry['aperture_mag_err'] = 2.5 / np.log(10.) * dflux / flux
    photometry = hstack([catalog, photometry])
    photometry['zeropoint'] = photometry['catalog_mag'] - photometry['aperture_mag'].value
    zeropoints = photometry['zeropoint'][~target]
    if photometry.has_masked_values and photometry.mask['zeropoint'][~target].any():
        zeropoints = zeropoints.filled(np.nan)
    zp = np.nanmedian(zeropoints)
    zperr = mad_std(zeropoints, ignore_nan=True) / np.isfinite(zeropoints).sum() ** 0.5  # std error
    target_row = photometry[target][0]
    mag = target_row['aperture_mag'].value + zp
    dmag = (target_row['aperture_mag_err'].value ** 2. + zperr ** 2.) ** 0.5
    with open(lc_file, 'a') as f:
        f.write(f'{ccddata.meta["MJD-OBS"]:.5f} {mag:.3f} {dmag:.3f} {zp:.3f} {zperr:.3f} {ccddata.meta["FILTER"]:>6s} '
                f'{ccddata.meta["TELESCOP"]:>16s}\n')

    if image_path is not None:
        ax = plt.axes()
        mark = ',' if np.isfinite(photometry['aperture_mag']).sum() > 1000 else '.'
        ax.plot(photometry['aperture_mag'], photometry['catalog_mag'], ls='none', marker=mark, zorder=1,
                label='calibration stars')
        ax.plot(mag - zp, mag, ls='none', marker='*', zorder=3, label='target')
        yfit = np.array([21., 13.])
        xfit = yfit - zp
        ax.plot(xfit, yfit, label=f'$Z = {zp:.2f}$ mag', zorder=2)
        ax.set_xlabel('Instrumental Magnitude')
        ax.set_ylabel('AB Magnitude')
        ax.legend()
        plt.savefig(image_path, overwrite=True)
        plt.savefig('latest_cal.pdf', overwrite=True)
        plt.close()


def update_light_curve():
    t = Table.read('lc.txt', format='ascii')
    grouped = t.group_by(['filter', 'telescope'])
    ax = plt.axes()
    for f in grouped.groups:
        ax.errorbar(f['MJD'], f['mag'], f['dmag'], ls='none', label=f['telescope'][0] + ' ' + f['filter'][0],
                    color=plot_colors.get(f['filter'][0]), marker=plot_markers.get(f['telescope'][0]))
    ax.set_xlabel('MJD')
    ax.set_ylabel('Magnitude')
    ax.invert_yaxis()
    ax.legend(loc='best')
    plt.savefig('lc.pdf', overwrite=True)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Reduce images of Spikey and extract photometry')
    parser.add_argument('filenames', nargs='+', help='Filenames to process')
    parser.add_argument('--astrometry', action='store_true', help='Use astrometry.net to solve WCS')
    args = parser.parse_args()

    catalog0, catalog_coords0, target0 = load_catalog(ps1_catalog_path)

    for filepath in args.filenames:
        filename = os.path.basename(filepath)
        if 'bkg' in filename or 'mask' in filename or 'var' in filename:
            continue
        image_path = os.path.join(image_dir, filename.replace('.fz', '').replace('.fits', '_cal.pdf'))

        if 'elp' in filename:  # Las Cumbres image
            ccddata = read_and_refine_wcs(filepath, catalog_coords0, use_astrometry_net=args.astrometry)
            ccddata.mask |= ccddata.data > saturation
            ccddata.uncertainty = ccddata.data ** 0.5
            background = Background2D(ccddata, 1024)
            ccddata.data -= background.background
        elif '.flat.' in filename:  # MDM image processed by me
            ccddata = CCDData.read(filepath, unit='adu')
            ccddata.mask = fits.getdata(filepath.replace('.flat', '.mask'))
            ccddata.uncertainty = fits.getdata(filepath.replace('.flat', '.var')) ** 0.5
            ccddata.data -= fits.getdata(filepath.replace('.flat', '.bkg'))
        else:  # other MDM image
            ccddata = CCDData.read(filepath, unit='adu')
            ccddata.mask = ccddata.data > saturation
            ccddata.uncertainty = ccddata.data ** 0.5
            background = Background2D(ccddata, 1016)
            ccddata.data -= background.background

        if 'MJD-OBS' not in ccddata.meta and 'MJD' in ccddata.meta:
            ccddata.meta['MJD-OBS'] = ccddata.meta['MJD']
        if 'FILTER' not in ccddata.meta and 'FILTID2' in ccddata.meta:
            ccddata.meta['FILTER'] = ccddata.meta['FILTID2']

        if ccddata.meta['FILTER'][0] in 'grizy':  # use Pan-STARRS catalog
            catalog, catalog_coords, target = catalog0.copy(), catalog_coords0, target0
            catalog['catalog_mag'] = catalog[ccddata.meta['FILTER'][0] + 'MeanPSFMag']
        elif ccddata.meta['FILTER'][0] in 'BV':  # use APASS catalog
            catalog, catalog_coords, target = load_catalog(apass_catalog_path)
            catalog['catalog_mag'] = catalog[ccddata.meta['FILTER'].replace('p', '_') + 'mag']
        else:
            raise ValueError('no catalog for filter ' + ccddata.meta['FILTER'])

        extract_photometry(ccddata, catalog, catalog_coords, target, image_path=image_path)

    update_light_curve()
