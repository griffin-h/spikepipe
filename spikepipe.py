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
from photutils import Background2D, SkyCircularAperture, aperture_photometry
import os
import logging

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

target_coords = SkyCoord('19:18:45.580 +49:37:56.03', unit=(u.hourangle, u.deg))
catalog_path = 'catalogs/spikey_catalog.csv'
data_dir = 'data'
image_dir = 'plots'
lc_file = 'lc.txt'

aperture_radius = 2. * u.arcsec
bkg_box_size = 1024
saturation = 50000.
mag_kwd = 'rMeanPSFMag'
make_diagnostic_plot = True

catalog = Table.read(catalog_path, format='ascii.csv', fill_values=[('-999.0', '0')])
catalog_coords = SkyCoord(catalog['raMean'], catalog['decMean'], unit=u.deg)
apertures = SkyCircularAperture(catalog_coords, aperture_radius)
target = catalog_coords.separation(target_coords).arcsec < 1.
if target.sum() == 0:
    logging.warning('target not in the catalog')
elif target.sum() > 1:
    logging.warning('catalog has multiple sources within 1" of the target')


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


def read_and_refine_wcs(filepath, show=False):
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
    radec = np.array([catalog[i[match]]['raMean'], catalog[i[match]]['decMean']]).T
    refine_wcs(ccddata.wcs, xy, radec)

    if show:
        imshow_norm(ccddata.data, interval=ZScaleInterval())
        x, y = ccddata.wcs.all_world2pix(radec, 0).T
        plt.plot(x, y, ls='none', marker='o', mec='r', mfc='none')
        plt.savefig(os.path.join(image_dir, os.path.basename(filepath)[:-8] + '.pdf'), overwrite=True)
        plt.savefig('latest_image.pdf', overwrite=True)
        plt.close()

    return ccddata


def reduce_one_image(filepath):
    ccddata = read_and_refine_wcs(filepath, show=make_diagnostic_plot)
    background = Background2D(ccddata, bkg_box_size)
    ccddata.mask |= ccddata.data > saturation
    ccddata.uncertainty = ccddata.data ** 0.5
    ccddata.data -= background.background

    photometry = aperture_photometry(ccddata, apertures)
    photometry['aperture_mag'] = u.Magnitude(photometry['aperture_sum'] / ccddata.meta['exptime'])
    photometry['aperture_mag_err'] = 2.5 / np.log(10.) * photometry['aperture_sum_err'] / photometry['aperture_sum']
    photometry = hstack([catalog, photometry])
    photometry['zeropoint'] = photometry[mag_kwd] - photometry['aperture_mag'].value
    zeropoints = photometry['zeropoint'][~target]
    if photometry.masked:
        zeropoints = zeropoints.filled(np.nan)
    zp = np.nanmedian(zeropoints)
    zperr = mad_std(zeropoints, ignore_nan=True) / np.isfinite(zeropoints).sum() ** 0.5  # std error
    target_row = photometry[target][0]
    mag = target_row['aperture_mag'].value + zp
    dmag = (target_row['aperture_mag_err'].value ** 2. + zperr ** 2.) ** 0.5
    with open(lc_file, 'a') as f:
        f.write(f'{ccddata.meta["MJD-OBS"]:.5f} {mag:.2f} {dmag:.2f} {zp:.2f} {zperr:.2f}\n')

    if make_diagnostic_plot:
        ax = plt.axes()
        ax.plot(photometry['aperture_mag'], photometry[mag_kwd], ls='none', marker=',', zorder=1,
                label='calibration stars')
        ax.plot(mag - zp, mag, ls='none', marker='*', zorder=3, label='target')
        yfit = np.array([21., 16.])
        xfit = yfit - zp
        ax.plot(xfit, yfit, label=f'$Z = {zp:.2f}$ mag', zorder=2)
        ax.set_xlabel('Instrumental Magnitude')
        ax.set_ylabel('AB Magnitude')
        ax.legend()
        image_path = os.path.join(image_dir, os.path.basename(filepath)[:-8] + '_cal.pdf')
        plt.savefig(image_path, overwrite=True)
        plt.savefig('latest_cal.pdf', overwrite=True)
        plt.close()


def update_light_curve():
    t = Table.read('lc.txt', format='ascii')
    ax = plt.axes()
    ax.errorbar(t['MJD'], t['mag'], t['dmag'], marker='.', ls='none')
    ax.set_xlabel('MJD')
    ax.set_ylabel('Magnitude')
    ax.invert_yaxis()
    plt.savefig('lc.pdf', overwrite=True)
    plt.close()


if __name__ == '__main__':
    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        reduce_one_image(filename)
    update_light_curve()
