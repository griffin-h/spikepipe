from astropy.coordinates import SkyCoord
from astropy.table import Table, hstack
from astropy.nddata import CCDData
from astropy.stats import mad_std
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from photutils import Background2D, SkyCircularAperture, aperture_photometry
import os
import logging

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

target_coords = SkyCoord('12:42:20.0183 -0:32:29.155', unit=(u.hourangle, u.deg))  # random star in L104
catalog_path = '/Users/griffin/Downloads/PS-1_27_2020.csv'
data_dir = '/Users/griffin/Downloads/lcogtdata-20200127-32/'

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

for filename in os.listdir(data_dir):
    filepath = os.path.join(data_dir + filename)
    ccddata = CCDData.read(filepath, unit='adu', hdu='SCI')
    background = Background2D(ccddata, bkg_box_size)
    ccddata.mask = ccddata.data > saturation
    ccddata.uncertainty = ccddata.data ** 0.5
    ccddata.data -= background.background

    photometry = aperture_photometry(ccddata, apertures)
    photometry['aperture_mag'] = u.Magnitude(photometry['aperture_sum'] / ccddata.meta['exptime'])
    photometry['aperture_mag_err'] = 2.5 / np.log(10.) * photometry['aperture_sum_err'] / photometry['aperture_sum']
    photometry = hstack([catalog, photometry])
    photometry['zeropoint'] = photometry[mag_kwd] - photometry['aperture_mag'].value
    zeropoints = photometry['zeropoint'][~target].filled(np.nan)
    ccddata.meta['ZP'] = np.nanmedian(zeropoints)
    ccddata.meta['ZPERR'] = mad_std(zeropoints, ignore_nan=True) / np.isfinite(zeropoints).sum() ** 0.5  # std error
    target_row = photometry[target][0]
    ccddata.meta['TARGMAG'] = target_row['aperture_mag'].value + ccddata.meta['ZP']
    ccddata.meta['TARGDMAG'] = (target_row['aperture_mag_err'].value ** 2. + ccddata.meta['ZPERR'] ** 2.) ** 0.5
    ccddata.write(filepath[:-3], overwrite=True)

    if make_diagnostic_plot:
        ax = plt.axes()
        ax.errorbar(photometry['aperture_mag'], photometry[mag_kwd], ls='none', marker='.', label='calibration stars')
        ax.scatter(ccddata.meta['TARGMAG'] - ccddata.meta['ZP'], ccddata.meta['TARGMAG'], marker='*', label='target')
        yfit = np.array([18., 10.])
        xfit = yfit - ccddata.meta['ZP']
        ax.plot(xfit, yfit, label=f'$Z = {ccddata.meta["ZP"]:.2f}$ mag')
        ax.set_xlabel('Instrumental Magnitude')
        ax.set_ylabel('AB Magnitude')
        ax.legend()
        plt.savefig(filepath[:-8] + '.pdf', overwrite=True)
        plt.close()

os.system(f'gethead {data_dir}/*.fits MJD-OBS TARGMAG TARGDMAG > lc.txt')
x = np.genfromtxt('lc.txt')
plt.errorbar(x[:, 1], x[:, 2], x[:, 3], marker='.', ls='none')
plt.xlabel('MJD')
plt.ylabel('Magnitude')
plt.savefig('lc.pdf')
