#!/usr/bin/env python
from LCOGTingest import get_metadata, download_frame
from spikepipe import *
from datetime import datetime
import os

catalog, catalog_coords, target = load_catalog(ps1_catalog_path)
today = datetime.utcnow().strftime('%Y-%m-%d')
frames = get_metadata(start=today, OBSTYPE='EXPOSE', RLEVEL=91, PROPID='DDT2020A-006')
for frame in frames:
    filename = download_frame(frame, 'data/')
    image_path = os.path.join(image_dir, os.path.basename(filename).replace('.fz', '').replace('.fits', '_cal.pdf'))
    ccddata = read_and_refine_wcs(filename, catalog_coords, use_astrometry_net=True)
    ccddata.mask |= ccddata.data > saturation
    ccddata.uncertainty = ccddata.data ** 0.5
    background = Background2D(ccddata, 1024)
    ccddata.data -= background.background
    catalog['catalog_mag'] = catalog[ccddata.meta['FILTER'][0] + 'MeanPSFMag']
    extract_photometry(ccddata, catalog, catalog_coords, target, image_path=image_path)
if frames:
    update_light_curve()
    os.system('mailx -s "Spikey Observation {}" '.format(today) +
              '-a latest_image.png -a latest_cal.pdf -a lc.pdf -a lc.txt ' +
              os.environ['SPIKEYPPL'] + ' < /dev/null')
