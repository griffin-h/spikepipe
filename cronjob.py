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
    filepath = os.path.join(data_dir, filename)
    image_path = os.path.join(image_dir, filename.replace('.fz', '').replace('.fits', '_cal.pdf'))
    ccddata = read_and_refine_wcs(filepath, catalog_coords)
    catalog['catalog_mag'] = catalog[ccddata.meta['FILTER'][0] + 'MeanPSFMag']
    extract_photometry(ccddata, catalog, catalog_coords, target, image_path=image_path)
if frames:
    update_light_curve()
    os.system('mailx -s "Spikey Observation {}" '.format(today) +
              '-a latest_image.pdf -a latest_cal.pdf -a lc.pdf -a lc.txt ' +
              os.environ['SPIKEYPPL'] + ' < /dev/null')
