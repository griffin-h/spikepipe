#!/usr/bin/env python
from LCOGTingest import get_metadata, download_frame
from spikepipe import *
from datetime import datetime
import os

catalog, catalog_coords, target = load_catalog(PS1_CATALOG_PATH, TARGET_COORDS)
today = datetime.utcnow().strftime('%Y-%m-%d')
frames = get_metadata(start=today, OBSTYPE='EXPOSE', RLEVEL=91, PROPID=PROPID)
for frame in frames:
    filepath = download_frame(frame, 'data/')
    filename = os.path.basename(filepath)
    plot_path = os.path.join(IMAGE_DIR, filename.replace('.fz', '').replace('.fits', '.png'))
    image_path = os.path.join(IMAGE_DIR, filename.replace('.fz', '').replace('.fits', '_cal.pdf'))
    ccddata = preprocess_lco_image(filepath, catalog_coords, use_astrometry_net=True)
    catalog['catalog_mag'] = catalog[ccddata.meta['FILTER'][0] + 'MeanPSFMag']
    results = extract_photometry(ccddata, catalog, catalog_coords, target, plot_path=plot_path, image_path=image_path)
    results['filename'] = filename
    update_light_curve(LC_FILE, results)
if frames:
    email_cmd = 'mailx -s "Spikey Observation {}" -a latest_image.png -a latest_cal.pdf -a {} -a {} {} < /dev/null'
    os.system(email_cmd.format(today, LC_FILE.replace('.txt', '.pdf'), LC_FILE, os.environ['SPIKEYPPL']))
