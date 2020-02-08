#!/usr/bin/env python
from LCOGTingest import get_metadata, download_frame
from spikepipe import reduce_one_image, update_light_curve
from datetime import datetime
import os

today = datetime.utcnow().strftime('%Y-%m-%d')
frames = get_metadata(start=today, OBSTYPE='EXPOSE', RLEVEL=91, PROPID='DDT2020A-006')
for frame in frames:
    filename = download_frame(frame, 'data/')
    reduce_one_image(filename)
if frames:
    update_light_curve()
    os.system('mailx -s "Spikey Observation {}" '.format(today) +
              '-a latest_image.pdf -a latest_cal.pdf -a lc.pdf -a lc.txt ' +
              os.environ['SPIKEYPPL'] + ' < /dev/null')
