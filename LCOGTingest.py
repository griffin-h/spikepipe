#!/usr/bin/env python
import requests
import os
from glob import glob
import logging

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)


def authenticate():
    """Get the authentication token"""
    username = os.environ['LCOUNAME']
    password = os.environ['LCOPASSWD']
    response = requests.post('https://archive-api.lco.global/api-token-auth/',
                             data={'username': username, 'password': password}).json()
    token = response.get('token')
    if token is None:
        raise Exception('Authentication failed with username {}'.format(username))
    else:
        authtoken = {'Authorization': 'Token ' + token}
    return authtoken


def get_metadata(**kwargs):
    """Get the list of files meeting criteria in kwargs"""
    authtoken = authenticate()
    url = 'https://archive-api.lco.global/frames/?' + '&'.join(
            [key + '=' + str(val) for key, val in kwargs.items() if val is not None])
    url = url.replace('False', 'false')
    url = url.replace('True', 'true')
    logging.info(url)

    response = requests.get(url, headers=authtoken, stream=True).json()
    frames = response['results']
    while response['next']:
        logging.info(response['next'])
        response = requests.get(response['next'], headers=authtoken, stream=True).json()
        frames += response['results']
    return frames


def download_frame(frame, dest='.', force=False):
    """Download a single image from the LCO archive and put it in the right directory"""
    os.makedirs(dest, exist_ok=True)
    filename = os.path.join(dest, frame['filename'])

    matches = glob(filename)
    if not matches or force:
        logging.info('downloading {}'.format(filename))
        with open(filename, 'wb') as f:
            f.write(requests.get(frame['url']).content)
    else:
        matches_filenames = [os.path.basename(fullpath) for fullpath in matches]
        if filename not in matches_filenames:
            filename = matches_filenames[0]
        logging.info('{} already exists'.format(filename))

    if os.path.isfile(filename) and os.stat(filename).st_size == 0:
        logging.warning('{} has size 0. Redownloading.'.format(filename))
        filename = frame['filename']
        with open(filename, 'wb') as f:
            f.write(requests.get(frame['url']).content)

    return filename


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description='Downloads data from archive.lco.global')
    parser.add_argument("-d", "--destination", default='.', help="destination directory for the downloaded files")
    parser.add_argument("-F", "--force-dl", action="store_true", help="download files even if they already exist")

    parser.add_argument("-S", "--site", choices=['bpl', 'coj', 'cpt', 'elp', 'lsc', 'ogg', 'sqa', 'tfn'])
    parser.add_argument("-T", "--telescope", choices=['0m4a', '0m4b', '0m4c', '0m8a', '1m0a', '2m0a'])
    parser.add_argument("-I", "--instrument")
    parser.add_argument("-f", "--filter", choices=['up', 'gp', 'rp', 'ip', 'zs', 'U', 'B', 'V', 'R', 'I'])
    parser.add_argument("-P", "--proposal", help="proposal ID (PROPID in the header)")
    parser.add_argument("-n", "--name", help="target name")
    parser.add_argument("-s", "--start", help="start date")
    parser.add_argument("-e", "--end", help="end date")

    parser.add_argument("-t", "--obstype", default='EXPOSE', choices=['ARC', 'BIAS', 'CATALOG', 'DARK', 'EXPERIMENTAL',
                                                                      'EXPOSE', 'LAMPFLAT', 'SKYFLAT', 'SPECTRUM',
                                                                      'STANDARD'])
    parser.add_argument("-r", "--reduction", default=91, type=int, choices=[0, 10, 11, 90, 91],
                        help="reduction state of the data: 0 = raw, 10-11 = quicklook, 90-91 = reduced")
    parser.add_argument("--public", action='store_true', help="include public data")

    args = parser.parse_args()

    frames = get_metadata(SITEID=args.site, TELID=args.telescope, INSTRUME=args.instrument, FILTER=args.filter,
                          PROPID=args.proposal, OBJECT=args.name, start=args.start, end=args.end, OBSTYPE=args.obstype,
                          RLEVEL=args.reduction, public=args.public)

    for frame in frames:
        filename = download_frame(frame, args.destination, args.force_dl)
