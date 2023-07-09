from datetime import UTC, datetime
from operator import itemgetter
from pathlib import Path

import xmltodict


def gpx_merge(dir: Path) -> Path:
    merged_path = dir / 'merged.gpx'

    if merged_path.is_file():
        return merged_path

    merged: dict[datetime, dict] = {}

    for gpx_path in dir.glob("*.gpx"):
        if not gpx_path.is_file():
            continue

        data = xmltodict.parse(gpx_path.read_text(),
                               force_list=('trk', 'trkseg', 'trkpt'))

        for trk in data['gpx']['trk']:
            for trkseg in trk['trkseg']:
                for trkpt in trkseg['trkpt']:
                    date = datetime \
                        .strptime(trkpt['time'], '%Y-%m-%dT%H:%M:%SZ') \
                        .astimezone(UTC)

                    merged[date] = trkpt

    merged_trkpt = map(itemgetter(1), sorted(merged.items(), key=itemgetter(0)))
    merged_data = {
        'gpx': {
            'trk': [{
                'trkseg': [{
                    'trkpt': merged_trkpt
                }]
            }]
        }
    }

    merged_path.write_text(xmltodict.unparse(merged_data, pretty=True))

    return merged_path
