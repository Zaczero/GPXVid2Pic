import asyncio
import math
import os
import pickle
import sys
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Coroutine, NoReturn

import numpy as np
import xmltodict
from exiftool import ExifTool
from geopy.distance import geodesic
from skimage import color, io, measure

from gpx_merge import gpx_merge

FPS_POWER = 2  # Power of 2 for FPS calculation
FPS = 2 ** FPS_POWER

# Extra filters for dashcam
DASHCAM_FRONT_EXTRA = 'crop=x=0:y=ih*10/100:w=iw:h=ih*85/100'
DASHCAM_REAR_EXTRA = 'crop=x=0:y=0:w=iw:h=ih*90/100'

# Consecutive points distance thresholds
POINT_DST_MIN = 1.85
POINT_DST_TARGET = 0  # 5
POINT_DST_MAX = 8

BLURRY_THRESHOLD = 100


def detect_dashcam(path: Path) -> bool:
    return next(path.iterdir()).name.startswith('202')


def is_dashcam_front(path_str: str) -> bool:
    return '_f' in path_str.lower()


def is_blurry(image_path: str) -> bool:
    image = io.imread(image_path)

    src_height = image.shape[0]
    src_width = image.shape[1]
    dst_size = src_height // 2
    dst_height_padding = dst_size // 2
    dst_width_padding = (src_width - dst_size) // 2

    # Crop the image to the central region
    image = image[
        dst_height_padding:dst_height_padding + dst_size,
        dst_width_padding:dst_width_padding + dst_size, :]

    image_gray = color.rgb2gray(image)
    image_blur = measure.blur_effect(image_gray)

    return image_blur < BLURRY_THRESHOLD


async def main() -> NoReturn:
    # Set up paths and directories
    target = sorted((p for p in Path('/home/user/Videos/Car').iterdir() if p.is_dir()))[-1]
    target_dir = Path(target)
    print(f'Target: {target}')

    gpx_path = gpx_merge(target_dir)
    src_dir = target_dir / Path('src')
    dst_dir = target_dir / Path('dst')
    dst_f_dir = target_dir / Path('dst_f')
    dst_r_dir = target_dir / Path('dst_r')
    state_dir = target_dir / Path('state')

    # Create source directory if it doesn't exist
    if not src_dir.exists():
        src_dir.mkdir()

        for pattern in ['*.*4']:
            for target_path in target_dir.glob(pattern):
                target_path.rename(src_dir / target_path.name)

    # Detect dashcam and create destination directories
    dashcam = detect_dashcam(src_dir)
    print(f'Dashcam: {"✅ Yes" if dashcam else "🚫 No"}')

    for target_dir in [dst_f_dir, dst_r_dir] if dashcam else [dst_dir]:
        target_dir.mkdir(exist_ok=True)

    state_dir.mkdir(exist_ok=True)

    with ExifTool() as et:
        def read_tag(path: Path, tag: str) -> str:
            return et.execute('-b', '-m', f'-{tag}', '-api', 'LargeFileSupport=1', str(path))

        async def export_images(src_path: Path, current_dst_dir: Path, dashcam: bool):
            dst_path = current_dst_dir / src_path.with_suffix('').name
            ts_path = state_dir / src_path.with_suffix('.pkl').name

            # Skip if timestamp file already exists (i.e. already exported)
            if ts_path.exists():
                return

            filters = []
            filters.append(f'atadenoise=s=15')
            filters.append(f'fps={FPS}')

            if dashcam:
                filters.append(DASHCAM_FRONT_EXTRA if is_dashcam_front(str(src_path)) else DASHCAM_REAR_EXTRA)

            filters.append(f'eq=contrast=1.05:saturation=1.05')
            filters.append(f'unsharp=lx=13:ly=13:la=1')
            filters_str = ','.join(filters)

            print(f'Exporting {src_path.name}…')

            # Execute FFmpeg subprocess to apply filters and export images
            proc = await asyncio.create_subprocess_exec(
                'ffmpeg',
                '-v', 'fatal',
                '-hwaccel', 'cuda',
                # '-hwaccel_output_format', 'cuda',
                '-i', str(src_path),
                '-filter:v', filters_str,
                '-q:v', '2',
                str(dst_path) + '_%05d.jpg',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE)

            await proc.wait()

            stdout = await proc.stdout.read()
            stderr = await proc.stderr.read()

            if stderr:
                print(stderr.decode(), file=sys.stderr)
                raise RuntimeError(f'ffmpeg returned an error for {src_path.name}')

            # Read metadata tags from source file
            projection_type = read_tag(src_path, 'ProjectionType')
            ts = {}

            # Get the creation date of the video frame
            # NOTE: this should be UTC
            date = read_tag(src_path, 'CreateDate')

            frame_mtime = datetime.strptime(date, '%Y:%m:%d %H:%M:%S') \
                .replace(tzinfo=timezone.utc) \
                .timestamp()

            if dashcam:
                frame_mtime -= int(float(read_tag(src_path, 'MediaDuration')))
            else:
                frame_mtime -= 3600

            # Update the timestamp of each exported image file
            for i, dst_frame_path in enumerate(sorted(current_dst_dir.glob(dst_path.name + '_*'))):
                ts[str(dst_frame_path)] = dst_frame_mtime = frame_mtime + i / FPS
                date = datetime.utcfromtimestamp(dst_frame_mtime).strftime('%Y:%m:%d %H:%M:%S.%f')

                et.execute('-overwrite_original',
                           f'-ProjectionType={projection_type}',
                           f'-SubSecDateTimeOriginal={date}+00:00',
                           str(dst_frame_path))

            with open(ts_path, 'wb') as f:
                pickle.dump(ts, f, protocol=pickle.HIGHEST_PROTOCOL)

        futures: list[Coroutine] = []

        # Export images and update timestamps for each source file
        for src_path in sorted(src_dir.glob('*')):
            current_dst_dir = (dst_f_dir if is_dashcam_front(str(src_path)) else dst_r_dir) if dashcam else dst_dir
            futures.append(export_images(src_path, current_dst_dir, dashcam))

        tasks_limit = 4
        tasks: list[asyncio.Task] = []

        # Limit the number of concurrent tasks and wait for them to complete
        while tasks or futures:
            for _ in range(min(tasks_limit - len(tasks), len(futures))):
                tasks.append(asyncio.create_task(futures.pop()))

            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

            for task in done:
                if task.exception():
                    raise task.exception()

            tasks = list(pending)

        while True:
            # Read GPX file and parse track segments
            with open(gpx_path) as f:
                points = xmltodict.parse(f.read())['gpx']['trk']['trkseg']['trkpt']

            # Convert string values to float and timestamps to seconds
            for p in points:
                p['@lat'] = float(p['@lat'])
                p['@lon'] = float(p['@lon'])
                p['ts'] = datetime.strptime(p['time'], '%Y-%m-%dT%H:%M:%SZ') \
                    .astimezone(UTC) \
                    .timestamp()

            # Interpolate timestamps for invalid timestamps
            for a, b, c, d in zip(points, points[1:], points[2:], points[3:]):
                if b['ts'] < c['ts']:
                    continue

                _, b_ts, c_ts, _ = np.linspace(a['ts'], d['ts'], 4)
                b['ts'] = b_ts
                c['ts'] = c_ts

            i = 0

            # Remove points that are too close together
            while i < len(points) - 2:
                p = points[i]
                pn = points[i + 1]
                pnn = points[i + 2]

                d12 = distance(p, pn)
                d23 = distance(pn, pnn)
                d13 = distance(p, pnn)

                if d12 < POINT_DST_MIN or d12 + d23 < POINT_DST_TARGET:
                    points.pop(i + 1)
                elif d13 < POINT_DST_MIN * 1.5:
                    points.pop(i + 2)
                else:
                    i += 1

            # Interpolate points that are too far apart
            for (i, p), pn in reversed([*zip(enumerate(points), points[1:])]):
                dst = distance(p, pn)
                power = 0

                while POINT_DST_MAX < dst and power < FPS_POWER:
                    dst /= 2
                    power += 1

                if power:
                    ts = np.linspace(p['ts'], pn['ts'], 2 ** power + 1)
                    xs = np.interp(ts, (ts[0], ts[-1]), (p['@lat'], pn['@lat']))
                    ys = np.interp(ts, (ts[0], ts[-1]), (p['@lon'], pn['@lon']))

                    for x, y, t in reversed([*zip(xs, ys, ts)][1:-1]):
                        points.insert(i + 1, {
                            '@lat': x,
                            '@lon': y,
                            'ts': t
                        })

            state_ts = {}

            # Load timestamp information from state files
            for ts_path in state_dir.glob('*.pkl'):
                with open(ts_path, 'rb') as f:
                    state_ts |= pickle.load(f)

            rear_keys = set(filter(lambda kk: dashcam and not is_dashcam_front(kk), state_ts))
            other_keys = set(state_ts) - rear_keys

            # Rename image files so that they are sorted by timestamp
            for keys in [rear_keys, other_keys]:
                for i, k in enumerate(sorted(keys)):
                    k_path = Path(k)
                    k_new = k_path.with_name(f'{i:05d}{k_path.suffix}')

                    state_ts[str(k_new)] = state_ts[k]
                    del state_ts[k]

                    if k_path.exists():
                        os.rename(k_path, k_new)

            first_point_ts = points[0]['ts']
            first_state_ts = min(state_ts.values())
            first_offset = first_point_ts - first_state_ts
            first_offset_m, first_offset_s = divmod(first_offset, 60)
            print(f'First offset: {first_offset_m:01.0f}:{first_offset_s:02.1f}')

            # Ask the user to select a time offset
            offset = parse_time_offset(input('Enter time offset ([-]M:S[:MS]): '))

            if offset is None:
                continue

            # Update timestamps with the given offset
            for k in state_ts:
                state_ts[k] += offset

            # Assign image paths to each point based on the closest timestamp
            for p in points:
                p['img_path'] = []

            for current_state_items in [
                {k: v for k, v in state_ts.items() if filter_fn(k)}.items()
                for filter_fn
                in [is_dashcam_front, lambda s: not is_dashcam_front(s)]
            ] if dashcam else [state_ts.items()]:
                for p in points:
                    best_k, best_v = min(current_state_items, key=lambda kv: abs(kv[1] - p['ts']))

                    if abs(best_v - p['ts']) <= 1.1 / FPS:
                        p['img_path'].append(best_k)

            # Remove points without image paths
            points = [p for p in points if p['img_path']]

            # Calculate the direction for each point
            for p, pn in zip(points, points[1:]):
                p['direction'] = bearing(p, pn)

            points[-1]['direction'] = points[-2]['direction']

            print(f'Writing EXIF data…')

            # Clear EXIF metadata from all images
            for target_dir in [dst_f_dir, dst_r_dir] if dashcam else [dst_dir]:
                et.execute('-overwrite_original',
                           '-ImageDescription=',
                           '-GPSLatitude=',
                           '-GPSLatitudeRef=',
                           '-GPSLongitude=',
                           '-GPSLongitudeRef=',
                           '-GPSImgDirection=',
                           '-GPSImgDirectionRef=',
                           '-SubSecDateTimeOriginal=',
                           '-r', str(target_dir))

            p_ts_min = points[0]['ts']
            p_ts_max = points[-1]['ts']

            # Update EXIF metadata for assigned images
            for p in points:
                for img_path in p['img_path']:
                    p_lat = p['@lat']
                    p_lon = p['@lon']
                    p_ts = p['ts']
                    p_direction = p['direction']

                    # Adjust position and direction for rear camera
                    if dashcam and not is_dashcam_front(img_path):
                        move_meters = 1
                        move_degrees = -move_meters / 111_320
                        p_lat += move_degrees * math.cos(math.radians(p_direction))
                        p_lon += move_degrees * math.sin(math.radians(p_direction))

                        p_direction = (p_direction + 180) % 360

                    if dashcam and not is_dashcam_front(img_path):
                        p_ts = p_ts_max - (p_ts - p_ts_min)

                    date = datetime.utcfromtimestamp(p_ts).strftime('%Y:%m:%d %H:%M:%S.%f')

                    et.execute('-overwrite_original',
                               f'-GPSLatitude={p_lat:.7f}',
                               '-GPSLatitudeRef=N',
                               f'-GPSLongitude={p_lon:.7f}',
                               '-GPSLongitudeRef=E',
                               f'-GPSImgDirection={p_direction}',
                               '-GPSImgDirectionRef=M',
                               f'-SubSecDateTimeOriginal={date}+00:00',
                               img_path)

            if points:
                print('✅ Done! Done! Done!')
            else:
                print('🚨 No images were updated.')


def parse_time_offset(text: str) -> float | None:
    try:
        arr = text.strip().split(':')

        m = int(arr[0])
        s = int(arr[1])
        ms = int(arr[2]) if len(arr) == 3 else 0

        positive = arr[0][0] != '-'
        m = abs(m)

        return (60 * m + s + ms / 10) * (1 if positive else -1)

    except Exception:
        return None


def distance(p1: dict, p2: dict) -> float:
    return geodesic((p1['@lat'], p1['@lon']), (p2['@lat'], p2['@lon'])).m


def bearing(p1: dict, p2: dict) -> float:
    lat1, long1 = p1['@lat'], p1['@lon']
    lat2, long2 = p2['@lat'], p2['@lon']
    d_lon = (long2 - long1)
    x = math.cos(math.radians(lat2)) * math.sin(math.radians(d_lon))
    y = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - \
        math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(math.radians(d_lon))
    bearing = math.atan2(x, y)  # use atan2 to determine the quadrant
    bearing = math.degrees(bearing)

    return bearing


if __name__ == '__main__':
    asyncio.run(main())
