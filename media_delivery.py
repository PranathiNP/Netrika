"""Browser-compatible video encoding and authenticated range delivery."""
import re
import subprocess
from pathlib import Path

from fastapi.responses import Response, StreamingResponse


def browser_video(source, destination):
    import imageio_ffmpeg
    try:
        subprocess.run([
            imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error',
            '-nostdin', '-y', '-i', str(source), '-map', '0:v:0', '-an',
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
            '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart', str(destination),
        ], check=True, capture_output=True, timeout=120)
    except (subprocess.SubprocessError, OSError) as error:
        raise RuntimeError('Could not encode the browser video') from error
    if not Path(destination).is_file() or Path(destination).stat().st_size == 0:
        raise RuntimeError('The video encoder produced no output')


def stored_media_response(stored, request):
    size = stored.length
    headers = {'Accept-Ranges': 'bytes', 'Cache-Control': 'private, no-store',
               'Content-Disposition': f'inline; filename="{stored.filename}"'}
    start, end, status = 0, size - 1, 200
    raw = request.headers.get('range', '')
    # Ignore multipart ranges and unsupported units; send the complete resource.
    if request.method != 'HEAD' and raw.startswith('bytes=') and ',' not in raw:
        try:
            match = re.fullmatch(r'bytes=(\d*)-(\d*)', raw)
            if not match or not any(match.groups()):
                raise ValueError
            first, last = match.groups()
            if first:
                start = int(first)
                end = min(int(last), size - 1) if last else size - 1
            else:
                suffix = int(last)
                if suffix <= 0:
                    raise ValueError
                start = max(0, size - suffix)
            if start >= size or start > end:
                raise ValueError
        except ValueError:
            stored.close()
            return Response(status_code=416, headers={**headers, 'Content-Range': f'bytes */{size}'})
        status = 206
        headers['Content-Range'] = f'bytes {start}-{end}/{size}'
    headers['Content-Length'] = str(max(0, end - start + 1))
    content_type = stored.content_type
    if request.method == 'HEAD':
        stored.close()
        return Response(headers=headers, media_type=content_type)
    stored.seek(start)

    def chunks():
        remaining = end - start + 1
        try:
            while remaining > 0:
                block = stored.read(min(64 * 1024, remaining))
                if not block:
                    break
                remaining -= len(block)
                yield block
        finally:
            stored.close()
    return StreamingResponse(chunks(), status_code=status, media_type=content_type, headers=headers)
