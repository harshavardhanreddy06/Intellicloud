import os
import sys
import json
import random
import shutil
import subprocess
import zipfile
import psutil
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def run_ffmpeg(*args, check=True):
    """Run ffmpeg command, return (returncode, stdout, stderr)."""
    cmd = ["ffmpeg", "-y"] + list(args)
    print("ffmpeg:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stderr[-2000:] if r.stderr else "")
    if check and r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed (code {r.returncode}): {r.stderr[-500:]}")
    return r


def ffprobe_duration(path):
    """Return video duration in seconds via ffprobe."""
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True
    )
    try:
        return float(r.stdout.strip())
    except Exception:
        return 30.0   # safe fallback


# ─────────────────────────────────────────────────────────────────────────────
#  IMAGE TASKS  (10)
# ─────────────────────────────────────────────────────────────────────────────

def process_image(task_type, input_path, params, output_path):
    print(f"Processing IMAGE task: {task_type}")
    img = Image.open(input_path)

    if task_type == 'img_resize':
        w = int(params.get('width', 800))
        h = int(params.get('height', 600))
        img = img.resize((w, h), Image.LANCZOS)
        img.convert("RGB").save(output_path, format='JPEG')

    elif task_type == 'img_cropping':
        l = int(params.get('left', 0))
        t = int(params.get('top', 0))
        w = int(params.get('width', min(400, img.width)))
        h = int(params.get('height', min(400, img.height)))
        img = img.crop((l, t, min(l + w, img.width), min(t + h, img.height)))
        img.convert("RGB").save(output_path, format='JPEG')

    elif task_type == 'img_compression':
        quality = int(params.get('quality', 80))
        img.convert("RGB").save(output_path, format='JPEG', quality=quality, optimize=True)

    elif task_type == 'img_format_conv':
        fmt = params.get('target_format', 'webp').lower().strip()
        fmt_map = {'jpg': 'JPEG', 'jpeg': 'JPEG', 'png': 'PNG', 'webp': 'WEBP', 'bmp': 'BMP'}
        pil_fmt = fmt_map.get(fmt, 'WEBP')
        img.convert("RGB").save(output_path, format=pil_fmt)

    elif task_type == 'img_watermark':
        text    = params.get('text', 'IntelliCloud 2026')
        opacity = float(params.get('opacity', 0.5))
        base    = img.convert("RGBA")
        overlay = Image.new('RGBA', base.size, (0, 0, 0, 0))
        draw    = ImageDraw.Draw(overlay)
        alpha   = int(255 * opacity)
        positions = [(20, 20), (base.width//2 - 80, base.height//2 - 15),
                     (base.width - 220, base.height - 40)]
        for pos in positions:
            draw.text(pos, text, fill=(255, 255, 0, alpha))
        result = Image.alpha_composite(base, overlay).convert("RGB")
        result.save(output_path, format='JPEG')

    elif task_type == 'img_puzzle_split':
        tiles_str = str(params.get('tiles', '2x2')).lower()
        try:
            cols, rows = [int(x) for x in tiles_str.split('x')]
        except Exception:
            cols, rows = 2, 2
        cols, rows = max(1, min(cols, 8)), max(1, min(rows, 8))
        tw = img.width  // cols
        th = img.height // rows
        pieces = [img.crop((c*tw, r*th, (c+1)*tw, (r+1)*th))
                  for r in range(rows) for c in range(cols)]
        random.shuffle(pieces)
        new_img = Image.new("RGB", (img.width, img.height))
        for idx, piece in enumerate(pieces):
            r, c = divmod(idx, cols)
            new_img.paste(piece, (c*tw, r*th))
        new_img.save(output_path, format='JPEG')

    elif task_type == 'img_color_corr':
        sat        = float(params.get('saturation', 1.3))
        brightness = float(params.get('brightness', 1.1))
        img = ImageEnhance.Color(img.convert("RGB")).enhance(sat)
        img = ImageEnhance.Brightness(img).enhance(brightness)
        img.save(output_path, format='JPEG')

    elif task_type == 'img_bg_removal':
        # Naïve white-bg removal using alpha threshold
        rgba = img.convert("RGBA")
        data = rgba.getdata()
        new_data = []
        for r, g, b, a in data:
            if r > 230 and g > 230 and b > 230:
                new_data.append((255, 255, 255, 0))
            else:
                new_data.append((r, g, b, a))
        rgba.putdata(new_data)
        out_png = output_path.rsplit('.', 1)[0] + '.png'
        rgba.save(out_png, format='PNG')
        if out_png != output_path:
            shutil.move(out_png, output_path)

    elif task_type == 'img_annotation':
        msg  = params.get('msg', 'Processed by IntelliCloud')
        base = img.convert("RGB")
        draw = ImageDraw.Draw(base)
        bh   = 40
        draw.rectangle([(0, base.height - bh), (base.width, base.height)], fill=(20, 20, 20))
        draw.text((10, base.height - bh + 10), msg, fill=(0, 220, 120))
        base.save(output_path, format='JPEG')

    elif task_type == 'img_batch_rename':
        shutil.copy2(input_path, output_path)

    else:
        shutil.copy2(input_path, output_path)

    print(f"Image saved: {output_path}")


def ffprobe_resolution(path):
    """Return (width, height) via ffprobe."""
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
         "-of", "csv=s=x:p=0", path],
        capture_output=True, text=True
    )
    try:
        w, h = r.stdout.strip().split('x')
        return int(w), int(h)
    except Exception:
        return 640, 480


# ─────────────────────────────────────────────────────────────────────────────
#  VIDEO TASKS  (10)  — all ffmpeg-based for maximum reliability
# ─────────────────────────────────────────────────────────────────────────────

def process_video(task_type, input_path, params, output_path):
    print(f"Processing VIDEO task: {task_type}")
    
    # 🔍 Initial Check: Try to open with ffprobe to catch corruption early
    check_r = subprocess.run(["ffprobe", "-v", "error", input_path], capture_output=True, text=True)
    if "moov atom not found" in check_r.stderr or "Invalid data found" in check_r.stderr:
        raise RuntimeError("CORRUPTED_VIDEO: The moov atom is missing. This file was likely incompletely uploaded or is corrupted.")

    duration = ffprobe_duration(input_path)
    vid_w, vid_h = ffprobe_resolution(input_path)
    print(f"Detected: {duration:.1f}s, {vid_w}x{vid_h}")

    if task_type == 'vid_trimming':
        start = float(params.get('start', 0))
        end   = min(float(params.get('end', 10)), duration)
        run_ffmpeg("-ss", str(start), "-to", str(end),
                   "-i", input_path, "-c", "copy", "-map", "0", output_path)

    elif task_type == 'vid_compression':
        # Default to a decent CRF for quality/size balance
        bitrate = params.get('bitrate')
        if bitrate:
            target_bitrate = str(bitrate) + 'M'
            run_ffmpeg("-i", input_path, "-vcodec", "libx264", "-b:v", target_bitrate,
                       "-acodec", "aac", "-strict", "-2", "-preset", "fast", output_path)
        else:
            # High efficiency compression using CRF
            run_ffmpeg("-i", input_path, "-vcodec", "libx264", "-crf", "28",
                       "-acodec", "aac", "-strict", "-2", "-preset", "fast", output_path)

    elif task_type == 'vid_remove_audio':
        run_ffmpeg("-i", input_path, "-an", "-vcodec", "copy", output_path)

    elif task_type == 'vid_cropping':
        # Default values from UI can be 800x600
        # We must cap them to the actual video resolution
        x  = max(0, min(int(params.get('left', 0)), vid_w - 1))
        y  = max(0, min(int(params.get('top', 0)), vid_h - 1))
        w  = max(1, min(int(params.get('w', 640)), vid_w - x))
        h  = max(1, min(int(params.get('h', 480)), vid_h - y))
        
        print(f"Capping crop to: {w}x{h} at {x},{y}")
        run_ffmpeg("-i", input_path,
                   "-filter:v", f"crop={w}:{h}:{x}:{y}",
                   "-acodec", "copy", output_path)

    elif task_type == 'vid_add_subtitles':
        text = params.get('text', 'IntelliCloud')
        run_ffmpeg("-i", input_path,
                   "-vf", f"drawtext=text='{text}':fontsize=36:fontcolor=white:"
                          f"x=(w-text_w)/2:y=h-50:box=1:boxcolor=black@0.5",
                   "-acodec", "copy", output_path)

    elif task_type == 'vid_format_conv':
        target = params.get('target', 'mp4').lower()
        out = output_path.rsplit('.', 1)[0] + '.' + target
        run_ffmpeg("-i", input_path, "-c", "copy", out)
        if out != output_path:
            shutil.move(out, output_path)

    elif task_type == 'vid_watermarking':
        text = params.get('text', 'CLOUD_ID_01')
        run_ffmpeg("-i", input_path,
                   "-vf", f"drawtext=text='{text}':fontsize=28:fontcolor=white@0.7:"
                          f"x=w-tw-20:y=20",
                   "-acodec", "copy", output_path)

    elif task_type == 'vid_frame_extraction':
        interval = int(params.get('every_seconds', 5))
        frame_dir = "/tmp/frames_out"
        os.makedirs(frame_dir, exist_ok=True)
        run_ffmpeg("-i", input_path,
                   "-vf", f"fps=1/{interval}",
                   os.path.join(frame_dir, "frame_%04d.jpg"))
        zip_path = output_path.rsplit('.', 1)[0] + '.zip'
        with zipfile.ZipFile(zip_path, 'w') as zf:
            for fn in sorted(os.listdir(frame_dir)):
                zf.write(os.path.join(frame_dir, fn), fn)
        shutil.move(zip_path, output_path)

    elif task_type == 'vid_gif_creation':
        fps     = int(params.get('fps', 10))
        end_sec = min(8.0, duration)  # limit GIF length
        palette  = "/tmp/palette.png"
        run_ffmpeg("-ss", "0", "-to", str(end_sec), "-i", input_path,
                   "-vf", "fps=10,scale=480:-1:flags=lanczos,palettegen", palette)
        gif_path = output_path.rsplit('.', 1)[0] + '.gif'
        run_ffmpeg("-ss", "0", "-to", str(end_sec), "-i", input_path,
                   "-i", palette,
                   "-filter_complex", "fps=10,scale=480:-1:flags=lanczos[x];[x][1:v]paletteuse",
                   gif_path)
        shutil.move(gif_path, output_path)

    elif task_type == 'vid_split_segments':
        parts      = max(2, int(params.get('parts', 4)))
        seg_dur    = duration / parts
        seg_dir    = "/tmp/segments_out"
        os.makedirs(seg_dir, exist_ok=True)
        for i in range(parts):
            start_t = i * seg_dur
            out_seg = os.path.join(seg_dir, f"segment_{i+1:02d}.mp4")
            run_ffmpeg("-ss", str(start_t), "-t", str(seg_dur),
                       "-i", input_path, "-c", "copy", out_seg)
        zip_path = output_path.rsplit('.', 1)[0] + '.zip'
        with zipfile.ZipFile(zip_path, 'w') as zf:
            for fn in sorted(os.listdir(seg_dir)):
                zf.write(os.path.join(seg_dir, fn), fn)
        shutil.move(zip_path, output_path)

    else:
        run_ffmpeg("-i", input_path, "-c", "copy", output_path)

    print(f"Video task done: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  AUDIO TASKS  (5)
# ─────────────────────────────────────────────────────────────────────────────

def process_audio(task_type, input_path, params, output_path):
    print(f"Processing AUDIO task: {task_type}")

    try:
        from pydub import AudioSegment
    except ImportError:
        raise RuntimeError("pydub not installed")

    audio = AudioSegment.from_file(input_path)

    if task_type == 'aud_noise_red':
        sensitivity = float(params.get('sensitivity', 0.5))
        audio = audio.high_pass_filter(200)
        audio = audio + (sensitivity * 4)   # slight boost

    elif task_type == 'aud_format_conv':
        codec = params.get('codec', 'wav').lower().strip()
        ext = 'wav' if codec == 'wav' else 'mp3'
        tmp = f"/tmp/aud_conv.{ext}"
        audio.export(tmp, format=ext)
        shutil.move(tmp, output_path)
        return

    elif task_type == 'aud_trimming':
        start_ms = int(float(params.get('from', 0)) * 1000)
        end_ms   = int(float(params.get('to', 30)) * 1000)
        audio    = audio[start_ms: min(end_ms, len(audio))]

    elif task_type == 'aud_normalization':
        headroom = float(params.get('level', 1.0))
        audio    = audio.normalize(headroom=headroom)

    elif task_type == 'aud_split_track':
        parts    = max(2, int(params.get('parts', 2)))
        chunk_ms = len(audio) // parts
        seg_paths = []
        for i in range(parts):
            seg = audio[i * chunk_ms: (i + 1) * chunk_ms]
            sp  = f"/tmp/aud_seg_{i+1:02d}.mp3"
            seg.export(sp, format='mp3')
            seg_paths.append(sp)
        zip_path = output_path.rsplit('.', 1)[0] + '.zip'
        with zipfile.ZipFile(zip_path, 'w') as zf:
            for sp in seg_paths:
                zf.write(sp, os.path.basename(sp))
        shutil.move(zip_path, output_path)
        return

    audio.export(output_path, format='mp3')
    print(f"Audio saved: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  PDF TASKS  (5)
# ─────────────────────────────────────────────────────────────────────────────

def process_pdf(task_type, workspace_dir, params, output_path):
    print(f"Processing PDF task: {task_type}")
    import PyPDF2

    if task_type == 'pdf_merge':
        writer = PyPDF2.PdfWriter()
        files  = sorted([f for f in os.listdir(workspace_dir)
                         if f.lower().endswith('.pdf') and f != 'params.json'])
        if not files:
            raise Exception("No PDF files found for merge")
        print(f"Merging {len(files)} PDFs: {files}")
        for fname in files:
            reader = PyPDF2.PdfReader(os.path.join(workspace_dir, fname))
            for page in reader.pages:
                writer.add_page(page)
        with open(output_path, 'wb') as f:
            writer.write(f)
        return

    # Single-file PDF tasks
    pdf_path = next((os.path.join(workspace_dir, f) for f in os.listdir(workspace_dir)
                     if f.lower().endswith('.pdf') and os.path.isfile(os.path.join(workspace_dir, f))), None)
    if not pdf_path:
        raise Exception("No PDF file found in workspace")

    reader = PyPDF2.PdfReader(pdf_path)
    writer = PyPDF2.PdfWriter()

    if task_type == 'pdf_split':
        rng   = params.get('range', '1-2').split('-')
        start = max(0, int(rng[0]) - 1)
        end   = min(int(rng[-1]), len(reader.pages))
        for i in range(start, end):
            writer.add_page(reader.pages[i])
        with open(output_path, 'wb') as f:
            writer.write(f)

    elif task_type == 'pdf_password':
        password = params.get('pass', 'cloud')
        for page in reader.pages:
            writer.add_page(page)
        writer.encrypt(password)
        with open(output_path, 'wb') as f:
            writer.write(f)

    elif task_type in ('pdf_to_office', 'pdf_extraction'):
        text_all = "\n\n".join(page.extract_text() or "" for page in reader.pages)
        txt_path = output_path.rsplit('.', 1)[0] + '.txt'
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(text_all)
        shutil.move(txt_path, output_path)

    else:
        for page in reader.pages:
            writer.add_page(page)
        with open(output_path, 'wb') as f:
            writer.write(f)

    print(f"PDF saved: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 4:
        print("Usage: python3 unified_executor.py <workspace_dir> <params_json> <output>")
        sys.exit(1)

    workspace_dir = sys.argv[1]
    params_f      = sys.argv[2]
    output_f      = sys.argv[3]

    with open(params_f, 'r') as f:
        params = json.load(f)

    task_type = params.get('task_type', 'img_resize')
    print(f"Task: {task_type} | Output: {output_f}")

    def get_input():
        for fname in sorted(os.listdir(workspace_dir)):
            full = os.path.join(workspace_dir, fname)
            if fname != 'params.json' and not fname.startswith('.') and os.path.isfile(full):
                return full
        return None

    try:
        inp = get_input()
        print(f"Input: {inp}")

        if task_type.startswith('img_'):
            process_image(task_type, inp, params, output_f)
        elif task_type.startswith('vid_'):
            process_video(task_type, inp, params, output_f)
        elif task_type.startswith('aud_'):
            process_audio(task_type, inp, params, output_f)
        elif task_type.startswith('pdf_'):
            process_pdf(task_type, workspace_dir, params, output_f)
        else:
            raise Exception(f"Unknown task type: {task_type}")

        print(f"SUCCESS: {output_f}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
