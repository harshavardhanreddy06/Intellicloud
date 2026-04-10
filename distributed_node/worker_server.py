from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import time
import subprocess
import json
import shutil
import platform

app = Flask(__name__)
CORS(app)

# Local configurations for the worker laptop
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
RESULT_FOLDER = os.path.join(BASE_DIR, 'results')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Default Tier Configurations (Mirror Master-Node specs)
TIER_CONFIGS = {
    0: {"name": "Tiny",   "cpus": 0.25, "mem": "1024m", "static_power": 15.0},
    1: {"name": "Small",  "cpus": 0.50, "mem": "2048m", "static_power": 35.0},
    2: {"name": "Medium", "cpus": 1.00, "mem": "4096m", "static_power": 55.0},
}

def compute_dynamic_metrics(task_type, size_mb, tier_id, duration):
    base_cpu = 0.5 if 'image' in task_type else (0.8 if 'matrix' in task_type else 0.3)
    cpu_util = min(100, base_cpu * 100 * (1.2 if tier_id == 0 else 1.0))
    mem_base = size_mb * 1.5 if 'image' in task_type else (size_mb * 0.8)
    mem_req = mem_base + 30.0
    return {"CPU": round(cpu_util, 2), "MEM": round(mem_req, 2)}

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "online",
        "node": platform.node(),
        "arch": platform.machine(),
        "os": platform.system()
    })

@app.route('/execute_task', methods=['POST'])
def execute_task():
    try:
        # 1. Received Payload and Params
        if not request.files:
            return jsonify({"status": "error", "message": "No files uploaded"}), 400

        params_json = request.form.get('params', '{}')
        params = json.loads(params_json)
        task_type = params.get('task_type', 'img_resize')
        
        task_id = f"worker_task_{int(time.time())}"
        task_dir = os.path.join(UPLOAD_FOLDER, task_id)
        os.makedirs(task_dir, exist_ok=True)

        # Save all Payload files
        input_size_mb = 0
        for key in request.files:
            f = request.files[key]
            file_path = os.path.join(task_dir, f.filename)
            f.save(file_path)
            input_size_mb += os.path.getsize(file_path) / (1024 * 1024)

        # Save params.json for the container
        with open(os.path.join(task_dir, "params.json"), 'w') as f:
            json.dump(params, f)

        # 2. Extract Deployment Tier
        tier_id = int(params.get('tier_id', 2))
        tier = TIER_CONFIGS.get(tier_id, TIER_CONFIGS[2])
        # Global Extension Map for Workers (aligned with Master)
        EXT_MAP = {
            'img_resize': 'jpg', 'img_cropping': 'jpg', 'img_compression': 'jpg', 'img_format_conv': 'jpg',
            'img_watermark': 'jpg', 'img_puzzle_split': 'jpg', 'img_color_corr': 'jpg', 'img_bg_removal': 'png',
            'img_annotation': 'jpg', 'img_batch_rename': 'jpg',
            'vid_cropping': 'mp4', 'vid_trimming': 'mp4', 'vid_compression': 'mp4', 'vid_remove_audio': 'mp4',
            'vid_add_subtitles': 'mp4', 'vid_format_conv': 'mp4', 'vid_frame_extraction': 'zip',
            'vid_gif_creation': 'gif', 'vid_watermarking': 'mp4', 'vid_split_segments': 'zip',
            'aud_noise_red': 'mp3', 'aud_format_conv': 'mp3', 'aud_trimming': 'mp3', 'aud_normalization': 'mp3',
            'aud_split_track': 'zip',
            'pdf_merge': 'pdf', 'pdf_split': 'pdf', 'pdf_to_office': 'txt', 'pdf_password': 'pdf', 'pdf_extraction': 'txt',
        }
        ext = EXT_MAP.get(task_type, 'pdf')
        output_filename = f"res_{task_id}.{ext}"
        output_path = os.path.join(RESULT_FOLDER, output_filename)

        # 3. Docker Execution Logic (Distributed)
        abs_workspace_dir = os.path.abspath(task_dir)
        abs_result_dir = os.path.abspath(RESULT_FOLDER)

        container_cmd = [
            "docker", "run", "--rm",
            f"--name", f"{tier['name'].lower()}_remote_{task_id}",
            f"--cpus={tier['cpus']}",
            f"--memory={tier['mem']}",
            "-v", f"{abs_workspace_dir}:/app/workspace",
            "-v", f"{abs_result_dir}:/app/out_d",
            "intellicloud-task:unified",
            "unified_executor.py", "/app/workspace", "/app/workspace/params.json", f"/app/out_d/{output_filename}"
        ]

        print(f"🚀 [WORKER] Executing: {' '.join(container_cmd)}")
        start_t = time.time()
        proc = subprocess.run(container_cmd, capture_output=True, text=True)
        duration = time.time() - start_t

        if proc.returncode != 0:
            return jsonify({"status": "error", "message": f"Worker execution failed: {proc.stderr}"}), 500

        # Compute Telemetry
        metrics = compute_dynamic_metrics(task_type, input_size_mb, tier_id, duration)
        
        # 4. Success Response
        return jsonify({
            "status": "success",
            "execution": {
                "vm": tier['name'],
                "duration": round(duration, 3),
                "metrics": metrics,
                "result_file_url": f"/results/{output_filename}"
            }
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/results/<path:filename>')
def serve_result(filename):
    return send_from_directory(RESULT_FOLDER, filename)

import argparse
import requests
import socket

def register_with_master(master_url, my_ip):
    try:
        # Pokes the master to let it know we exist!
        requests.post(f"{master_url}/api/register_node", json={"ip": my_ip}, timeout=5)
        print(f"✅ AUTO-REGISTRATION: Successfully connected to master at {master_url}")
    except Exception as e:
        print(f"⚠️ AUTO-REGISTRATION: Couldn't connect to master at {master_url}. Please ensure Master API is running.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IntelliCloud Distributed Worker')
    parser.add_argument('--master', help='Master node URL (e.g. http://192.168.1.10:5001)')
    args = parser.parse_args()

    # Automatic local IP detection for registration
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        my_ip = s.getsockname()[0]
    finally:
        s.close()

    if args.master:
        # Strip trailing slash if present
        master_url = args.master.rstrip('/')
        register_with_master(master_url, my_ip)
    else:
        print(f"⚠️ No master URL provided. Run with --master http://MASTER_IP:5001 to auto-register.")

    print(f"💻 WORKER NODE: Listening on http://{my_ip}:5002")
    app.run(host='0.0.0.0', port=5002, debug=False)
