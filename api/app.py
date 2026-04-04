from gevent import monkey
monkey.patch_all()
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import os
import sys
import json
import time
import subprocess
import requests
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg')

# Project Root Resolution
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
from core.start import IntelliCloudPredictor

app = Flask(__name__)
app.config['SECRET_KEY'] = 'intellicloud_secret'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='gevent')

# Configuration (Anchored to Root)
UPLOAD_FOLDER = str(BASE_DIR / 'uploads')
# SHAP images are saved inside api/shap_explanations/ by core/start.py
SHAP_FOLDER = str(Path(__file__).resolve().parent / 'shap_explanations')
RESULT_FOLDER = str(BASE_DIR / 'results')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SHAP_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

predictor = IntelliCloudPredictor()

# --- DISTRIBUTED NODES CONFIGURATION ---
# Add your other laptop IPs here. Leave "LOCAL" for this laptop.
# Example: ["LOCAL", "192.168.1.15", "192.168.1.20"]
WORKER_NODES = ["LOCAL","10.177.21.128"] 

current_worker_index = 0

@app.route('/api/register_node', methods=['POST'])
def register_node():
    data = request.get_json()
    new_ip = data.get('ip')
    if new_ip and new_ip not in WORKER_NODES:
        WORKER_NODES.append(new_ip)
        socketio.emit('pipeline_update', {
            'stage': 'NETWORK',
            'message': f'New dynamic worker registered: {new_ip}'
        })
    return jsonify({"status": "success", "nodes": WORKER_NODES})

@app.route('/api/cluster_status', methods=['GET'])
def cluster_status():
    return jsonify({"nodes": WORKER_NODES, "active_index": current_worker_index})

# DQN Tiers
TIER_CONFIGS = {
    0: {"name": "Tiny",   "cpus": 0.25, "mem": "1024m", "static_power": 15.0},
    1: {"name": "Small",  "cpus": 0.50, "mem": "2048m", "static_power": 35.0},
    2: {"name": "Medium", "cpus": 1.00, "mem": "4096m", "static_power": 55.0},
}

def normalize_filename(filename):
    return "".join([c if c.isalnum() or c in "._-" else "_" for c in filename])

def get_script_for_task(task_type):
    return "unified_executor.py"

def compute_dynamic_metrics(task_type, size_mb, tier_id, duration):
    base_cpu = 0.5 if 'image' in task_type else (0.8 if 'matrix' in task_type else 0.3)
    cpu_util = min(100, base_cpu * 100 * (1.2 if tier_id == 0 else 1.0))
    mem_base = size_mb * 1.5 if 'image' in task_type else (size_mb * 0.8)
    mem_req = mem_base + 30.0
    return {"CPU": round(cpu_util, 2), "MEM": round(mem_req, 2)}

@app.route('/api/submit_task', methods=['POST'])
def submit_task():
    global current_worker_index
    try:
        data = request.form if request.form else (request.json if request.json else {})
        task_type = data.get('task_type', 'img_resize')
        priority = data.get('priority', 'medium')
        raw_params = data.get('params', '{}')
        
        timestamp = int(time.time())
        task_id = f"task_{timestamp}"
        task_dir = os.path.join(UPLOAD_FOLDER, task_id)
        os.makedirs(task_dir, exist_ok=True)
        
        input_filename = "payload"
        input_size_mb = 0
        first_file_path = ""

        # Progress: Task Received
        socketio.emit('pipeline_update', {'stage': 'RECEIVED', 'message': 'Task received by Master Master'})

        # Save all uploaded files 
        if request.files:
            for key in request.files:
                f = request.files[key]
                safe_name = normalize_filename(f.filename)
                first_file_path = os.path.join(task_dir, safe_name)
                f.save(first_file_path)
                real_size = os.path.getsize(first_file_path) / (1024 * 1024)
                input_size_mb += real_size
            first_key = list(request.files.keys())[0]
            input_filename = normalize_filename(request.files[first_key].filename)
        else:
            raw_data = data.get('raw_data', 'Sample data for demo.')
            input_size_mb = float(data.get('input_size_mb', len(raw_data) / (1024 * 1024)))
            input_filename = "data.txt"
            first_file_path = os.path.join(task_dir, input_filename)
            with open(first_file_path, 'w') as f:
                f.write(raw_data)

        # --- 6-STEP INTELLIGENT PIPELINE START ---
        # 1. Feature Extraction
        socketio.emit('pipeline_update', {'stage': 'FEATURE_EXTRACT', 'message': 'Scanning task payload for high-fidelity telemetry...'})
        time.sleep(1.0)
        
        # 2. Autoencoder State Reduction
        socketio.emit('pipeline_update', {'stage': 'AUTOENCODER', 'message': 'Compressing state into 13-dimension latent space...'})
        time.sleep(0.8)
        
        # 3. RF Efficiency Prediction
        socketio.emit('pipeline_update', {'stage': 'RF_PREDICT', 'message': 'Random Forest predicting energy efficiency class...'})
        task_dict = {
            "task_type": task_type,
            "input_size_mb": max(0.1, input_size_mb),
            "complexity": 'high' if input_size_mb > 100 else 'medium',
            "priority": priority,
            "application": "intellicloud_gui"
        }
        prediction_res = predictor.predict_energy_efficiency(task_dict, include_shap=True)
        time.sleep(1.0)

        # 4. SHAP Explanation Generation
        socketio.emit('pipeline_update', {'stage': 'SHAP_GEN', 'message': 'Generating SHAP correlation explanation report...'})
        time.sleep(0.8)
        
        # 5. DQN Decision
        socketio.emit('pipeline_update', {'stage': 'DQN_DECISION', 'message': 'DQN agent selecting optimal container deployment tier...'})
        action_id = prediction_res['vm_scheduling']['dqn_action_id']
        tier = TIER_CONFIGS[action_id]
        time.sleep(0.8)

        # 6. Node Selection & Execution (Existing logic)
        selected_node = WORKER_NODES[current_worker_index % len(WORKER_NODES)]
        current_worker_index += 1
        
        socketio.emit('pipeline_update', {
            'stage': 'EXECUTING', 
            'message': f'Dispatching to {selected_node} VM (Tier: {tier["name"]})',
            'node': selected_node
        })

        execution_data = {}
        duration = 0
        metrics = {}

        if selected_node == "LOCAL":
            # --- LOCAL EXECUTION ---
            socketio.emit('pipeline_update', {'stage': 'EXECUTING', 'message': 'Running on Master Laptop...'})
            script_name = get_script_for_task(task_type)
            base_name = input_filename.rsplit('.', 1)[0] if '.' in input_filename else input_filename
            # Complete extension map for all 30 tasks — no .bin fallback
            EXT_MAP = {
                # Image (10)
                'img_resize':        'jpg',
                'img_cropping':      'jpg',
                'img_compression':   'jpg',
                'img_format_conv':   'jpg',   # executor may change ext internally
                'img_watermark':     'jpg',
                'img_puzzle_split':  'jpg',
                'img_color_corr':    'jpg',
                'img_bg_removal':    'png',
                'img_annotation':    'jpg',
                'img_batch_rename':  'jpg',
                # Video (10)
                'vid_cropping':         'mp4',
                'vid_trimming':         'mp4',
                'vid_compression':      'mp4',
                'vid_remove_audio':     'mp4',
                'vid_add_subtitles':    'mp4',
                'vid_format_conv':      'mp4',
                'vid_frame_extraction': 'zip',
                'vid_gif_creation':     'gif',
                'vid_watermarking':     'mp4',
                'vid_split_segments':   'zip',
                # Audio (5)
                'aud_noise_red':     'mp3',
                'aud_format_conv':   'mp3',
                'aud_trimming':      'mp3',
                'aud_normalization': 'mp3',
                'aud_split_track':   'zip',
                # PDF (5)
                'pdf_merge':      'pdf',
                'pdf_split':      'pdf',
                'pdf_to_office':  'txt',
                'pdf_password':   'pdf',
                'pdf_extraction': 'txt',
            }
            ext = EXT_MAP.get(task_type, 'pdf')  # safe fallback
            output_filename = f"res_{int(time.time())}_{base_name}.{ext}"

            params_dict = json.loads(raw_params)
            params_dict['task_type'] = task_type
            params_path = os.path.join(task_dir, "params.json")
            with open(params_path, 'w') as f: json.dump(params_dict, f)

            container_cmd = [
                "docker", "run", "--rm",
                f"--cpus={tier['cpus']}", f"--memory={tier['mem']}",
                "-v", f"{os.path.abspath(task_dir)}:/app/workspace",
                "-v", f"{os.path.abspath(RESULT_FOLDER)}:/app/out_d",
                "intellicloud-task:unified",
                script_name, "/app/workspace", "/app/workspace/params.json", f"/app/out_d/{output_filename}"
            ]
            
            start_t = time.time()
            proc = subprocess.run(container_cmd, capture_output=True, text=True)
            duration = time.time() - start_t

            if proc.returncode != 0:
                return jsonify({"status": "error", "message": f"Local Execution Failed: {proc.stderr}"}), 500
            
            metrics = compute_dynamic_metrics(task_type, input_size_mb, action_id, duration)
            execution_data = {
                "vm": tier['name'],
                "duration": round(duration, 3),
                "metrics": metrics,
                "result_file": f"/results/{output_filename}"
            }
        else:
            # --- REMOTE EXECUTION ---
            socketio.emit('pipeline_update', {
                'stage': 'DISPATCHING', 
                'message': f'Uploading payload to {selected_node}...'
            })
            
            worker_url = f"http://{selected_node}:5002/execute_task"
            params_dict = json.loads(raw_params)
            params_dict.update({'task_type': task_type, 'tier_id': action_id})
            
            with open(first_file_path, 'rb') as f:
                files = {'file': (input_filename, f)}
                payload = {'params': json.dumps(params_dict)}
                
                socketio.emit('pipeline_update', {'stage': 'EXECUTING', 'message': f'Worker {selected_node} executing Docker...'})
                response = requests.post(worker_url, files=files, data=payload, timeout=300)
            
            if response.status_code != 200:
                return jsonify({"status": "error", "message": f"Remote Worker Error: {response.text}"}), 500
            
            remote_res = response.json()
            socketio.emit('pipeline_update', {'stage': 'COLLECTING', 'message': 'Downloading results from worker...'})
            
            # Download result from worker back to master's result folder
            remote_file_url = f"http://{selected_node}:5002{remote_res['execution']['result_file_url']}"
            output_filename = remote_res['execution']['result_file_url'].split('/')[-1]
            local_result_path = os.path.join(RESULT_FOLDER, output_filename)
            
            socketio.emit('pipeline_update', {'stage': 'COLLECTING', 'message': f'Syncing result artifact: {output_filename}...'})
            
            # Robust file download from remote worker
            with requests.get(remote_file_url, stream=True) as r:
                r.raise_for_status()
                with open(local_result_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            
            duration = remote_res['execution']['duration']
            metrics = remote_res['execution']['metrics']
            execution_data = {
                "vm": tier['name'],
                "duration": round(duration, 3),
                "metrics": metrics,
                "result_file": f"/results/{output_filename}",
                "remote_node": selected_node,
                "synced": True
            }

        # Common Post-Processing (Telemetry & Online Learning)
        # ... (Keeping the energy and recording logic from before)
        actual_power = tier['static_power'] + (40.0 * (metrics['CPU'] / 100.0) * tier['cpus'])
        actual_joules = actual_power * duration
        baseline_joules = (TIER_CONFIGS[2]['static_power'] + 40.0) * (duration * 0.45)
        saved_percent = max(0, min(95, ((baseline_joules - actual_joules) / baseline_joules) * 100 if baseline_joules > 0 else 0))

        socketio.emit('pipeline_update', {'stage': 'COMPLETED', 'message': 'Task Pipeline Finished Successfully!'})

        execution_data.update({
            "energy": {
                "joules": round(actual_joules, 2),
                "saved_joules": round(max(0, baseline_joules - actual_joules), 2),
                "percent": round(saved_percent, 1)
            }
        })

        return jsonify({
            "status": "success",
            "prediction": prediction_res,
            "execution": execution_data
        })
        
    except Exception as e:
        socketio.emit('pipeline_update', {'stage': 'ERROR', 'message': str(e)})
        return jsonify({"status": "error", "message": f"Pipeline Error: {str(e)}"}), 500

@app.route('/shap_explanations/<path:filename>')
def serve_shap(filename):
    return send_from_directory(SHAP_FOLDER, filename)

@app.route('/results/<path:filename>')
def serve_results_folder_static(filename):
    return send_from_directory(RESULT_FOLDER, filename)

@app.route('/api/latest_shap')
def latest_shap():
    """Returns URL of the most recently generated SHAP explanation image."""
    try:
        files = [f for f in os.listdir(SHAP_FOLDER) if f.endswith('.png')]
        if not files:
            return jsonify({'url': None}), 404
        latest = max(files, key=lambda f: os.path.getmtime(os.path.join(SHAP_FOLDER, f)))
        return jsonify({'url': f'/shap_explanations/{latest}', 'filename': latest})
    except Exception as e:
        return jsonify({'url': None, 'error': str(e)}), 500

@app.route('/results/<path:filename>')
def serve_result(filename):
    return send_from_directory(RESULT_FOLDER, filename)

if __name__ == '__main__':
    socketio.run(app, port=5001, debug=False, allow_unsafe_werkzeug=True)
