# 🧱 IntelliCloud Worker Node Setup

Run these steps on each of the **2 additional laptops** to prepare them for distributed tasks.

## 1. Prerequisites
Ensure the following are installed:
- **Python 3.8+**
- **Docker Home/Desktop** (Must be running)

## 2. Prepare Docker Image
The worker laptop needs the same Docker image as the Master. Run this in the root repo folder:
```bash
docker build -t intellicloud-task:unified -f orchestrator/Dockerfile_task_unified .
```

## 3. Run the Worker Server
Navigate to the `distributed_node` folder and install dependencies:
```bash
pip install -r requirements.txt
```

Start the worker server (Listen on all interfaces):
```bash
python worker_server.py
```
*The worker will start on port **5002**.*

## 4. Master Node Connection
Note down the **LAN IP Address** of this laptop (e.g., `192.168.1.15`). 
Add this IP to the `WORKER_IPS` list in the Master's `api/app.py`.

---

### Communication Flow:
1. Master Node (DQN) chooses this laptop.
2. Master `POST`es the payload (image/video/etc.) to `http://<worker_ip>:5002/execute_task`.
3. This laptop runs Docker locally.
4. This laptop sends back the result JSON + URL to the result file.
5. Master downloads the file and shows it on the dashboard.
