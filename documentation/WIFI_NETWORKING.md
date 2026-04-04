# Connecting Remote Laptops via WiFi

The IntelliCloud project supports a distributed computing architecture where multiple laptops connected to the same WiFi network can act as worker nodes to process tasks.

## 1. Network Requirements
- **Local Network**: All laptops must be on the same WiFi router or local network.
- **Firewall Configuration**: Each laptop must allow inbound connections onports 5001 (Master) and 5002 (Worker).

## 2. Setting Up the Worker Laptop
The worker laptop acts as a slave and receives tasks from the master laptop.

1. **Find Worker IP**: Use `ipconfig` (Windows) or `ifconfig` (macOS/Linux) to find the worker's local IP (e.g., `192.168.1.15`).
2. **Setup Dependencies**: Ensure Docker and Python (with Flask) are installed.
3. **Start Worker Server**:
   ```bash
   python distributed_node/worker_server.py --master http://<MASTER_IP>:5001
   ```
   *Replace `<MASTER_IP>` with the local IP of the master laptop.*

## 3. How 2 Laptops Connect
The connection is established through an **Auto-Registration** process:

1. **IP Discovery**: The `worker_server.py` uses the standard Python `socket` module to detect its own local IP address.
2. **Registration Poke**: The worker sends a `POST /api/register_node` request to the master's IP.
3. **Dynamic Cluster Update**: The master laptop adds the worker's IP to its `WORKER_NODES` list and emits a "NETWORK" update to the dashboard.
4. **Heartbeat/Health Check**: The master can verify the worker's status using the `/health` endpoint at any time.

## 4. Key Components used in Connectivity
- **Master URL**: The central hub that coordinates all tasks (e.g., `http://192.168.1.10:5001`).
- **Worker Node**: A laptop listening on its local IP (e.g., `http://192.168.1.15:5002`).
- **Shared WiFi**: Provides the underlying transmission medium for all JSON and file data.

## 5. Summary of the Process
1. Worker starts and pokes Master.
2. Master registers Worker IP.
3. Dashboard shows a new active node.
4. Tasks are now load-balanced between the Master and Worker.
