# Task Sharing and Downloading Results

Task sharing is the process where the master laptop distributes an incoming task to a remote worker node for execution and eventually downloads the result.

## 1. How 2 Laptops are Sharing Tasks
When a remote node is selected by the **Master Orchestrator**:

1. **Payload Packaging**: The master collects the task's input file (e.g., `payload.jpg`) and creates a `params.json` file.
2. **REST API Dispatch**:
   - The master makes a `POST /execute_task` request to the worker's IP (`http://<WORKER_IP>:5002`).
   - The request contains the **file payload** (as `multipart/form-data`) and the **task parameters** (task type, priority, and DQN-selected container tier).
3. **Execution Acknowledgement**: The worker confirms receipt and starts a Docker container for the task.

## 2. Remote Execution Flow
The worker processes the task using the following steps:
1. **Workspace Creation**: A temporary folder is created in the worker's `uploads/` directory.
2. **Docker Orchestration**: The worker runs the `intellicloud-task:unified` Docker image with the resource limits (CPU/Memory) specified by the master.
3. **Result Generation**: The final processed file is saved in the worker's `results/` folder.

## 3. Downloading Results
After the task is finished on the remote worker:

1. **Success Callback**: The worker sends a JSON response back to the master containing:
   - **Metrics**: CPU usage, memory usage, and total duration.
   - **Result URL**: The endpoint where the processed result is stored (`/results/res_worker_task_...`).
2. **Master Syncing**:
   - The master initiates a `GET` request to the worker's result URL.
   - The master downloads the artifact (image, video, etc.) in small chunks (8KB at a time) for stability.
   - The result is synchronized and saved in the master's own `results/` folder.
3. **Frontend Notification**: The master emits a `COMPLETED` event via SocketIO, and the dashboard is updated with the final metrics and the download link for the result.

## 4. Key Components used in Task Sharing
- **requests (Python library)**: Used for all inter-node communication (Master → Worker and Worker ← Master).
- **Multipart Form Upload**: Used to transfer binary files (up to 100MB+) between nodes.
- **JSON Metadata**: Used for synchronizing task settings across different systems.
- **8KB Chunks**: Used for reliable large-file downloads over local WiFi networks.

## 5. Summary of Task State during Sharing
| Phase | Location of File | Action |
|-------|------------------|--------|
| **Initial** | Master Desktop | User uploads through the dashboard |
| **Dispatch** | Network | Transferred via WiFi POST request |
| **Execution** | Worker File System | Processed inside a Docker container |
| **Collection** | Network | Transferred via WiFi GET request |
| **Final** | Master Dashboard | Available for download by the user |
