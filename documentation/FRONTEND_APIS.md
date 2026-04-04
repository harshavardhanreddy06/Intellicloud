# Frontend APIs

The IntelliCloud backend is a Flask-based REST API that serves as the central hub for task submission, visualization, and distributed orchestration.

## 1. Task Submission API
### `POST /api/submit_task`
Submits a new cloud task for intelligent scheduling and execution.

- **Request Type**: `multipart/form-data` or `application/json`.
- **Payload Parameters**:
  - `file`: The task input file (image, video, etc.).
  - `task_type`: The type of task (e.g., `img_resize`, `vid_trimming`).
  - `priority`: The task priority (`low`, `medium`, `high`, `critical`).
  - `params`: JSON string with additional parameters (e.g., `{"width": 800, "height": 600}`).
- **Key Pipeline Stages Triggered**:
  1. **FEATURE_EXTRACT**: Historical telemetry lookup.
  2. **AUTOENCODER**: Dimensionality reduction.
  3. **RF_PREDICT**: Energy efficiency classification.
  4. **SHAP_GEN**: AI explanation report generation.
  5. **DQN_DECISION**: Container tier selection.
  6. **EXECUTING**: Deployment to local or remote Docker VM.

## 2. Distributed Node Management
### `POST /api/register_node`
Registers a new worker node's IP address with the cluster.

### `GET /api/cluster_status`
Returns a list of all active remote workers and their current load index.

## 3. Telemetry and Results
### `GET /api/latest_shap`
Returns the URL and filename of the most recently generated SHAP explanation image for UI rendering.

### `GET /shap_explanations/<filename>`
Serves the SHAP explanation image files from the backend storage.

### `GET /results/<filename>`
Serves the completed task result files (e.g., processed images or videos).

## 4. Real-time Communication (SocketIO)
The backend uses **Flask-SocketIO** to push live updates to the frontend dashboard without the need for manual refreshing.

### Emitted Events:
- **`pipeline_update`**: Notifies the dashboard of a stage change in the task pipeline (e.g., "AUTOENCODER" -> "RF_PREDICT").
- **`cluster_update`**: Notifies of new nodes joining or leaving the cluster.

## 5. Development Details
- **Port**: 5001 (Master Application), 5002 (Remote Worker Server).
- **CORS**: Enabled for all origins (`*`) to allow local and remote frontend access.
- **Async Mode**: Gevent is used for high-performance non-blocking I/O.
