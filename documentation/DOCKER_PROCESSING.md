# Docker Task Processing

Docker is the fundamental unit of execution in the IntelliCloud project. It provides isolated environments for processing tasks with specific resource limits (CPU and Memory).

## 1. Unified Task Image
The `intellicloud-task:unified` image is built using a custom Dockerfile located in `orchestrator/Dockerfile_task_unified`. It contains all the necessary tools for processing multiple task types:
- **Base Image**: Python 3.9 (slim-buster).
- **Core Tools**: `ffmpeg` (video/audio processing), `libmagick` (image processing), `ghostscript` (PDF handling).
- **Libraries**: `opencv`, `pillow`, `pypdf2`, `moviepy`, and others.

## 2. Docker Execution Strategy
When a task is submitted (either locally or remotely), the orchestrator triggers a `docker run` command:

### The Commands Breakdown
```bash
docker run --rm \
  --cpus=<DQN_CPU_LIMIT> \
  --memory=<DQN_MEMORY_LIMIT> \
  -v <ABS_WORKSPACE_DIR>:/app/workspace \
  -v <ABS_RESULT_DIR>:/app/out_d \
  intellicloud-task:unified \
  unified_executor.py /app/workspace /app/workspace/params.json /app/out_d/<OUTPUT_FILENAME>
```

- **`--rm`**: Automatically removes the container after it's finished to save disk space.
- **`--cpus`**: Implements the **CPU Resource Limit** (e.g., 0.25 cores if the DQN Agent selects a "Tiny" container).
- **`--memory`**: Implements the **Memory Resource Limit** (e.g., 1024m).
- **`-v` (Volumes)**:
  - **Workspace Volume**: Mounts the task input folder containing the input file (`payload`).
  - **Result Volume**: Mounts the output folder where the final processed file will be saved.

## 3. How Docker Process is Working
1. **Container Initialization**: The container starts and launches `unified_executor.py`.
2. **Input Scanning**: The executor reads the mounted `params.json` to identify the task type and settings.
3. **Task Logic**:
   - **Video/Audio**: Invokes `ffmpeg` commands via `subprocess` for trimming, resizing, or format conversion.
   - **Images**: Uses `Pillow` or `OpenCV` to perform filters, watermarking, or compression.
   - **PDFs**: Uses `PyPDF2` to merge, split, or extract text from documents.
4. **Output Generation**: The final processed file is saved directly into the mounted output volume on the host system.
5. **Autorelease**: The container shuts down instantly, releasing all allocated CPU and Memory resources back to the host laptop.

## 4. Key Advantages of Docker in IntelliCloud
- **Isolation**: Each task runs in its own "sandbox," preventing one task from crashing the entire server.
- **Resource Enforcement**: Guarantees that the **DQN Agent's** scheduling choices (e.g., restricted cores) are strictly followed.
- **Portability**: The same task image works on both the master laptop and the remote worker laptop (regardless of OS version).

## 5. Summary of Docker Actions
| Component | Interaction with Docker |
|-----------|-------------------------|
| **DQN Agent** | Selects the `--cpus` and `--memory` flags. |
| **Orchestrator** | Spawns the container and mounts the task volumes. |
| **Worker Server** | Receives the call and initiates the Docker run locally. |
| **Unified Executor** | Runs the actual task logic *inside* the container. |
