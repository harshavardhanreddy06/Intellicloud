# Feature Extraction

Feature extraction is the first stage in the IntelliCloud intelligent task pipeline. It converts a raw user request into a 12-feature telemetry vector for the AI models.

## 1. Feature Extraction Process
The `TaskFeatureExtractor` class in `core/feature_extractor.py` handles the extraction phase.

### Stage 1: Categorization
The input task type is mapped to one of five domains:
- `compute`: Low I/O, High CPU (e.g., matrix multiplication).
- `analysis`: High Compute, High Memory (e.g., log parsing).
- `io_heavy`: High Network and I/O (e.g., data deduplication).
- `media`: High CPU, High Memory (e.g., image/video processing).
- `general`: Standard tasks.

### Stage 2: Historical Lookup
The system consults `task_profiles_clean_final.json` to find a matching task.

#### A. Exact Match (Tolerance 5%)
If a task of the same `task_type` and compatible `input_size_mb` (within 5%) exists in history:
- The system uses the **median** value of all matching historical records.
- This ensures high confidence (exact_match_found = True).

#### B. Scaling from History (Log-Log Regression)
If no exact match is found, the system performs **Logarithmic Scaling**:
- It finds the closest historical task of the same type.
- It calculates a **scaling exponent (ε)** using historical data for that task type.
- Estimated Metric = `Historical Metric * (Incoming Size / Historical Size) ^ ε`.
- This handles arbitrary task sizes (e.g., a 250MB image resize when only 10MB data is available).

### Stage 3: Feature Adjustments
The extracted metrics are refined based on:
- **Complexity Factor**: Low: 0.9x, Medium: 1.0x, High: 1.15x.
- **Priority Factor**: Low: 1.05x, Critical: 0.9x (affects execution time estimates).

## 2. Telemetry Output
The final extraction results in an 8-feature vector:
1. `input_size_mb`
2. `cpu_usage_cores_absolute`
3. `memory_usage_mb`
4. `execution_time_normalized`
5. `instruction_count`
6. `network_io_mb`
7. `power_consumption_watts`
8. `task_size_category` (Ordinal: 0, 1, 2)

## 3. Power Model Calculation
The power consumption is calculated using a dynamic model:
`Power (W) = Base Power (40W) + (40W * CPU Core Allocation)`
- This correlates estimated CPU utilization with real electrical draw.

## 4. Key Performance Indicators (KPIs)
- **exact_match_found**: Boolean flag indicating if a similar task existed in history.
- **is_scaled**: Boolean flag indicating if regression was used for estimation.
- **is_having_history**: Boolean flag indicating if a record of this task type exists at all.
