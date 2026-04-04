import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import io from 'socket.io-client';
import { 
  Zap, Cpu, Server, Activity, Database,
  BarChart, FileText, Settings, Shield, Maximize, Cloud, Upload,
  Download, FileDigit, CheckCircle, Loader2,
  Search, Layers, Laptop, Wifi, ArrowRight
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const API_BASE_URL = `http://${window.location.hostname}:5001`;
const socket = io(API_BASE_URL, { transports: ['websocket'], reconnection: true });

const App = () => {
    const [taskType, setTaskType] = useState('img_resize');
    const [priority, setPriority] = useState('medium');
    const [isUpload, setIsUpload] = useState(true);
    const [file, setFile] = useState(null);
    const [manualText, setManualText] = useState('');
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);
    const [extraParams, setExtraParams] = useState({});
    const [showShapModal, setShowShapModal] = useState(false);
    const [shapUrl, setShapUrl] = useState(null);

    // Pipeline State
    const [pipelineEvents, setPipelineEvents] = useState([]);
    const [selectedNode, setSelectedNode] = useState(null);
    const [currentStatus, setCurrentStatus] = useState('Ready for Task Dispatch');
    const [activeStage, setActiveStage] = useState('IDLE');

    const taskOptions = [
        { id: 'img_resize',       name: '🖼️ Image Resizing',       category: 'image', accept: 'image/*' },
        { id: 'img_cropping',     name: '🖼️ Image Cropping',       category: 'image', accept: 'image/*' },
        { id: 'img_compression',  name: '🖼️ Image Compression',    category: 'image', accept: 'image/*' },
        { id: 'img_format_conv',  name: '🖼️ Image Format Conv',    category: 'image', accept: 'image/*' },
        { id: 'img_watermark',    name: '🖼️ Image Watermarking',   category: 'image', accept: 'image/*' },
        { id: 'img_puzzle_split', name: '🖼️ Image Puzzle Split',   category: 'image', accept: 'image/*' },
        { id: 'img_color_corr',   name: '🖼️ Image Color Corr',    category: 'image', accept: 'image/*' },
        { id: 'img_bg_removal',   name: '🖼️ Image BG Removal',    category: 'image', accept: 'image/*' },
        { id: 'img_annotation',   name: '🖼️ Image Annotation',    category: 'image', accept: 'image/*' },
        { id: 'img_batch_rename', name: '🖼️ Image Batch Rename',  category: 'image', accept: 'image/*' },
        { id: 'vid_cropping',     name: '🎥 Video Cropping',       category: 'video', accept: 'video/*' },
        { id: 'vid_trimming',     name: '🎥 Video Trimming',       category: 'video', accept: 'video/*' },
        { id: 'vid_compression',  name: '🎥 Video Compression',   category: 'video', accept: 'video/*' },
        { id: 'vid_remove_audio', name: '🎥 Video Mute/Remove',   category: 'video', accept: 'video/*' },
        { id: 'vid_add_subtitles',name: '🎥 Video Subtitles',     category: 'video', accept: 'video/*' },
        { id: 'vid_format_conv',  name: '🎥 Video Format Conv',   category: 'video', accept: 'video/*' },
        { id: 'vid_frame_extraction', name: '🎥 Video Frame Extract', category: 'video', accept: 'video/*' },
        { id: 'vid_gif_creation', name: '🎥 Video GIF Creation',  category: 'video', accept: 'video/*' },
        { id: 'vid_watermarking', name: '🎥 Video Watermarking',  category: 'video', accept: 'video/*' },
        { id: 'vid_split_segments', name: '🎥 Video Split Segments', category: 'video', accept: 'video/*' },
        { id: 'aud_noise_red',    name: '🎧 Audio Noise Red',     category: 'audio', accept: 'audio/*' },
        { id: 'aud_format_conv',  name: '🎧 Audio Format Conv',   category: 'audio', accept: 'audio/*' },
        { id: 'aud_trimming',     name: '🎧 Audio Trimming',      category: 'audio', accept: 'audio/*' },
        { id: 'aud_normalization',name: '🎧 Audio Normalization', category: 'audio', accept: 'audio/*' },
        { id: 'aud_split_track',  name: '🎧 Audio Split Tracks',  category: 'audio', accept: 'audio/*' },
        { id: 'pdf_merge',        name: '📑 PDF Merge',           category: 'pdf',   accept: '.pdf' },
        { id: 'pdf_split',        name: '📑 PDF Split Pages',     category: 'pdf',   accept: '.pdf' },
        { id: 'pdf_to_office',    name: '📑 PDF to Office',       category: 'pdf',   accept: '.pdf' },
        { id: 'pdf_password',     name: '📑 PDF Password',        category: 'pdf',   accept: '.pdf' },
        { id: 'pdf_extraction',   name: '📑 PDF Extract Data',    category: 'pdf',   accept: '.pdf' },
    ];

    const paramsConfig = {
        // IMAGE (10) ─────────────────────────────────────────────────────────
        'img_resize':    [
            { key: 'width',  label: 'Width (px)',  type: 'number', default: 800 },
            { key: 'height', label: 'Height (px)', type: 'number', default: 600 },
        ],
        'img_cropping':  [
            { key: 'left',   label: 'Left (px)',   type: 'number', default: 0 },
            { key: 'top',    label: 'Top (px)',    type: 'number', default: 0 },
            { key: 'width',  label: 'Width (px)',  type: 'number', default: 400 },
            { key: 'height', label: 'Height (px)', type: 'number', default: 400 },
        ],
        'img_compression':  [
            { key: 'quality', label: 'Quality (1–100)', type: 'number', default: 80 },
        ],
        'img_format_conv':  [
            { key: 'target_format', label: 'Target Format (jpg/png/webp/bmp)', type: 'text', default: 'webp' },
        ],
        'img_watermark': [
            { key: 'text',    label: 'Watermark Text', type: 'text',   default: 'IntelliCloud' },
            { key: 'opacity', label: 'Opacity (0–1)',  type: 'number', default: 0.5 },
        ],
        'img_puzzle_split': [
            { key: 'tiles', label: 'Grid (e.g. 2x2, 3x3)', type: 'text', default: '2x2' },
        ],
        'img_color_corr': [
            { key: 'saturation', label: 'Saturation (0–3)', type: 'number', default: 1.3 },
            { key: 'brightness', label: 'Brightness (0–3)', type: 'number', default: 1.1 },
        ],
        'img_bg_removal':   [],   // no params needed
        'img_annotation':   [
            { key: 'msg', label: 'Annotation Text', type: 'text', default: 'Processed by IntelliCloud' },
        ],
        'img_batch_rename': [],   // no params needed

        // VIDEO (10) ─────────────────────────────────────────────────────────
        'vid_trimming':   [
            { key: 'start', label: 'Start (seconds)', type: 'number', default: 0 },
            { key: 'end',   label: 'End (seconds)',   type: 'number', default: 10 },
        ],
        'vid_compression': [
            { key: 'bitrate', label: 'Bitrate (Mbps, blank=auto CRF)', type: 'number', default: '' },
        ],
        'vid_remove_audio': [],   // no params needed
        'vid_cropping':    [
            { key: 'left', label: 'Crop X (px)', type: 'number', default: 0 },
            { key: 'top',  label: 'Crop Y (px)', type: 'number', default: 0 },
            { key: 'w',    label: 'Width (px)',  type: 'number', default: 640 },
            { key: 'h',    label: 'Height (px)', type: 'number', default: 480 },
        ],
        'vid_add_subtitles': [
            { key: 'text', label: 'Subtitle / Overlay Text', type: 'text', default: 'IntelliCloud' },
        ],
        'vid_format_conv': [
            { key: 'target', label: 'Target Format (mp4/avi/mkv)', type: 'text', default: 'mp4' },
        ],
        'vid_frame_extraction': [
            { key: 'every_seconds', label: 'Extract Every N Seconds', type: 'number', default: 5 },
        ],
        'vid_gif_creation': [
            { key: 'fps', label: 'GIF FPS', type: 'number', default: 10 },
        ],
        'vid_watermarking': [
            { key: 'text', label: 'Watermark Text', type: 'text', default: 'CLOUD_ID_01' },
        ],
        'vid_split_segments': [
            { key: 'parts', label: 'Number of Segments', type: 'number', default: 4 },
        ],

        // AUDIO (5) ──────────────────────────────────────────────────────────
        // aud_trimming uses 'from' and 'to' keys in backend
        'aud_trimming': [
            { key: 'from', label: 'Start (seconds)', type: 'number', default: 0 },
            { key: 'to',   label: 'End (seconds)',   type: 'number', default: 30 },
        ],
        'aud_format_conv': [
            { key: 'codec', label: 'Output Format (mp3 / wav)', type: 'text', default: 'mp3' },
        ],
        'aud_noise_red': [
            { key: 'sensitivity', label: 'Sensitivity (0–2)', type: 'number', default: 0.5 },
        ],
        'aud_normalization': [
            { key: 'level', label: 'Headroom dB (0–6)', type: 'number', default: 1.0 },
        ],
        'aud_split_track': [
            { key: 'parts', label: 'Number of Parts', type: 'number', default: 2 },
        ],

        // PDF (5) ─────────────────────────────────────────────────────────────
        'pdf_split':   [
            { key: 'range', label: 'Page Range (e.g. 1-3)', type: 'text', default: '1-2' },
        ],
        'pdf_password': [
            { key: 'pass', label: 'Password', type: 'text', default: 'cloud123' },
        ],
        'pdf_merge':      [],   // merges all PDFs uploaded — no extra params
        'pdf_to_office':  [],   // extracts text — no extra params
        'pdf_extraction': [],   // extracts text — no extra params
    };

    const fileInputRef = useRef(null);

    const DONE_AFTER = {
        'FEATURE_EXTRACT': ['AUTOENCODER','RF_PREDICT','SHAP_GEN','DQN_DECISION','EXECUTING','COLLECTING','COMPLETED'],
        'AUTOENCODER':     ['RF_PREDICT','SHAP_GEN','DQN_DECISION','EXECUTING','COLLECTING','COMPLETED'],
        'RF_PREDICT':      ['SHAP_GEN','DQN_DECISION','EXECUTING','COLLECTING','COMPLETED'],
        'SHAP_GEN':        ['DQN_DECISION','EXECUTING','COLLECTING','COMPLETED'],
        'DQN_DECISION':    ['EXECUTING','COLLECTING','COMPLETED'],
        'EXECUTING':       ['COLLECTING','COMPLETED'],
    };
    const isDone  = (stage) => (DONE_AFTER[stage] || []).includes(activeStage);
    const isCurrent = (stage) => activeStage === stage;

    useEffect(() => {
        socket.on('pipeline_update', (data) => {
            setPipelineEvents(prev => [...prev.slice(-15), data]);
            setCurrentStatus(data.message);
            setActiveStage(data.stage);
            if (data.stage === 'EXECUTING') setSelectedNode(data.node);
            if (data.stage === 'COMPLETED' || data.stage === 'ERROR') {
                setLoading(false);
                fetch(`${API_BASE_URL}/api/latest_shap`)
                    .then(r => r.json())
                    .then(d => { if (d.url) setShapUrl(`${API_BASE_URL}${d.url}?t=${Date.now()}`); })
                    .catch(() => {});
                setTimeout(() => { setActiveStage('IDLE'); setSelectedNode(null); }, 4000);
            }
        });
        return () => socket.off('pipeline_update');
    }, []);

    useEffect(() => {
        const config = paramsConfig[taskType] || [];
        const initial = {};
        config.forEach(c => initial[c.key] = c.default);
        setExtraParams(initial);
        setFile(null);
    }, [taskType]);

    const handleParamChange = (key, val) => setExtraParams(prev => ({ ...prev, [key]: val }));

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setPipelineEvents([]);
        setSelectedNode(null);
        setError(null);
        setResult(null);
        setActiveStage('STARTING');

        const formData = new FormData();
        formData.append('task_type', taskType);
        formData.append('priority', priority);
        formData.append('params', JSON.stringify(extraParams));

        if (isUpload) {
            const files = fileInputRef.current.files;
            if (!files || files.length === 0) { alert('Please upload a file.'); setLoading(false); return; }
            formData.append('file', files[0]);
            formData.append('input_size_mb', (files[0].size / (1024 * 1024)).toFixed(2));
        } else {
            formData.append('raw_data', manualText);
            formData.append('input_size_mb', 0.1);
        }

        try {
            const res = await axios.post(`${API_BASE_URL}/api/submit_task`, formData);
            setResult(res.data);
            setLoading(false);
        } catch (err) {
            setError(err.response?.data?.message || 'Master API unreachable.');
            setLoading(false);
        }
    };

    const currentAccept = taskOptions.find(o => o.id === taskType)?.accept || '*/*';

    const NodeCard = ({ name, isSelected, isMaster }) => (
        <motion.div
            animate={isSelected ? { scale: 1.05, borderColor: '#3b82f6' } : { scale: 1 }}
            className={`node-card ${isSelected ? 'active' : ''}`}
        >
            <div className={`node-dot ${isSelected ? 'working' : 'online'}`} />
            <Laptop size={26} color={isSelected ? '#3b82f6' : '#475569'} />
            <div className="node-label">{isMaster ? 'MASTER' : 'WORKER'}</div>
            <div className="node-addr">{name === 'LOCAL' ? '127.0.0.1' : (name || 'Standby')}</div>
        </motion.div>
    );

    const PIPELINE_STEPS = [
        { key: 'FEATURE_EXTRACT', label: 'Extract',  icon: <Search size={11}/> },
        { key: 'AUTOENCODER',     label: 'Encoder',  icon: <Layers size={11}/> },
        { key: 'RF_PREDICT',      label: 'RF',       icon: <BarChart size={11}/> },
        { key: 'SHAP_GEN',        label: 'SHAP',     icon: <Shield size={11}/> },
        { key: 'DQN_DECISION',    label: 'DQN',      icon: <Cpu size={11}/> },
        { key: 'EXECUTING',       label: 'Execute',  icon: <Server size={11}/> },
    ];

    return (
        <div className="app">
            <div className="container">
                {/* HEADER */}
                <header className="header">
                    <Cloud size={44} color="#3b82f6" />
                    <h1>IntelliCloud Dashboard</h1>
                    <p className="subtitle">Distributed DQN-Cluster · Real-Time AI Pipeline</p>
                </header>

                {/* TWO-COLUMN GRID — fixed equal width, never resizes */}
                <div className="main-grid">

                    {/* LEFT: Task Controller */}
                    <div className="panel card">
                        <form onSubmit={handleSubmit}>
                            <h3 className="panel-title"><Settings size={15}/> Workflow Controller</h3>

                            <div className="form-row">
                                <div className="form-group">
                                    <label>Task Domain</label>
                                    <select value={taskType} onChange={e => setTaskType(e.target.value)}>
                                        {taskOptions.map(o => <option key={o.id} value={o.id}>{o.name}</option>)}
                                    </select>
                                </div>
                                <div className="form-group">
                                    <label>DQN Priority</label>
                                    <select value={priority} onChange={e => setPriority(e.target.value)}>
                                        <option value="low">Low Energy</option>
                                        <option value="medium">Balanced</option>
                                        <option value="high">SLA First</option>
                                    </select>
                                </div>
                            </div>

                            {paramsConfig[taskType] && (
                                <div className="params-box">
                                    <div className="params-label"><Layers size={12}/> Parameters</div>
                                    <div className="params-grid">
                                        {paramsConfig[taskType].map(p => (
                                            <div key={p.key} className="param-field">
                                                <label>{p.label}</label>
                                                <input type={p.type} value={extraParams[p.key] || ''} onChange={e => handleParamChange(p.key, e.target.value)} />
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}

                            <div className="mode-tabs">
                                <button type="button" className={isUpload ? 'tab active' : 'tab'} onClick={() => setIsUpload(true)}><Upload size={13}/> Artifact</button>
                                <button type="button" className={!isUpload ? 'tab active' : 'tab'} onClick={() => setIsUpload(false)}><FileDigit size={13}/> Raw</button>
                            </div>

                            {isUpload ? (
                                <div className="dropzone" onClick={() => fileInputRef.current.click()}>
                                    <input ref={fileInputRef} type="file" accept={currentAccept} hidden onChange={e => setFile(e.target.files[0])} />
                                    <Upload size={26} color="#3b82f6" />
                                    <div className="drop-hint">
                                        {file
                                            ? <span className="drop-filename" title={file.name}>{file.name}</span>
                                            : <span className="drop-placeholder">Drop {currentAccept.replace('/*','').replace('.','')} file here</span>
                                        }
                                    </div>
                                </div>
                            ) : (
                                <textarea className="payload-box" value={manualText} onChange={e => setManualText(e.target.value)} placeholder="Enter raw task data..." />
                            )}

                            <button className="submit-btn" disabled={loading}>
                                {loading ? <Loader2 className="spin" size={18}/> : <Zap size={18}/>}
                                {loading ? 'Pipeline Active...' : 'Execute Pipeline'}
                            </button>
                        </form>
                    </div>

                    {/* RIGHT: Pipeline Visualization */}
                    <div className="panel card">
                        <h3 className="panel-title"><Activity size={15}/> Intelligent Execution Pipeline</h3>

                        {/* 6-step stepper */}
                        <div className="stepper">
                            <div className="stepper-line" />
                            {PIPELINE_STEPS.map(s => (
                                <div key={s.key} className={`step ${isDone(s.key) ? 'done' : ''} ${isCurrent(s.key) ? 'current' : ''}`}>
                                    <div className="step-dot">{s.icon}</div>
                                    <span>{s.label}</span>
                                </div>
                            ))}
                        </div>

                        {/* Node Cluster */}
                        <div className="cluster">
                            <NodeCard name="LOCAL" isSelected={selectedNode === 'LOCAL'} isMaster />
                            <div className="cluster-wire">
                                {loading && (
                                    <motion.div
                                        className="wire-packet"
                                        animate={{ x: [0, 64], opacity: [0, 1, 0] }}
                                        transition={{ repeat: Infinity, duration: 0.9 }}
                                    />
                                )}
                                <Wifi size={14} color="#334155" />
                                <ArrowRight size={13} color="#334155" />
                            </div>
                            <NodeCard name="Worker" isSelected={selectedNode && selectedNode !== 'LOCAL'} />
                        </div>

                        {/* Live Log */}
                        <div className="log-box">
                            <div className="log-status">
                                <span className={`status-pill ${loading ? 'active' : ''}`}>{currentStatus}</span>
                            </div>
                            <div className="log-scroll">
                                {pipelineEvents.length === 0
                                    ? <div className="log-idle">IDLE · Awaiting task dispatch...</div>
                                    : pipelineEvents.map((ev, i) => (
                                        <div key={i} className="log-row">
                                            <span className={`log-tag tag-${ev.stage}`}>{ev.stage}</span>
                                            <span className="log-msg">{ev.message}</span>
                                        </div>
                                    ))
                                }
                            </div>
                        </div>
                    </div>
                </div>

                {/* RESULTS DASHBOARD */}
                <AnimatePresence>
                    {result && (
                        <motion.div initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }} className="results">

                            {/* Telemetry Grid */}
                            <div className="card full-row">
                                <div className="card-head">
                                    <h3><Database size={18}/> SHERA Intelligent Telemetry</h3>
                                    <span className="eff-badge">{result.prediction.prediction.energy_efficiency_class}</span>
                                </div>
                                <div className="tele-grid">
                                    <div className="tele-cell"><div className="tele-label">Input Vol</div><div className="tele-val">{(result.prediction.features?.input_size_mb || 0).toFixed(2)} <em>MB</em></div></div>
                                    <div className="tele-cell"><div className="tele-label">Execute Site</div><div className="tele-val green">{result.execution.remote_node || 'MASTER'}</div></div>
                                    <div className="tele-cell"><div className="tele-label">Confidence</div><div className="tele-val">{(result.prediction.prediction.confidence * 100).toFixed(1)}<em>%</em></div></div>
                                    <div className="tele-cell"><div className="tele-label">Exec Time</div><div className="tele-val">{result.execution.duration.toFixed(3)} <em>s</em></div></div>
                                    <div className="tele-cell"><div className="tele-label">Allocation</div><div className="tele-val">{(result.execution?.metrics?.CPU / 100 || 0).toFixed(2)} <em>cores</em></div></div>
                                    <div className="tele-cell"><div className="tele-label">Mem Load</div><div className="tele-val">{(result.execution?.metrics?.MemoryMB || result.execution?.metrics?.MEM || 0).toFixed(0)} <em>MB</em></div></div>
                                    <div className="tele-cell"><div className="tele-label">Memory Used</div><div className="tele-val">{(result.execution?.metrics?.MemoryMB || result.execution?.metrics?.MEM || 0).toFixed(0)} <em>MB used</em></div></div>
                                    <div className="tele-cell"><div className="tele-label">Energy Used</div><div className="tele-val">{result.execution.energy.joules} <em>J</em> <span className="eff-inline">({result.prediction.prediction.energy_efficiency_class})</span></div></div>
                                </div>
                            </div>

                            {/* DQN + SHAP side-by-side */}
                            <div className="two-col">
                                <div className="card">
                                    <h3 className="card-title"><Server size={16}/> DQN Scheduler</h3>
                                    <div className="vm-board">
                                        <div className="vm-tier">{result.execution.vm}</div>
                                        <div className="vm-meta"><Activity size={14}/> {result.execution.duration.toFixed(3)}s latency</div>
                                    </div>
                                </div>

                                <div className="card shap-card" onClick={() => shapUrl && setShowShapModal(true)}>
                                    <h3 className="card-title"><Shield size={16}/> SHAP Correlation <Maximize size={12} style={{marginLeft:4}}/></h3>
                                    <div className="shap-preview">
                                        {shapUrl
                                            ? <img src={shapUrl} alt="SHAP" />
                                            : <div className="shap-wait">⏳ Generating...</div>
                                        }
                                    </div>
                                </div>
                            </div>

                            {/* Final Asset */}
                            <div className="card full-row">
                                <h3 className="card-title"><CheckCircle size={16}/> Final Processed Asset</h3>
                                <div className="asset-area">
                                    {/* VIDEO */}
                                    {result.execution.result_file.match(/\.(mp4|mov|avi|mkv)$/i) ? (
                                        <video controls className="asset-video" key={result.execution.result_file}>
                                            <source src={`${API_BASE_URL}${result.execution.result_file}`} type="video/mp4" />
                                        </video>
                                    ) : 
                                    /* IMAGE */
                                    result.execution.result_file.match(/\.(jpg|jpeg|png|webp|gif|bmp)$/i) ? (
                                        <img
                                            src={`${API_BASE_URL}${result.execution.result_file}?t=${Date.now()}`}
                                            alt="Result"
                                            className="asset-img"
                                            onError={e => { e.target.style.display = 'none'; e.target.nextSibling.style.display = 'flex'; }}
                                        />
                                    ) : 
                                    /* AUDIO */
                                    result.execution.result_file.match(/\.(mp3|wav|ogg|aac)$/i) ? (
                                        <div className="audio-preview">
                                            <Activity size={48} color="#3b82f6" style={{marginBottom: 16}} />
                                            <audio controls className="asset-audio" key={result.execution.result_file}>
                                                <source src={`${API_BASE_URL}${result.execution.result_file}`} type="audio/mpeg" />
                                            </audio>
                                        </div>
                                    ) : 
                                    /* PDF */
                                    result.execution.result_file.match(/\.pdf$/i) ? (
                                        <div className="pdf-preview">
                                            <iframe 
                                                src={`${API_BASE_URL}${result.execution.result_file}#toolbar=0`} 
                                                className="asset-pdf-frame"
                                                title="PDF Preview"
                                            />
                                        </div>
                                    ) : 
                                    /* OTHERS (ZIP, TXT, etc) */
                                    (
                                        <div className="file-stub-enhanced">
                                            {result.execution.result_file.match(/\.zip$/i) ? <Layers size={64} color="#f59e0b" /> : 
                                             result.execution.result_file.match(/\.txt$/i) ? <FileText size={64} color="#94a3b8" /> : 
                                             <FileDigit size={64} color="#64748b" />}
                                            <h3>{result.execution.result_file.split('/').pop()}</h3>
                                            <p>{result.execution.result_file.match(/\.zip$/i) ? 'Compressed Archive Result' : 'Document Data Result'}</p>
                                        </div>
                                    )}

                                    <div className="asset-actions">
                                        <a
                                            href={`${API_BASE_URL}${result.execution.result_file}`}
                                            download={result.execution.result_file.split('/').pop()}
                                            className="btn-download"
                                        >
                                            <Download size={16}/> Download to Device
                                        </a>
                                        <button className="btn-native" onClick={() => window.open(`${API_BASE_URL}${result.execution.result_file}`, '_blank')}>
                                            <Maximize size={16}/> View Fullscreen
                                        </button>
                                    </div>
                                </div>
                            </div>

                            {/* Energy Footer */}
                            <div className="card full-row energy-row">
                                <div className="energy-left">
                                    <Zap size={32} color="#3b82f6"/>
                                    <div>
                                        <div className="energy-j">{result.execution.energy.joules} J</div>
                                        <div className="energy-sub">Total Execution Energy</div>
                                    </div>
                                </div>
                                <div className="energy-right">
                                    <div className="energy-save">-{result.execution.energy.percent}%</div>
                                    <div className="energy-sub">DQN Green Savings</div>
                                </div>
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>

                {/* SHAP FULLSCREEN MODAL */}
                {showShapModal && shapUrl && (
                    <div className="shap-overlay" onClick={() => setShowShapModal(false)}>
                        <div className="shap-modal" onClick={e => e.stopPropagation()}>
                            <button className="shap-close" onClick={() => setShowShapModal(false)}>✕ Close</button>
                            <img src={shapUrl} alt="SHAP Full" className="shap-full-img" />
                        </div>
                    </div>
                )}
            </div>

            <style>{`
                @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=JetBrains+Mono:wght@400;700&display=swap');

                *, *::before, *::after { box-sizing: border-box; }

                .app { min-height: 100vh; background: #020617; color: #f8fafc; font-family: 'Inter', sans-serif; padding: 36px 20px 80px; }
                .container { max-width: 1200px; margin: 0 auto; }

                /* HEADER */
                .header { text-align: center; margin-bottom: 36px; }
                .header h1 { font-size: 2.6rem; font-weight: 800; background: linear-gradient(135deg, #3b82f6, #60a5fa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 8px 0 4px; }
                .subtitle { color: #475569; font-size: 0.75rem; letter-spacing: 2px; text-transform: uppercase; font-weight: 700; }

                /* MAIN 2-COLUMN GRID — FIXED, NEVER RESIZES */
                .main-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 36px; align-items: start; }
                .panel { min-width: 0; overflow: hidden; }

                /* CARD */
                .card { background: rgba(15,23,42,0.7); border: 1px solid rgba(255,255,255,0.06); border-radius: 20px; padding: 22px; backdrop-filter: blur(20px); }
                .panel-title, .card-title { font-size: 0.8rem; font-weight: 800; color: #3b82f6; margin-bottom: 20px; display: flex; align-items: center; gap: 8px; text-transform: uppercase; letter-spacing: 1px; white-space: nowrap; }

                /* FORM */
                .form-row { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 16px; }
                .form-group label { display: block; font-size: 0.65rem; color: #64748b; font-weight: 700; margin-bottom: 5px; text-transform: uppercase; }
                .form-group select { width: 100%; background: #0a0f1e; border: 1px solid #1e293b; color: #fff; padding: 10px; border-radius: 10px; outline: none; font-size: 0.82rem; }

                .params-box { background: rgba(0,0,0,0.25); border-radius: 14px; padding: 14px; margin-bottom: 16px; border: 1px solid rgba(59,130,246,0.12); }
                .params-label { font-size: 0.65rem; font-weight: 800; color: #3b82f6; margin-bottom: 12px; display: flex; align-items: center; gap: 6px; text-transform: uppercase; }
                .params-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); gap: 10px; }
                .param-field label { display: block; font-size: 0.6rem; color: #475569; margin-bottom: 3px; }
                .param-field input { width: 100%; background: #020617; border: 1px solid #1e293b; color: #fff; padding: 7px; border-radius: 6px; font-size: 0.78rem; }

                .mode-tabs { display: flex; gap: 10px; margin-bottom: 12px; }
                .tab { flex: 1; padding: 9px; border-radius: 10px; border: 1px solid #1e293b; background: none; color: #64748b; cursor: pointer; display: flex; align-items: center; justify-content: center; gap: 7px; font-weight: 700; font-size: 0.78rem; }
                .tab.active { background: rgba(59,130,246,0.1); color: #fff; border-color: #3b82f6; }

                /* DROPZONE — fixed height, no expansion from filename */
                .dropzone { border: 2px dashed #1e293b; border-radius: 14px; padding: 20px 14px; text-align: center; cursor: pointer; background: rgba(0,0,0,0.1); transition: 0.3s; height: 100px; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 8px; overflow: hidden; }
                .dropzone:hover { border-color: #3b82f6; background: rgba(59,130,246,0.05); }
                .drop-hint { max-width: 100%; overflow: hidden; }
                .drop-filename { color: #10b981; font-weight: 700; font-size: 0.8rem; display: block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 220px; }
                .drop-placeholder { color: #475569; font-size: 0.8rem; }
                .payload-box { width: 100%; height: 100px; background: #020617; border: 1px solid #1e293b; color: #fff; padding: 12px; border-radius: 12px; font-family: 'JetBrains Mono'; font-size: 0.78rem; resize: none; }

                .submit-btn { width: 100%; margin-top: 18px; background: #3b82f6; color: #fff; padding: 16px; border: none; border-radius: 14px; font-weight: 800; cursor: pointer; display: flex; align-items: center; justify-content: center; gap: 10px; text-transform: uppercase; letter-spacing: 1.5px; font-size: 0.82rem; transition: 0.2s; }
                .submit-btn:hover { background: #2563eb; transform: translateY(-2px); box-shadow: 0 6px 24px rgba(59,130,246,0.4); }
                .submit-btn:disabled { opacity: 0.5; transform: none; cursor: not-allowed; }

                /* PIPELINE STEPPER */
                .stepper { display: flex; justify-content: space-between; margin-bottom: 24px; position: relative; }
                .stepper-line { position: absolute; top: 11px; left: 4%; right: 4%; height: 2px; background: #1e293b; z-index: 0; }
                .step { display: flex; flex-direction: column; align-items: center; gap: 5px; z-index: 1; flex: 1; }
                .step-dot { width: 24px; height: 24px; background: #0f172a; border: 2px solid #334155; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: #334155; transition: 0.4s; }
                .step span { font-size: 0.5rem; font-weight: 800; color: #475569; text-transform: uppercase; letter-spacing: 0.5px; }
                .step.done .step-dot { border-color: #10b981; background: #10b981; color: #fff; box-shadow: 0 0 12px rgba(16,185,129,0.4); }
                .step.done span { color: #10b981; }
                .step.current .step-dot { border-color: #3b82f6; background: #3b82f6; color: #fff; animation: glow 1.4s infinite; }
                @keyframes glow { 0% { box-shadow: 0 0 0 0 rgba(59,130,246,0.5); } 70% { box-shadow: 0 0 0 8px rgba(59,130,246,0); } 100% { box-shadow: 0 0 0 0 rgba(59,130,246,0); } }

                /* NODE CLUSTER */
                .cluster { display: flex; align-items: center; justify-content: space-around; padding: 16px 0; border-bottom: 1px solid rgba(255,255,255,0.05); margin-bottom: 18px; }
                .node-card { background: #020617; border: 2px solid #1e293b; padding: 14px 10px; border-radius: 16px; width: 100px; min-width: 0; text-align: center; display: flex; flex-direction: column; align-items: center; gap: 5px; overflow: hidden; position: relative; }
                .node-card.active { border-color: #3b82f6; box-shadow: 0 0 20px rgba(59,130,246,0.2); }
                .node-dot { width: 8px; height: 8px; border-radius: 50%; position: absolute; top: 10px; right: 10px; }
                .node-dot.online { background: #10b981; }
                .node-dot.working { background: #3b82f6; animation: glow 1s infinite; }
                .node-label { font-size: 0.55rem; font-weight: 800; color: #64748b; text-transform: uppercase; letter-spacing: 1px; width: 100%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
                .node-addr  { font-size: 0.65rem; font-weight: 700; font-family: 'JetBrains Mono'; color: #cbd5e1; width: 100%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

                .cluster-wire { display: flex; flex-direction: column; align-items: center; gap: 4px; color: #334155; position: relative; width: 70px; }
                .wire-packet { width: 9px; height: 9px; background: #3b82f6; border-radius: 50%; position: absolute; top: 0; filter: blur(1px); }

                /* LIVE LOG */
                .log-box { background: #000; border-radius: 14px; padding: 14px; display: flex; flex-direction: column; border: 1px solid rgba(255,255,255,0.03); }
                .log-status { margin-bottom: 10px; }
                .status-pill { font-size: 0.65rem; background: #1e293b; color: #94a3b8; padding: 5px 12px; border-radius: 20px; font-weight: 800; display: inline-block; }
                .status-pill.active { background: rgba(16,185,129,0.1); color: #10b981; border: 1px solid #10b981; }
                .log-scroll { max-height: 280px; overflow-y: auto; font-family: 'JetBrains Mono'; }
                .log-idle { color: #334155; font-size: 0.7rem; padding: 12px 0; }
                .log-row { display: flex; gap: 12px; padding: 9px 0; border-bottom: 1px solid rgba(255,255,255,0.02); align-items: flex-start; }
                .log-tag { font-weight: 800; min-width: 85px; font-size: 0.55rem; border-radius: 4px; padding: 2px 5px; text-align: center; text-transform: uppercase; }
                .tag-FEATURE_EXTRACT { background: rgba(251,191,36,0.2); color: #fbbf24; }
                .tag-AUTOENCODER    { background: rgba(168,85,247,0.2); color: #a855f7; }
                .tag-RF_PREDICT     { background: rgba(236,72,153,0.2); color: #ec4899; }
                .tag-SHAP_GEN       { background: rgba(14,165,233,0.2); color: #0ea5e9; }
                .tag-DQN_DECISION   { background: rgba(59,130,246,0.2); color: #3b82f6; }
                .tag-EXECUTING      { background: rgba(16,185,129,0.2); color: #10b981; }
                .tag-COMPLETED      { background: rgba(16,185,129,0.3); color: #10b981; border: 1px solid #10b981; }
                .tag-ERROR          { background: rgba(239,68,68,0.2); color: #ef4444; }
                .log-msg { color: #94a3b8; font-size: 0.68rem; line-height: 1.5; }

                /* RESULTS */
                .results { display: flex; flex-direction: column; gap: 24px; margin-bottom: 60px; }
                .full-row { width: 100%; }
                .two-col  { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }

                .card-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 18px; }
                .card-head h3 { font-size: 0.85rem; font-weight: 800; display: flex; align-items: center; gap: 8px; }
                .eff-badge { background: #3b82f6; color: #fff; padding: 4px 14px; border-radius: 8px; font-weight: 900; font-size: 0.8rem; }

                .tele-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }
                .tele-cell { background: rgba(2,6,23,0.5); padding: 16px; border-radius: 14px; border: 1px solid rgba(255,255,255,0.03); }
                .tele-label { font-size: 0.55rem; color: #475569; text-transform: uppercase; font-weight: 800; letter-spacing: 1px; margin-bottom: 6px; }
                .tele-val { font-size: 1.1rem; font-weight: 800; color: #f8fafc; }
                .tele-val em { font-style: normal; font-size: 0.65rem; color: #64748b; margin-left: 3px; }
                .tele-val.green { color: #10b981; }
                .eff-inline { color: #3b82f6; font-weight: 900; font-size: 0.75rem; }

                .vm-board { background: linear-gradient(135deg, rgba(59,130,246,0.12), rgba(15,23,42,0.5)); border-radius: 24px; border: 1px solid rgba(59,130,246,0.2); padding: 40px 20px; text-align: center; margin-top: 14px; }
                .vm-tier  { font-size: 3.5rem; font-weight: 900; color: #fff; line-height: 1; }
                .vm-meta  { color: #4ade80; font-size: 0.9rem; font-weight: 700; margin-top: 12px; display: flex; align-items: center; justify-content: center; gap: 6px; }

                .shap-card { cursor: pointer; }
                .shap-preview { width: 100%; aspect-ratio: 16/9; border-radius: 16px; overflow: hidden; background: #000; margin-top: 14px; display: flex; align-items: center; justify-content: center; border: 1px solid #1e293b; }
                .shap-preview img { width: 100%; height: 100%; object-fit: contain; }
                .shap-wait { color: #475569; font-size: 0.8rem; font-weight: 700; }

                /* ASSET OUTPUT */
                .asset-area { margin-top: 14px; text-align: center; display: flex; flex-direction: column; align-items: center; gap: 20px; }
                .asset-img { max-width: 100%; max-height: 480px; border-radius: 14px; box-shadow: 0 10px 40px rgba(0,0,0,0.6); object-fit: contain; }
                .asset-video { width: 100%; max-width: 800px; border-radius: 16px; box-shadow: 0 10px 40px rgba(0,0,0,0.6); }
                
                .audio-preview { background: rgba(0,0,0,0.3); padding: 40px; border-radius: 20px; width: 100%; max-width: 500px; border: 1px solid rgba(59,130,246,0.1); }
                .asset-audio { width: 100%; border-radius: 30px; }
                
                .pdf-preview { width: 100%; height: 600px; border-radius: 14px; overflow: hidden; border: 1px solid #1e293b; background: #0f172a; }
                .asset-pdf-frame { width: 100%; height: 100%; border: none; }

                .file-stub-enhanced { background: #0f172a; padding: 60px 40px; border-radius: 20px; color: #94a3b8; border: 1px dashed #334155; display: flex; flex-direction: column; align-items: center; gap: 14px; width: 100%; max-width: 400px; }
                .file-stub-enhanced h3 { font-size: 1rem; color: #f1f5f9; margin: 0; word-break: break-all; }
                .file-stub-enhanced p { font-size: 0.75rem; color: #475569; margin: 0; text-transform: uppercase; letter-spacing: 1px; }

                .asset-actions { display: flex; justify-content: center; gap: 14px; margin-top: 10px; flex-wrap: wrap; }
                .btn-download { display: inline-flex; align-items: center; gap: 8px; background: #3b82f6; color: #fff; padding: 13px 26px; border-radius: 12px; text-decoration: none; font-weight: 800; font-size: 0.82rem; transition: 0.2s; }
                .btn-download:hover { background: #2563eb; transform: scale(1.04); }
                .btn-native { display: inline-flex; align-items: center; gap: 8px; background: rgba(51,65,85,0.5); color: #fff; border: 1px solid #334155; padding: 13px 26px; border-radius: 12px; font-weight: 800; cursor: pointer; font-size: 0.82rem; }

                /* ENERGY FOOTER */
                .energy-row { display: flex; justify-content: space-between; align-items: center; padding: 36px 48px; background: linear-gradient(to right, #1e3a8a, #020617); border: 2px solid #3b82f6; }
                .energy-left { display: flex; align-items: center; gap: 20px; }
                .energy-j { font-size: 2.8rem; font-weight: 900; color: #fff; line-height: 1; }
                .energy-sub { color: #64748b; font-size: 0.75rem; margin-top: 4px; text-transform: uppercase; letter-spacing: 1px; }
                .energy-save { font-size: 3.2rem; font-weight: 900; color: #10b981; line-height: 1; text-align: right; }

                /* SHAP MODAL — always fixed-centered */
                .shap-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.94); z-index: 9999; display: flex; align-items: center; justify-content: center; padding: 40px; backdrop-filter: blur(14px); cursor: pointer; }
                .shap-modal { position: relative; max-width: 88vw; max-height: 90vh; display: flex; flex-direction: column; align-items: center; gap: 14px; cursor: default; }
                .shap-close { background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); color: #fff; padding: 7px 18px; border-radius: 20px; cursor: pointer; font-weight: 700; font-size: 0.78rem; transition: 0.2s; align-self: flex-end; }
                .shap-close:hover { background: rgba(255,255,255,0.2); }
                .shap-full-img { max-width: 100%; max-height: 82vh; object-fit: contain; border-radius: 12px; box-shadow: 0 20px 60px rgba(0,0,0,0.8); }

                .spin { animation: spin 1s linear infinite; }
                @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
            `}</style>
        </div>
    );
};

export default App;
