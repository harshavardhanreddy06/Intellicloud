import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import io from 'socket.io-client';
import { 
  Zap, Cpu, Server, Activity, Database, Image as ImageIcon, 
  BarChart, FileText, Settings, Shield, Maximize, Cloud, Upload,
  Download, FileDigit, CpuIcon, CheckCircle, ExternalLink, Loader2,
  Search, Info, AlertTriangle, Layers, Copy, Check, Laptop, Wifi, ArrowRight, Play, Video
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

// Use dynamic hostname detection for the Master API
const API_BASE_URL = `http://${window.location.hostname}:5001`;
const socket = io(API_BASE_URL, { 
    transports: ['websocket'],
    reconnection: true 
});

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
    
    // Distributed State
    const [pipelineEvents, setPipelineEvents] = useState([]);
    const [selectedNode, setSelectedNode] = useState(null);
    const [currentStatus, setCurrentStatus] = useState('Ready for Task Dispatch');
    const [activeStage, setActiveStage] = useState('IDLE');

    const taskOptions = [
        // IMAGE (10)
        { id: 'img_resize', name: '🖼️ Image Resizing', category: 'image', accept: 'image/*' },
        { id: 'img_cropping', name: '🖼️ Image Cropping', category: 'image', accept: 'image/*' },
        { id: 'img_compression', name: '🖼️ Image Compression', category: 'image', accept: 'image/*' },
        { id: 'img_format_conv', name: '🖼️ Image Format Conv', category: 'image', accept: 'image/*' },
        { id: 'img_watermark', name: '🖼️ Image Watermarking', category: 'image', accept: 'image/*' },
        { id: 'img_puzzle_split', name: '🖼️ Image Puzzle Split', category: 'image', accept: 'image/*' },
        { id: 'img_color_corr', name: '🖼️ Image Color Corr', category: 'image', accept: 'image/*' },
        { id: 'img_bg_removal', name: '🖼️ Image BG Removal', category: 'image', accept: 'image/*' },
        { id: 'img_annotation', name: '🖼️ Image Annotation', category: 'image', accept: 'image/*' },
        { id: 'img_batch_rename', name: '🖼️ Image Batch Rename', category: 'image', accept: 'image/*' },
        
        // VIDEO (10)
        { id: 'vid_cropping', name: '🎥 Video Cropping', category: 'video', accept: 'video/*' },
        { id: 'vid_trimming', name: '🎥 Video Trimming', category: 'video', accept: 'video/*' },
        { id: 'vid_compression', name: '🎥 Video Compression', category: 'video', accept: 'video/*' },
        { id: 'vid_remove_audio', name: '🎥 Video Mute/Remove', category: 'video', accept: 'video/*' },
        { id: 'vid_add_subtitles', name: '🎥 Video Subtitles', category: 'video', accept: 'video/*' },
        { id: 'vid_format_conv', name: '🎥 Video Format Conv', category: 'video', accept: 'video/*' },
        { id: 'vid_frame_extraction', name: '🎥 Video Frame Extract', category: 'video', accept: 'video/*' },
        { id: 'vid_gif_creation', name: '🎥 Video GIF creation', category: 'video', accept: 'video/*' },
        { id: 'vid_watermarking', name: '🎥 Video Watermarking', category: 'video', accept: 'video/*' },
        { id: 'vid_split_segments', name: '🎥 Video Split Segments', category: 'video', accept: 'video/*' },
        
        // AUDIO (5)
        { id: 'aud_noise_red', name: '🎧 Audio Noise Red', category: 'audio', accept: 'audio/*' },
        { id: 'aud_format_conv', name: '🎧 Audio Format Conv', category: 'audio', accept: 'audio/*' },
        { id: 'aud_trimming', name: '🎧 Audio Trimming', category: 'audio', accept: 'audio/*' },
        { id: 'aud_normalization', name: '🎧 Audio Normalization', category: 'audio', accept: 'audio/*' },
        { id: 'aud_split_track', name: '🎧 Audio Split Tracks', category: 'audio', accept: 'audio/*' },
        
        // PDF (5)
        { id: 'pdf_merge', name: '📑 PDF Merge', category: 'pdf', accept: '.pdf' },
        { id: 'pdf_split', name: '📑 PDF Split Pages', category: 'pdf', accept: '.pdf' },
        { id: 'pdf_to_office', name: '📑 PDF to Office', category: 'pdf', accept: '.pdf' },
        { id: 'pdf_password', name: '📑 PDF Password Protect', category: 'pdf', accept: '.pdf' },
        { id: 'pdf_extraction', name: '📑 PDF Extract Data', category: 'pdf', accept: '.pdf' },
    ];

    const paramsConfig = {
        'img_resize': [{ key: 'width', label: 'Width (px)', type: 'number', default: 800 }, { key: 'height', label: 'Height (px)', type: 'number', default: 600 }],
        'img_cropping': [{ key: 'left', label: 'Left (px)', type: 'number', default: 0 }, { key: 'top', label: 'Top (px)', type: 'number', default: 0 }, { key: 'width', label: 'Width', type: 'number', default: 400 }, { key: 'height', label: 'Height', type: 'number', default: 400 }],
        'img_compression': [{ key: 'quality', label: 'Quality (1-100)', type: 'number', default: 80 }],
        'img_watermark': [{ key: 'text', label: 'Watermark Text', type: 'text', default: 'IntelliCloud' }],
        'vid_cropping': [{ key: 'left', label: 'Crop Left', type: 'number', default: 0 }, { key: 'top', label: 'Crop Top', type: 'number', default: 0 }],
        'vid_trimming': [{ key: 'start', label: 'Start (s)', type: 'number', default: 0 }, { key: 'end', label: 'End (s)', type: 'number', default: 10 }],
        'aud_noise_red': [{ key: 'sensitivity', label: 'Sensitivity', type: 'number', default: 0.5 }],
        'pdf_merge': [{ key: 'files', label: 'File List', type: 'text', default: '1.pdf, 2.pdf' }]
    };

    const fileInputRef = useRef(null);

    useEffect(() => {
        socket.on('pipeline_update', (data) => {
            setPipelineEvents(prev => [...prev.slice(-15), data]);
            setCurrentStatus(data.message);
            setActiveStage(data.stage);
            if (data.stage === 'EXECUTING') setSelectedNode(data.node);
            if (data.stage === 'COMPLETED' || data.stage === 'ERROR') { 
                setLoading(false); 
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
        setFile(null); // Clear file when task type changes for safety
    }, [taskType]);

    const handleParamChange = (key, val) => {
        setExtraParams(prev => ({ ...prev, [key]: val }));
    };

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
            if (!files || files.length === 0) { alert('Please upload file(s).'); setLoading(false); return; }
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
            console.error(err);
            setError(err.response?.data?.message || 'Master API unreachable.');
            setLoading(false);
        }
    };

    const NodeIcon = ({ name, isSelected, isMaster }) => (
        <motion.div 
            animate={isSelected ? { scale: 1.1, borderColor: '#3b82f6', boxShadow: '0 0 25px rgba(59,130,246,0.3)' } : { scale: 1 }}
            className={`node-card ${isSelected ? 'active' : ''} ${isMaster ? 'master' : ''}`}
        >
            <div className={`node-status-dot ${isSelected ? 'working' : 'online'}`}></div>
            <Laptop size={32} color={isSelected ? '#3b82f6' : '#64748b'} />
            <div className="node-info">
                <span className="node-name">{isMaster ? 'MASTER' : 'WORKER'}</span>
                <span className="node-ip">{name === 'LOCAL' ? '127.0.0.1' : name}</span>
            </div>
            {isSelected && <motion.div layoutId="glow" className="node-glow" />}
        </motion.div>
    );

    // Get current accept type
    const currentAccept = taskOptions.find(o => o.id === taskType)?.accept || '*/*';

    return (
        <div className="App dark-theme">
            <div className="container">
                <header className="header">
                    <motion.div initial={{ y: -20, opacity: 0 }} animate={{ y: 0, opacity: 1 }}>
                        <Cloud size={52} color="#3b82f6" style={{ margin: '0 auto 10px' }} />
                        <h1>IntelliCloud Dashboard</h1>
                        <p className="subtitle">Distributed Distributed DQN-Cluster • Real-Time States</p>
                    </motion.div>
                </header>

                <div className="distributed-grid">
                    <div className="card orchestration-panel">
                        <form onSubmit={handleSubmit}>
                            <h3 className="section-title"><Settings size={18}/> Workflow Controller</h3>
                            <div className="form-row">
                                <div className="form-group">
                                    <label>Task Domain</label>
                                    <select value={taskType} onChange={(e) => setTaskType(e.target.value)}>
                                        {taskOptions.map(opt => <option key={opt.id} value={opt.id}>{opt.name}</option>)}
                                    </select>
                                </div>
                                <div className="form-group">
                                    <label>DQN QoS Priority</label>
                                    <select value={priority} onChange={(e) => setPriority(e.target.value)}>
                                        <option value="low">Low Energy</option>
                                        <option value="medium">Balanced</option>
                                        <option value="high">SLA First</option>
                                    </select>
                                </div>
                            </div>

                            {paramsConfig[taskType] && (
                                <div className="params-panel-restored">
                                    <h4 className="params-label"><Layers size={14}/> Parameters</h4>
                                    <div className="params-grid">
                                        {paramsConfig[taskType].map(p => (
                                            <div key={p.key} className="form-group-mini">
                                                <label>{p.label}</label>
                                                <input type={p.type} value={extraParams[p.key] || ''} onChange={(e) => handleParamChange(p.key, e.target.value)} />
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}

                            <div className="upload-section">
                                <div className="mode-selector">
                                    <button type="button" className={isUpload ? 'active' : ''} onClick={() => setIsUpload(true)}><Upload size={14}/> Artifact</button>
                                    <button type="button" className={!isUpload ? 'active' : ''} onClick={() => setIsUpload(false)}><FileDigit size={14}/> Raw</button>
                                </div>
                                {isUpload ? (
                                    <div className="dropzone" onClick={() => fileInputRef.current.click()}>
                                        <input ref={fileInputRef} type="file" accept={currentAccept} hidden onChange={(e) => setFile(e.target.files[0])} />
                                        <div className="icon-stack">
                                            <Upload size={32} color="#3b82f6" />
                                            <Database size={16} className="sub-icon" />
                                        </div>
                                        <p>{file ? <span className="selected-text">{file.name}</span> : `Select ${currentAccept} Payload`}</p>
                                    </div>
                                ) : (
                                    <textarea className="payload-box" value={manualText} onChange={(e) => setManualText(e.target.value)} placeholder="Entry task rows..."/>
                                )}
                            </div>

                            <button className="submit-btn" disabled={loading}>
                                {loading ? <Loader2 className="spin" size={20}/> : <Zap size={20}/>}
                                {loading ? "Active Dispatch..." : "Execute Pipeline"}
                            </button>
                        </form>
                    </div>

                    <div className="card pipeline-panel">
                        <h3 className="section-title"><Activity size={18}/> Intelligent Execution Pipeline</h3>
                        
                        <div className="pipeline-stepper-6">
                           {/* Points turn green ONLY when the PREVIOUS stage is done (based on activeStage received) */}
                           <div className={`step-6 ${['AUTOENCODER', 'RF_PREDICT', 'SHAP_GEN', 'DQN_DECISION', 'EXECUTING', 'COLLECTING', 'COMPLETED'].includes(activeStage) ? 'done' : ''} ${activeStage === 'FEATURE_EXTRACT' ? 'current' : ''}`}>
                             <div className="step-point"><Search size={12}/></div>
                             <span>Extract</span>
                           </div>
                           <div className={`step-6 ${['RF_PREDICT', 'SHAP_GEN', 'DQN_DECISION', 'EXECUTING', 'COLLECTING', 'COMPLETED'].includes(activeStage) ? 'done' : ''} ${activeStage === 'AUTOENCODER' ? 'current' : ''}`}>
                             <div className="step-point"><Layers size={12}/></div>
                             <span>Encoder</span>
                           </div>
                           <div className={`step-6 ${['SHAP_GEN', 'DQN_DECISION', 'EXECUTING', 'COLLECTING', 'COMPLETED'].includes(activeStage) ? 'done' : ''} ${activeStage === 'RF_PREDICT' ? 'current' : ''}`}>
                             <div className="step-point"><BarChart size={12}/></div>
                             <span>RF Predict</span>
                           </div>
                           <div className={`step-6 ${['DQN_DECISION', 'EXECUTING', 'COLLECTING', 'COMPLETED'].includes(activeStage) ? 'done' : ''} ${activeStage === 'SHAP_GEN' ? 'current' : ''}`}>
                             <div className="step-point"><Shield size={12}/></div>
                             <span>SHAP</span>
                           </div>
                           <div className={`step-6 ${['EXECUTING', 'COLLECTING', 'COMPLETED'].includes(activeStage) ? 'done' : ''} ${activeStage === 'DQN_DECISION' ? 'current' : ''}`}>
                             <div className="step-point"><Cpu size={12}/></div>
                             <span>DQN</span>
                           </div>
                           <div className={`step-6 ${['COLLECTING', 'COMPLETED'].includes(activeStage) ? 'done' : ''} ${activeStage === 'EXECUTING' ? 'current' : ''}`}>
                             <div className="step-point"><Server size={12}/></div>
                             <span>Execute</span>
                           </div>
                        </div>

                        <div className="node-cluster">
                            <NodeIcon name="LOCAL" isSelected={selectedNode === 'LOCAL'} isMaster />
                            <div className="network-path">
                                <motion.div 
                                    className="data-packet"
                                    style={{ background: ['EXECUTING', 'COLLECTING', 'COMPLETED'].includes(activeStage) ? '#10b981' : '#3b82f6' }}
                                    animate={loading ? { x: [0, 80], opacity: [0, 1, 0] } : {}}
                                    transition={{ repeat: Infinity, duration: 1.0 }}
                                />
                                <Wifi size={16} />
                                <ArrowRight size={16} style={{ marginTop: 5 }} />
                            </div>
                            <NodeIcon name="Remote Node" isSelected={selectedNode && selectedNode !== 'LOCAL'} />
                        </div>

                        <div className="live-log">
                            <div className="log-header"><span className={loading ? "status-badge active-task" : "status-badge"}>{currentStatus}</span></div>
                            <div className="log-scroll">
                                {pipelineEvents.map((ev, i) => (
                                    <div key={i} className="log-entry">
                                        <span className={`log-stage stage-${ev.stage}`}>{ev.stage}</span>
                                        <span className="log-msg">{ev.message}</span>
                                    </div>
                                ))}
                                {pipelineEvents.length === 0 && <div className="log-empty">IDLE • Awaiting input.</div>}
                            </div>
                        </div>
                    </div>
                </div>

                <AnimatePresence>
                    {result && (
                        <motion.div initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }} className="full-dashboard">
                            <div className="card stat-card full-width">
                                <div className="card-header-flex">
                                    <h3><Database size={20}/> SHERA Intelligent Telemetry</h3>
                                    <div className="efficiency-badge">{result.prediction.prediction.energy_efficiency_class}</div>
                                </div>
                                <div className="stat-grid-12">
                                    <div className="stat-blob"><span>Input Vol</span><strong>{(result.prediction.features?.input_size_mb || 0).toFixed(2)} MB</strong></div>
                                    <div className="stat-blob"><span>Execute Site</span><strong className="green">{result.execution.remote_node || "MASTER"}</strong></div>
                                    <div className="stat-blob"><span>Confidence</span><strong>{(result.prediction.prediction.confidence * 100).toFixed(1)}%</strong></div>
                                    <div className="stat-blob"><span>Complexity</span><strong>{(result.prediction.features?.complexity_score || 0).toFixed(2)}</strong></div>
                                    <div className="stat-blob"><span>Allocation</span><strong>{(result.execution?.metrics?.CPU / 100 || 0).toFixed(2)} c</strong></div>
                                    <div className="stat-blob"><span>Mem Load</span><strong>{(result.execution?.metrics?.MemoryMB || result.execution?.metrics?.MEM || 0).toFixed(1)} MB</strong></div>
                                    <div className="stat-blob"><span>Hist. Delta</span><strong>{(result.prediction.features?.historical_lat_ms || 0).toFixed(1)} ms</strong></div>
                                    <div className="stat-blob"><span>SLA Index</span><strong>{result.prediction.features?.priority_level}</strong></div>
                                </div>
                            </div>

                            <div className="results-grid-2">
                                <div className="card target-vm-card">
                                    <h3><Server size={20}/> DQN Scheduler</h3>
                                    <div className="vm-billboard">
                                        <div className="vm-name">{result.execution.vm}</div>
                                        <div className="vm-metrics">
                                            <div className="v-metric"><Activity size={16}/> {result.execution.duration.toFixed(3)}s Latency</div>
                                        </div>
                                    </div>
                                </div>
                                <div className="card visual-card" onClick={() => setShowShapModal(true)}>
                                    <h3><Shield size={20}/> Analysis (SHAP Correlation)</h3>
                                    <div className="img-contain">
                                        <img src={`${API_BASE_URL}/shap_explanations/${result.prediction.prediction.explanation_image.split('/').pop()}`} alt="SHAP" />
                                    </div>
                                </div>
                            </div>

                            <div className="card output-card full-width">
                                <h3><CheckCircle size={20}/> Final Processed Asset</h3>
                                <div className="output-content">
                                    {result.execution.result_file.match(/\.(mp4|mov|avi|mkv)$/) ? (
                                        <div className="video-player-container">
                                            <video controls className="video-player" key={result.execution.result_file}>
                                                <source src={`${API_BASE_URL}${result.execution.result_file}`} type="video/mp4" />
                                            </video>
                                        </div>
                                    ) : result.execution.result_file.match(/\.(jpg|png|webp|gif)$/) ? (
                                        <img src={`${API_BASE_URL}${result.execution.result_file}`} alt="Result" />
                                    ) : (
                                        <div className="file-placeholder"><FileText size={64} /><p>{result.execution.result_file.split('/').pop()}</p></div>
                                    )}
                                    <div className="action-row">
                                        <a href={`${API_BASE_URL}${result.execution.result_file}`} download className="download-btn"><Download size={18}/> Download</a>
                                        <button className="secondary-btn" onClick={() => window.open(`${API_BASE_URL}${result.execution.result_file}`, '_blank')}><Maximize size={18}/> Native</button>
                                    </div>
                                </div>
                            </div>

                            <div className="card energy-footer-card full-width">
                                <div className="footer-left"><Zap size={36} className="lightning-glow"/><div><div className="joules-val">{result.execution.energy.joules} J</div><div className="joules-label">Impact</div></div></div>
                                <div className="footer-right"><div className="saving-percent">-{result.execution.energy.percent}%</div><div className="saving-label">DQN Savings Index</div></div>
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>

                {/* SHAP MODAL */}
                {showShapModal && result && (
                    <div className="modal-overlay" onClick={() => setShowShapModal(false)}>
                        <img src={`${API_BASE_URL}/shap_explanations/${result.prediction.prediction.explanation_image.split('/').pop()}`} alt="SHAP Full" />
                    </div>
                )}
            </div>

            <style>{`
                @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=JetBrains+Mono&display=swap');
                .App { min-height: 100vh; background: #020617; color: #f8fafc; font-family: 'Inter', sans-serif; padding: 40px 20px; }
                .container { max-width: 1200px; margin: 0 auto; }
                .header { text-align: center; margin-bottom: 40px; }
                .header h1 { font-size: 2.8rem; font-weight: 800; background: linear-gradient(135deg, #3b82f6, #60a5fa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
                .subtitle { color: #475569; letter-spacing: 2px; text-transform: uppercase; font-size: 0.75rem; text-align: center; }

                .distributed-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 40px; }
                .card { background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(59, 130, 246, 0.2); border-radius: 20px; padding: 24px; backdrop-filter: blur(20px); border: 1px solid rgba(255,255,255,0.05); }
                .section-title { font-size: 0.85rem; font-weight: 800; color: #3b82f6; margin-bottom: 24px; display: flex; align-items: center; gap: 10px; text-transform: uppercase; }

                .pipeline-stepper-6 { display: flex; justify-content: space-between; margin-bottom: 30px; position: relative; }
                .pipeline-stepper-6::before { content: ''; position: absolute; top: 12px; left: 5%; right: 5%; height: 2px; background: #1e293b; z-index: 1; }
                .step-6 { display: flex; flex-direction: column; align-items: center; gap: 6px; z-index: 2; flex: 1; }
                .step-point { width: 26px; height: 26px; background: #0f172a; border: 2px solid #334155; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: #334155; transition: 0.5s; z-index: 3; }
                .step-6 span { font-size: 0.55rem; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.5px; }
                .step-6.done .step-point { border-color: #10b981; color: #fff; background: #10b981; box-shadow: 0 0 15px rgba(16,185,129,0.5); }
                .step-6.done span { color: #10b981; }
                .step-6.current .step-point { border-color: #3b82f6; color: #fff; background: #3b82f6; animation: activeGlow 1.5s infinite; }
                @keyframes activeGlow { 0% { box-shadow: 0 0 0 0 rgba(59,130,246,0.5); } 70% { box-shadow: 0 0 0 10px rgba(59,130,246,0); } 100% { box-shadow: 0 0 0 0 rgba(59,130,246,0); } }

                .form-row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 20px; }
                .form-group label { display: block; font-size: 0.7rem; color: #64748b; margin-bottom: 8px; font-weight: 600; }
                .form-group select { width: 100%; background: #0a0f1e; border: 1px solid #1e293b; color: #fff; padding: 12px; border-radius: 10px; outline: none; }

                .params-panel-restored { background: rgba(0,0,0,0.3); border-radius: 16px; padding: 20px; margin-bottom: 20px; border: 1px solid rgba(59,130,246,0.15); }
                .params-label { font-size: 0.75rem; font-weight: 800; color: #3b82f6; margin-bottom: 15px; display: flex; align-items: center; gap: 8px; text-transform: uppercase; }
                .params-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 12px; }
                .form-group-mini label { display: block; font-size: 0.65rem; color: #475569; margin-bottom: 4px; }
                .form-group-mini input { width: 100%; background: #020617; border: 1px solid #1e293b; color: #fff; padding: 8px; border-radius: 6px; }

                .mode-selector { display: flex; gap: 12px; margin-bottom: 16px; }
                .mode-selector button { flex: 1; padding: 10px; border-radius: 10px; border: 1px solid #1e293b; background: none; color: #64748b; cursor: pointer; display: flex; align-items: center; justify-content: center; gap: 8px; font-weight: 700; }
                .mode-selector button.active { background: rgba(59,130,246,0.1); color: #fff; border-color: #3b82f6; }

                .dropzone { border: 2px dashed #1e293b; border-radius: 16px; padding: 35px; text-align: center; cursor: pointer; transition: 0.3s; background: rgba(0,0,0,0.1); }
                .dropzone:hover { border-color: #3b82f6; background: rgba(59, 130, 246, 0.05); }
                .selected-text { color: #10b981; font-weight: 800; }
                .payload-box { width: 100%; height: 140px; background: #020617; border: 1px solid #1e293b; color: #fff; padding: 15px; border-radius: 12px; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; resize: none; }

                .submit-btn { width: 100%; margin-top: 24px; background: #3b82f6; color: #fff; padding: 18px; border: none; border-radius: 16px; font-weight: 800; cursor: pointer; display: flex; align-items: center; justify-content: center; gap: 12px; text-transform: uppercase; letter-spacing: 1px; }

                .node-cluster { display: flex; align-items: center; justify-content: space-around; padding: 30px 0; border-bottom: 1px solid rgba(255,255,255,0.05); margin-bottom: 25px; }
                .node-card { background: #020617; border: 2px solid transparent; padding: 20px; border-radius: 20px; position: relative; width: 130px; text-align: center; }
                .node-card.active { border-color: #3b82f6; box-shadow: 0 0 30px rgba(59,130,246,0.2); }
                .node-ip { font-size: 0.8rem; font-weight: 800; font-family: 'JetBrains Mono'; color: #cbd5e1; }
                
                .network-path { display: flex; flex-direction: column; align-items: center; color: #1e293b; position: relative; width: 80px; }
                .data-packet { width: 10px; height: 10px; border-radius: 50%; position: absolute; top: 0; filter: blur(1px); }

                .live-log { background: #000; border-radius: 16px; padding: 20px; height: 350px; display: flex; flex-direction: column; border: 1px solid rgba(255,255,255,0.03); }
                .log-scroll { overflow-y: auto; flex: 1; font-size: 0.7rem; font-family: 'JetBrains Mono'; }
                .log-entry { display: flex; gap: 15px; padding: 12px 0; border-bottom: 1px solid rgba(255,255,255,0.02); }
                .log-stage { font-weight: 800; min-width: 95px; font-size: 0.6rem; text-transform: uppercase; border-radius: 4px; padding: 2px 6px; text-align: center; }
                .stage-STARTING { background: rgba(71, 85, 105, 0.2); color: #94a3b8; }
                .stage-ANALYSING { background: rgba(251, 191, 36, 0.2); color: #fbbf24; }
                .stage-RF_PREDICT { background: rgba(168, 85, 247, 0.2); color: #a855f7; }
                .stage-EXECUTING { background: rgba(59, 130, 246, 0.2); color: #3b82f6; }
                .stage-COMPLETED { background: rgba(16, 185, 129, 0.3); color: #10b981; border: 1px solid #10b981; }
                
                .status-badge { font-size: 0.7rem; background: #1e293b; color: #94a3b8; padding: 6px 15px; border-radius: 20px; font-weight: 800; }
                .active-task { background: rgba(16, 185, 129, 0.1); color: #10b981; border: 1px solid #10b981; }

                .full-dashboard { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 60px; }
                .full-width { grid-column: span 2; }
                .efficiency-badge { padding: 6px 16px; background: #3b82f6; color: #fff; border-radius: 8px; font-weight: 900; }
                
                .stat-grid-12 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; }
                .stat-blob { background: rgba(2, 6, 23, 0.5); padding: 20px; border-radius: 16px; }
                .stat-blob span { display: block; font-size: 0.6rem; color: #475569; margin-bottom: 8px; text-transform: uppercase; font-weight: 800; }
                .stat-blob strong { font-size: 1.2rem; color: #f8fafc; font-weight: 800; }
                .stat-blob strong.green { color: #10b981; }
                .results-grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; width: 100%; margin-bottom: 24px; grid-column: span 2; }
                .target-vm-card, .visual-card { height: 100%; display: flex; flex-direction: column; }
                
                .vm-billboard { text-align: center; background: linear-gradient(135deg, rgba(59,130,246,0.1), rgba(15,23,42,0.4)); padding: 45px 20px; border-radius: 30px; border: 1px solid rgba(59, 130, 246, 0.2); margin-top: 20px; flex: 1; display: flex; flex-direction: column; justify-content: center; }
                .vm-name { font-size: 4rem; font-weight: 900; color: #fff; line-height: 1; }

                .img-contain { width: 100%; aspect-ratio: 16/9; overflow: hidden; border-radius: 20px; background: #000; margin-top: 20px; cursor: pointer; border: 1px solid #1e293b; }
                .img-contain img { width: 100%; height: 100%; object-fit: contain; }

                .video-player-container { width: 100%; max-width: 800px; margin: 0 auto; border-radius: 20px; overflow: hidden; background: #000; border: 1px solid #1e293b; }
                .video-player { width: 100%; display: block; }
                .action-row { display: flex; justify-content: center; gap: 15px; margin-top: 30px; }
                .secondary-btn { background: rgba(51, 65, 85, 0.5); color: #fff; border: 1px solid #334155; padding: 14px 28px; border-radius: 12px; font-weight: 800; cursor: pointer; display: flex; align-items: center; gap: 10px; }
                
                .energy-footer-card { padding: 40px 60px; background: linear-gradient(to right, #1e3a8a, #020617); border: 2px solid #3b82f6; }
                .joules-val { font-size: 3rem; font-weight: 900; color: #fff; }
                .saving-percent { font-size: 3.5rem; font-weight: 900; color: #10b981; line-height: 1; }

                .spin { animation: spin 1s linear infinite; }
                @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
            `}</style>
        </div>
    );
};

export default App;
