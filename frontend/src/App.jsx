import React, { useState, useEffect, useRef } from 'react';
import GrooveStudioTab from './GrooveStudioTab';
import GrooveTuningSection from './GrooveTuningSection';
import {
  Cpu,
  Zap,
  Play,
  Square,
  Upload,
  Video,
  Image as ImageIcon,
  Terminal,
  Activity,
  CheckCircle2,
  XCircle,
  Download,
  BarChart3,
  RefreshCw,
  Sliders,
  Maximize2,
  Layers,
  Clock,
  Sparkles,
  AlertCircle,
  GitCompare,
  ShieldCheck,
  ScanLine,
  ArrowRight,
  FileVideo,
  Check,
  ExternalLink
} from 'lucide-react';

// Dynamic CPU latency estimation helper based on exact empirical benchmarks (video per-frame latency)
const CPU_LATENCY_ESTIMATES = {
  1: '~9.5s',
  2: '~4.6s',
  3: '~3.4s',
  4: '~2.5s',
  5: '~1.9s',
  6: '~1.8s',
  7: '~1.7s',
  8: '~1.6s',
  9: '~1.6s',
  10: '~1.5s',
  11: '~1.5s',
  12: '~1.5s',
};

const getCpuLatency = (cores) => {
  const c = Math.max(1, Math.min(12, parseInt(cores, 10) || 4));
  return CPU_LATENCY_ESTIMATES[c] || '~2.1s';
};

const CPU_PRESETS = [
  { label: '2 Cores', cores: 2 },
  { label: '4 Cores', cores: 4 },
  { label: '5 Cores', cores: 5 },
  { label: '6 Cores (All Physical)', cores: 6 },
  { label: '12 Threads (Max Boost)', cores: 12 },
];

export default function App() {
  // Navigation: 'groove_studio' (prominent default for nose_lines), 'two_video', 'viewer', 'benchmark'
  const [activeTab, setActiveTab] = useState('groove_studio');

  // System status
  const [status, setStatus] = useState(null);
  const [loadingStatus, setLoadingStatus] = useState(true);

  // Active SSE connection
  const [eventSource, setEventSource] = useState(null);

  // ==========================================
  // TAB 1: 2-Video Muzzle Matching State
  // ==========================================
  const [compareVideoA, setCompareVideoA] = useState('/home/adis_mayur/F_CODES/muzzleVideo.mp4');
  const [compareVideoB, setCompareVideoB] = useState('/home/adis_mayur/F_CODES/muzzleVideo2.mp4');
  const [videoAMode, setVideoAMode] = useState('sample'); // 'sample' | 'upload'
  const [videoBMode, setVideoBMode] = useState('sample');
  const [uploadedNameA, setUploadedNameA] = useState('');
  const [uploadedNameB, setUploadedNameB] = useState('');
  const [uploadingA, setUploadingA] = useState(false);
  const [uploadingB, setUploadingB] = useState(false);

  // Hardware engine for matching
  const [compareDevice, setCompareDevice] = useState('gpu'); // 'gpu', 'cpu'
  const [compareCpuCores, setCompareCpuCores] = useState(4); // 1-12 CPU cores, default 4

  // Preprocessing & Matching algorithm parameters
  const [compareUpscale, setCompareUpscale] = useState('lanczos4'); // 'lanczos4', 'bicubic', 'bilinear'
  const [compareScale, setCompareScale] = useState(2);
  const [compareThreshold, setCompareThreshold] = useState(35); // statusprint.py duplicate threshold
  const [compareInterval, setCompareInterval] = useState(10);
  const [compareMaxFrames, setCompareMaxFrames] = useState(6);
  const [compareFastMode, setCompareFastMode] = useState(true); // 640px + Early Exit for fast CPU processing

  // Compare execution & results
  const [compareRunning, setCompareRunning] = useState(false);
  const [compareStep, setCompareStep] = useState(0); // 0: idle, 1: vid A, 2: vid B, 3: upscale, 4: ORB, 5: done
  const [compareLogs, setCompareLogs] = useState([]);
  const [compareResult, setCompareResult] = useState(null);

  // ==========================================
  // TAB 2: Single-Video Frame Extraction State
  // ==========================================
  const [device, setDevice] = useState('gpu');
  const [cpuCores, setCpuCores] = useState(4); // 1-12 CPU cores, default 4
  const [inputMode, setInputMode] = useState('sample_video');
  const [selectedSampleVideo, setSelectedSampleVideo] = useState('/home/adis_mayur/F_CODES/muzzleVideo.mp4');
  const [selectedSampleImage, setSelectedSampleImage] = useState('cow_video_frame_1.jpg');
  const [uploadedFile, setUploadedFile] = useState(null);
  const [uploadedFilePath, setUploadedFilePath] = useState(null);
  const [frameInterval, setFrameInterval] = useState(10);
  const [maxFrames, setMaxFrames] = useState(6);
  const [confidence, setConfidence] = useState(0.70);
  const [square, setSquare] = useState(true);
  const [resize, setResize] = useState(224);

  const [isProcessing, setIsProcessing] = useState(false);
  const [logs, setLogs] = useState([]);
  const [latestFrame, setLatestFrame] = useState(null);
  const [latestCrop, setLatestCrop] = useState(null);
  const [allCrops, setAllCrops] = useState([]);
  const [metrics, setMetrics] = useState({
    processedFrames: 0,
    totalMuzzles: 0,
    latestInferenceMs: 0,
    latestTotalMs: 0,
    latestFps: 0,
    avgInferenceMs: 0,
  });

  // ==========================================
  // TAB 3: Benchmark State
  // ==========================================
  const [benchmarkRunning, setBenchmarkRunning] = useState(false);
  const [benchmarkResults, setBenchmarkResults] = useState(null);

  // Refs
  const logsEndRef = useRef(null);
  const compareLogsEndRef = useRef(null);
  const fileInputRefA = useRef(null);
  const fileInputRefB = useRef(null);
  const fileInputRefSingle = useRef(null);

  // Fetch system status on load
  useEffect(() => {
    fetchStatus();
  }, []);

  // Auto-scroll logs
  useEffect(() => {
    if (logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs]);

  useEffect(() => {
    if (compareLogsEndRef.current) {
      compareLogsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [compareLogs]);

  const fetchStatus = async () => {
    setLoadingStatus(true);
    try {
      const res = await fetch('/api/status');
      const data = await res.json();
      setStatus(data);
      if (!data.hardware.gpu_available) {
        setDevice('cpu');
        setCompareDevice('cpu');
      }
    } catch (err) {
      console.error('Error fetching status:', err);
    } finally {
      setLoadingStatus(false);
    }
  };

  // Upload handler for Video A in 2-Video Matching
  const handleUploadVideoA = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setUploadingA(true);
    const formData = new FormData();
    formData.append('file', file);
    try {
      const res = await fetch('/api/upload', { method: 'POST', body: formData });
      const data = await res.json();
      if (data.success) {
        setCompareVideoA(data.path);
        setUploadedNameA(data.filename);
        setVideoAMode('upload');
      }
    } catch (err) {
      alert('Upload failed: ' + err.message);
    } finally {
      setUploadingA(false);
    }
  };

  // Upload handler for Video B in 2-Video Matching
  const handleUploadVideoB = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setUploadingB(true);
    const formData = new FormData();
    formData.append('file', file);
    try {
      const res = await fetch('/api/upload', { method: 'POST', body: formData });
      const data = await res.json();
      if (data.success) {
        setCompareVideoB(data.path);
        setUploadedNameB(data.filename);
        setVideoBMode('upload');
      }
    } catch (err) {
      alert('Upload failed: ' + err.message);
    } finally {
      setUploadingB(false);
    }
  };

  // Upload handler for Single-Video extraction
  const handleFileUploadSingle = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setUploadedFile(file);
    const formData = new FormData();
    formData.append('file', file);
    try {
      const res = await fetch('/api/upload', { method: 'POST', body: formData });
      const data = await res.json();
      if (data.success) {
        setUploadedFilePath(data.path);
        if (data.is_video) {
          setInputMode('upload_video');
        } else {
          setInputMode('upload_image');
        }
      }
    } catch (err) {
      alert('Upload failed: ' + err.message);
    }
  };

  // Add compare log helper
  const addCompareLog = (text) => {
    setCompareLogs((prev) => [...prev, { text, time: new Date().toLocaleTimeString() }]);
  };

  // ==========================================
  // Action: Start 2-Video Comparison Pipeline
  // ==========================================
  const startTwoVideoCompare = () => {
    if (compareRunning) return;

    if (!compareVideoA || !compareVideoB) {
      alert('Please specify both Video A and Video B paths.');
      return;
    }

    setCompareRunning(true);
    setCompareResult(null);
    setCompareLogs([]);
    setCompareStep(1);

    addCompareLog(`[INIT] Initializing Two-Video Comparison Pipeline...`);
    addCompareLog(`[CONFIG] Video A: ${compareVideoA}`);
    addCompareLog(`[CONFIG] Video B: ${compareVideoB}`);
    addCompareLog(
      `[CONFIG] Hardware: ${compareDevice.toUpperCase()} ${
        compareDevice === 'cpu'
          ? `(${compareCpuCores === 12 ? '12 Threads' : `${compareCpuCores} Cores`})`
          : ''
      }`
    );
    addCompareLog(
      `[CONFIG] Multi-Core Turbo: ${
        compareFastMode ? 'ENABLED (640px + Early Exit ~18x speedup)' : 'DISABLED (Full 1024px)'
      }`
    );
    addCompareLog(`[CONFIG] Super-Resolution: ${compareUpscale.toUpperCase()} (2x) + CLAHE Grayscale`);
    addCompareLog(`[CONFIG] Decision Threshold: ${compareThreshold} matches (statusprint.py)`);

    const params = new URLSearchParams({
      video_a: compareVideoA,
      video_b: compareVideoB,
      device: compareDevice,
      upscale: compareUpscale,
      scale: String(compareScale),
      threshold: String(compareThreshold),
      interval: String(compareInterval),
      max_frames: String(compareMaxFrames),
      fast_mode: String(compareFastMode),
    });
    if (compareDevice === 'cpu' && compareCpuCores) {
      params.append('cpu_cores', String(compareCpuCores));
    }

    const sseUrl = `/api/stream-compare-videos?${params.toString()}`;
    const es = new EventSource(sseUrl);
    setEventSource(es);

    es.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.event === 'log') {
          const text = msg.line || '';
          setCompareLogs((prev) => [...prev, { text, time: new Date().toLocaleTimeString() }]);

          if (text.includes('[Step 1/4]')) setCompareStep(1);
          else if (text.includes('[Step 2/4]')) setCompareStep(2);
          else if (text.includes('[Step 3/4]')) setCompareStep(3);
          else if (text.includes('[Step 4/4]')) setCompareStep(4);
        } else if (msg.event === 'two_video_result') {
          setCompareResult(msg.data);
          setCompareStep(5);
        } else if (msg.event === 'done') {
          addCompareLog('[COMPLETE] Comparison pipeline execution completed successfully.');
          es.close();
          setEventSource(null);
          setCompareRunning(false);
        } else if (msg.event === 'error') {
          addCompareLog(`[ERROR] ${msg.error}`);
          es.close();
          setEventSource(null);
          setCompareRunning(false);
        }
      } catch (e) {
        console.error('Error parsing SSE event:', e);
      }
    };

    es.onerror = (err) => {
      console.error('SSE Error in comparison:', err);
      es.close();
      setEventSource(null);
      setCompareRunning(false);
      addCompareLog('[ERROR] Connection closed or server error.');
    };
  };

  const stopTwoVideoCompare = () => {
    if (eventSource) {
      eventSource.close();
      setEventSource(null);
    }
    setCompareRunning(false);
    addCompareLog('[USER] Comparison halted by user.');
  };

  // ==========================================
  // Action: Single-Video Frame Extraction
  // ==========================================
  const startDetection = () => {
    if (isProcessing) return;

    setIsProcessing(true);
    setLogs([]);
    setAllCrops([]);
    setLatestFrame(null);
    setLatestCrop(null);
    setMetrics({
      processedFrames: 0,
      totalMuzzles: 0,
      latestInferenceMs: 0,
      latestTotalMs: 0,
      latestFps: 0,
      avgInferenceMs: 0,
    });

    let sseUrl = '';

    if (inputMode === 'sample_video' || inputMode === 'upload_video') {
      const vidPath = inputMode === 'sample_video' ? selectedSampleVideo : uploadedFilePath;
      const params = new URLSearchParams({
        video_path: vidPath,
        device: device,
        frame_interval: frameInterval,
        max_frames: maxFrames,
        confidence: confidence,
        square: square ? 'true' : 'false',
        resize: resize,
      });
      if (device === 'cpu' && cpuCores) {
        params.append('cpu_cores', cpuCores);
      }
      sseUrl = `/api/stream-video?${params.toString()}`;
    } else {
      const imgPath = inputMode === 'sample_image' ? selectedSampleImage : uploadedFilePath;
      const params = new URLSearchParams({
        image_path: imgPath,
        device: device,
        confidence: confidence,
        square: square ? 'true' : 'false',
        resize: resize,
      });
      if (device === 'cpu' && cpuCores) {
        params.append('cpu_cores', cpuCores);
      }
      sseUrl = `/api/stream-image?${params.toString()}`;
    }

    const es = new EventSource(sseUrl);
    setEventSource(es);

    es.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.event === 'log') {
          setLogs((prev) => [...prev, { text: msg.line, time: new Date().toLocaleTimeString() }]);
        } else if (msg.event === 'detection') {
          const payload = msg.data;
          if (payload.annotated_url) {
            setLatestFrame(payload.annotated_url);
          }
          if (payload.crops && payload.crops.length > 0) {
            const firstCrop = payload.crops[0];
            setLatestCrop(firstCrop);
            setAllCrops((prev) => [...prev, ...payload.crops]);
          }

          setMetrics((prev) => {
            const t = payload.timings || {};
            const infMs = t.t_inference_ms || 0;
            const newProcessed = prev.processedFrames + 1;
            const newMuzzles = prev.totalMuzzles + (payload.muzzle_count || (payload.crops ? payload.crops.length : 0));
            return {
              processedFrames: newProcessed,
              totalMuzzles: newMuzzles,
              latestInferenceMs: infMs,
              latestTotalMs: t.t_total_ms || 0,
              latestFps: t.fps || 0,
              avgInferenceMs: prev.avgInferenceMs === 0 ? infMs : (prev.avgInferenceMs + infMs) / 2,
            };
          });
        } else if (msg.event === 'done') {
          setLogs((prev) => [...prev, { text: '[INFO] Pipeline finished.', time: new Date().toLocaleTimeString() }]);
          es.close();
          setEventSource(null);
          setIsProcessing(false);
        }
      } catch (err) {
        console.error('SSE Parse Error:', err);
      }
    };

    es.onerror = (err) => {
      console.error('SSE Error:', err);
      es.close();
      setEventSource(null);
      setIsProcessing(false);
    };
  };

  const stopProcessing = () => {
    if (eventSource) {
      eventSource.close();
      setEventSource(null);
    }
    setIsProcessing(false);
    setBenchmarkRunning(false);
  };

  // ==========================================
  // Action: Comparative Benchmark
  // ==========================================
  const runBenchmark = () => {
    if (benchmarkRunning) return;

    setBenchmarkRunning(true);
    setActiveTab('benchmark');
    setLogs([]);
    setBenchmarkResults(null);

    const es = new EventSource('/api/stream-benchmark?max_images=2');
    setEventSource(es);

    es.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.event === 'log') {
          setLogs((prev) => [...prev, { text: msg.line, time: new Date().toLocaleTimeString() }]);
        } else if (msg.event === 'detection' && msg.data.type === 'benchmark') {
          setBenchmarkResults(msg.data.results);
        } else if (msg.event === 'done') {
          es.close();
          setEventSource(null);
          setBenchmarkRunning(false);
        }
      } catch (err) {
        console.error('Benchmark SSE error:', err);
      }
    };

    es.onerror = () => {
      es.close();
      setEventSource(null);
      setBenchmarkRunning(false);
    };
  };

  // Available sample video options
  const sampleVideos = status?.sample_videos || [
    { name: 'muzzleVideo.mp4', path: '/home/adis_mayur/F_CODES/muzzleVideo.mp4', size_mb: 24.5 },
    { name: 'muzzleVideo2.mp4', path: '/home/adis_mayur/F_CODES/muzzleVideo2.mp4', size_mb: 28.1 },
    { name: 'muzzleVideo3.mp4', path: '/home/adis_mayur/F_CODES/muzzleVideo3.mp4', size_mb: 19.4 },
    { name: 'muzzleVideo5.mp4', path: '/home/adis_mayur/F_CODES/muzzleVideo5.mp4', size_mb: 31.0 },
    { name: 'muzzleVideo6.mp4', path: '/home/adis_mayur/F_CODES/muzzleVideo6.mp4', size_mb: 22.8 },
    { name: 'muzzleVideo7.mp4', path: '/home/adis_mayur/F_CODES/muzzleVideo7.mp4', size_mb: 26.3 },
  ];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col selection:bg-emerald-500 selection:text-white">
      {/* Top Header */}
      <header className="border-b border-slate-800 bg-slate-900/70 backdrop-blur-md sticky top-0 z-50 px-6 py-4">
        <div className="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-emerald-500 to-teal-400 flex items-center justify-center text-slate-950 font-bold shadow-lg shadow-emerald-500/20 text-xl">
              <ScanLine className="w-6 h-6 text-slate-950" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-xl font-bold tracking-tight text-white">
                  Cattle Muzzle Groove Extraction &amp; Ridge Skeletonization
                </h1>
                <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 font-mono">
                  nose_lines Pipeline
                </span>
              </div>
              <p className="text-xs text-slate-400">
                Multi-Scale Morphological Black-Hat &amp; Frangi Filtering • Binary Groove Segmentation • 1-px Topological Ridge Skeletonization
              </p>
            </div>
          </div>

          {/* System Hardware Status Badges */}
          <div className="flex items-center gap-3">
            {status?.hardware?.gpu_available ? (
              <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-emerald-950/60 border border-emerald-500/30 text-emerald-300 text-xs font-medium">
                <Zap className="w-3.5 h-3.5 text-emerald-400" />
                <span>GPU: {status.hardware.gpu_name}</span>
              </div>
            ) : (
              <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-amber-950/60 border border-amber-500/30 text-amber-300 text-xs font-medium">
                <AlertCircle className="w-3.5 h-3.5 text-amber-400" />
                <span>GPU Not Available</span>
              </div>
            )}
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-800/80 border border-slate-700 text-slate-300 text-xs font-medium">
              <Cpu className="w-3.5 h-3.5 text-sky-400" />
              <span>CPU: {status?.hardware?.cpu_threads || 12} Threads</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Container */}
      <main className="max-w-7xl mx-auto w-full p-6 flex-1 flex flex-col gap-6">
        {/* Navigation Tabs */}
        <div className="flex flex-wrap items-center justify-between border-b border-slate-800 pb-3 gap-3">
          <div className="flex flex-wrap items-center gap-2">
            {/* Primary Tab: Groove Extraction Studio (nose_lines) */}
            <button
              onClick={() => setActiveTab('groove_studio')}
              className={`flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-bold transition-all ${
                activeTab === 'groove_studio'
                  ? 'bg-gradient-to-r from-emerald-500 to-teal-500 text-slate-950 shadow-lg shadow-emerald-500/25 ring-1 ring-emerald-400'
                  : 'text-slate-300 hover:text-white hover:bg-slate-900 border border-transparent'
              }`}
            >
              <ScanLine className="w-4 h-4" />
              <span>Muzzle Groove &amp; Skeleton Studio</span>
              <span className={`text-[10px] font-extrabold uppercase px-1.5 py-0.5 rounded ${
                activeTab === 'groove_studio' ? 'bg-slate-950/30 text-slate-950' : 'bg-emerald-500/20 text-emerald-300'
              }`}>
                nose_lines
              </span>
            </button>

            {/* Tab 2: 2-Video Comparison */}
            <button
              onClick={() => setActiveTab('two_video')}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold transition-all ${
                activeTab === 'two_video'
                  ? 'bg-emerald-500 text-slate-950 shadow-md shadow-emerald-500/20'
                  : 'text-slate-400 hover:text-white hover:bg-slate-900'
              }`}
            >
              <GitCompare className="w-4 h-4" />
              <span>2-Video Biometric Matcher</span>
            </button>

            {/* Tab 3: Single-Video Frame Extraction */}
            <button
              onClick={() => setActiveTab('viewer')}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold transition-all ${
                activeTab === 'viewer'
                  ? 'bg-emerald-500 text-slate-950 shadow-md shadow-emerald-500/20'
                  : 'text-slate-400 hover:text-white hover:bg-slate-900'
              }`}
            >
              <Activity className="w-4 h-4" />
              Single-Video Frame Cropper
            </button>

            {/* Tab 4: Hardware Benchmark */}
            <button
              onClick={() => setActiveTab('benchmark')}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold transition-all ${
                activeTab === 'benchmark'
                  ? 'bg-emerald-500 text-slate-950 shadow-md shadow-emerald-500/20'
                  : 'text-slate-400 hover:text-white hover:bg-slate-900'
              }`}
            >
              <BarChart3 className="w-4 h-4" />
              Hardware Benchmark
            </button>
          </div>

          <div className="flex items-center gap-2 text-xs text-slate-400">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
            Weights: <code className="text-slate-300">crop11dec.h5 ({status?.weights_size_mb || 245} MB)</code>
          </div>
        </div>

        {/* ========================================================= */}
        {/* PRIMARY TAB: GROOVE EXTRACTION STUDIO (NOSE_LINES)       */}
        {/* ========================================================= */}
        {activeTab === 'groove_studio' && (
          <GrooveStudioTab sampleVideos={status?.sample_videos || []} />
        )}

        {/* ========================================================= */}
        {/* TAB 2: 2-VIDEO MUZZLE MATCHING & VERIFICATION             */}
        {/* ========================================================= */}
        {activeTab === 'two_video' && (
          <div className="flex flex-col gap-6">
            {/* Quick-Test Presets Bar */}
            <div className="glass-panel p-3.5 rounded-xl flex flex-wrap items-center justify-between gap-3 text-xs">
              <div className="flex items-center gap-2 text-slate-300 font-semibold">
                <Sparkles className="w-4 h-4 text-emerald-400" />
                <span>Quick Verification Presets:</span>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <button
                  onClick={() => {
                    setCompareVideoA('/home/adis_mayur/F_CODES/muzzleVideo.mp4');
                    setCompareVideoB('/home/adis_mayur/F_CODES/muzzleVideo.mp4');
                    setVideoAMode('sample');
                    setVideoBMode('sample');
                  }}
                  className="px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-emerald-950 hover:text-emerald-300 border border-slate-800 hover:border-emerald-500/40 text-slate-300 transition-all font-medium"
                >
                  🟢 Same Cattle Self-Test (Video 1 vs Video 1)
                </button>
                <button
                  onClick={() => {
                    setCompareVideoA('/home/adis_mayur/F_CODES/muzzleVideo.mp4');
                    setCompareVideoB('/home/adis_mayur/F_CODES/muzzleVideo2.mp4');
                    setVideoAMode('sample');
                    setVideoBMode('sample');
                  }}
                  className="px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-300 transition-all font-medium"
                >
                  🔍 Cross-Video Test (Video 1 vs Video 2)
                </button>
                <button
                  onClick={() => {
                    setCompareVideoA('/home/adis_mayur/F_CODES/muzzleVideo2.mp4');
                    setCompareVideoB('/home/adis_mayur/F_CODES/muzzleVideo3.mp4');
                    setVideoAMode('sample');
                    setVideoBMode('sample');
                  }}
                  className="px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-300 transition-all font-medium"
                >
                  🔍 Cross-Video Test (Video 2 vs Video 3)
                </button>
              </div>
            </div>

            {/* Video A and Video B Input Cards Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 relative">
              {/* Card for Video A */}
              <div className="glass-panel p-5 rounded-2xl flex flex-col gap-4 border-l-4 border-l-emerald-500">
                <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                  <div className="flex items-center gap-2">
                    <div className="w-7 h-7 rounded-lg bg-emerald-500/20 text-emerald-400 flex items-center justify-center font-bold text-xs">
                      A
                    </div>
                    <div>
                      <h3 className="text-sm font-bold text-white">Cattle Video A (Reference Profile)</h3>
                      <p className="text-[11px] text-slate-400">Baseline cattle video for biometric registration</p>
                    </div>
                  </div>
                  <span className="text-[11px] px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-300 border border-emerald-500/20 font-mono">
                    Reference
                  </span>
                </div>

                {/* Video A Mode Selector */}
                <div className="grid grid-cols-2 gap-1 bg-slate-900 p-1 rounded-xl border border-slate-800 text-xs">
                  <button
                    onClick={() => setVideoAMode('sample')}
                    className={`py-1.5 px-3 rounded-lg font-medium transition-all ${
                      videoAMode === 'sample' ? 'bg-slate-800 text-white font-bold' : 'text-slate-400 hover:text-white'
                    }`}
                  >
                    Sample Video Library
                  </button>
                  <button
                    onClick={() => setVideoAMode('upload')}
                    className={`py-1.5 px-3 rounded-lg font-medium transition-all ${
                      videoAMode === 'upload' ? 'bg-slate-800 text-white font-bold' : 'text-slate-400 hover:text-white'
                    }`}
                  >
                    Upload Custom Video
                  </button>
                </div>

                {/* Sample Video Dropdown A */}
                {videoAMode === 'sample' && (
                  <div className="flex flex-col gap-2">
                    <label className="text-xs text-slate-400 font-medium">Choose Sample Video A:</label>
                    <select
                      value={compareVideoA}
                      onChange={(e) => setCompareVideoA(e.target.value)}
                      className="bg-slate-900 border border-slate-800 rounded-xl px-3 py-2.5 text-xs text-white focus:outline-none focus:border-emerald-500 font-mono"
                    >
                      {sampleVideos.map((v, i) => (
                        <option key={i} value={v.path}>
                          {v.name} ({v.size_mb} MB)
                        </option>
                      ))}
                    </select>
                    <div className="text-[11px] text-slate-500 font-mono truncate">
                      Path: {compareVideoA}
                    </div>
                  </div>
                )}

                {/* Upload Video A */}
                {videoAMode === 'upload' && (
                  <div>
                    <input
                      ref={fileInputRefA}
                      type="file"
                      accept="video/*"
                      onChange={handleUploadVideoA}
                      className="hidden"
                    />
                    <button
                      onClick={() => fileInputRefA.current?.click()}
                      disabled={uploadingA}
                      className="w-full border-2 border-dashed border-slate-700 hover:border-emerald-500/60 p-4 rounded-xl flex flex-col items-center justify-center gap-2 text-slate-400 hover:text-white transition-all bg-slate-900/30"
                    >
                      <Upload className="w-5 h-5 text-emerald-400" />
                      <span className="text-xs font-semibold">
                        {uploadingA ? 'Uploading Video A...' : uploadedNameA ? `Uploaded: ${uploadedNameA}` : 'Click to upload Cattle Video A (MP4/AVI)'}
                      </span>
                    </button>
                    {uploadedNameA && (
                      <div className="mt-2 text-[11px] text-emerald-400 flex items-center gap-1 font-mono">
                        <Check className="w-3.5 h-3.5" /> Video A Ready: {uploadedNameA}
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* Card for Video B */}
              <div className="glass-panel p-5 rounded-2xl flex flex-col gap-4 border-l-4 border-l-teal-500">
                <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                  <div className="flex items-center gap-2">
                    <div className="w-7 h-7 rounded-lg bg-teal-500/20 text-teal-400 flex items-center justify-center font-bold text-xs">
                      B
                    </div>
                    <div>
                      <h3 className="text-sm font-bold text-white">Cattle Video B (Target Query)</h3>
                      <p className="text-[11px] text-slate-400">Query cattle video to verify for duplicate match</p>
                    </div>
                  </div>
                  <span className="text-[11px] px-2 py-0.5 rounded bg-teal-500/10 text-teal-300 border border-teal-500/20 font-mono">
                    Query Target
                  </span>
                </div>

                {/* Video B Mode Selector */}
                <div className="grid grid-cols-2 gap-1 bg-slate-900 p-1 rounded-xl border border-slate-800 text-xs">
                  <button
                    onClick={() => setVideoBMode('sample')}
                    className={`py-1.5 px-3 rounded-lg font-medium transition-all ${
                      videoBMode === 'sample' ? 'bg-slate-800 text-white font-bold' : 'text-slate-400 hover:text-white'
                    }`}
                  >
                    Sample Video Library
                  </button>
                  <button
                    onClick={() => setVideoBMode('upload')}
                    className={`py-1.5 px-3 rounded-lg font-medium transition-all ${
                      videoBMode === 'upload' ? 'bg-slate-800 text-white font-bold' : 'text-slate-400 hover:text-white'
                    }`}
                  >
                    Upload Custom Video
                  </button>
                </div>

                {/* Sample Video Dropdown B */}
                {videoBMode === 'sample' && (
                  <div className="flex flex-col gap-2">
                    <label className="text-xs text-slate-400 font-medium">Choose Sample Video B:</label>
                    <select
                      value={compareVideoB}
                      onChange={(e) => setCompareVideoB(e.target.value)}
                      className="bg-slate-900 border border-slate-800 rounded-xl px-3 py-2.5 text-xs text-white focus:outline-none focus:border-teal-500 font-mono"
                    >
                      {sampleVideos.map((v, i) => (
                        <option key={i} value={v.path}>
                          {v.name} ({v.size_mb} MB)
                        </option>
                      ))}
                    </select>
                    <div className="text-[11px] text-slate-500 font-mono truncate">
                      Path: {compareVideoB}
                    </div>
                  </div>
                )}

                {/* Upload Video B */}
                {videoBMode === 'upload' && (
                  <div>
                    <input
                      ref={fileInputRefB}
                      type="file"
                      accept="video/*"
                      onChange={handleUploadVideoB}
                      className="hidden"
                    />
                    <button
                      onClick={() => fileInputRefB.current?.click()}
                      disabled={uploadingB}
                      className="w-full border-2 border-dashed border-slate-700 hover:border-teal-500/60 p-4 rounded-xl flex flex-col items-center justify-center gap-2 text-slate-400 hover:text-white transition-all bg-slate-900/30"
                    >
                      <Upload className="w-5 h-5 text-teal-400" />
                      <span className="text-xs font-semibold">
                        {uploadingB ? 'Uploading Video B...' : uploadedNameB ? `Uploaded: ${uploadedNameB}` : 'Click to upload Cattle Video B (MP4/AVI)'}
                      </span>
                    </button>
                    {uploadedNameB && (
                      <div className="mt-2 text-[11px] text-teal-400 flex items-center gap-1 font-mono">
                        <Check className="w-3.5 h-3.5" /> Video B Ready: {uploadedNameB}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>

            {/* Pipeline Configuration Card */}
            <div className="glass-panel p-5 rounded-2xl flex flex-col gap-5">
              <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                <div className="flex items-center gap-2">
                  <Sliders className="w-4 h-4 text-emerald-400" />
                  <h3 className="text-sm font-bold text-white">Biometric Pipeline & Algorithm Settings</h3>
                </div>
                <span className="text-[11px] text-slate-400">
                  statusprint.py threshold & Super-Resolution parameters
                </span>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
                {/* 1. Hardware Engine Selection */}
                <div className="flex flex-col gap-2.5">
                  <label className="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
                    <Cpu className="w-3.5 h-3.5 text-emerald-400" />
                    Hardware Engine
                  </label>
                  <div className="flex flex-col gap-2">
                    {/* Radio Option 1: NVIDIA RTX 3050 GPU */}
                    <label
                      className={`flex items-center justify-between p-2.5 rounded-xl border cursor-pointer text-xs transition-all ${
                        compareDevice === 'gpu'
                          ? 'bg-emerald-950/40 border-emerald-500/60 text-white font-semibold ring-1 ring-emerald-500/30'
                          : 'bg-slate-900/60 border-slate-800 text-slate-400 hover:border-slate-700'
                      }`}
                    >
                      <div className="flex items-center gap-2">
                        <input
                          type="radio"
                          name="compareDevice"
                          checked={compareDevice === 'gpu'}
                          onChange={() => setCompareDevice('gpu')}
                          className="text-emerald-500 focus:ring-emerald-500"
                        />
                        <span className="flex items-center gap-1.5">
                          <Zap className="w-3.5 h-3.5 text-emerald-400" />
                          NVIDIA RTX 3050 GPU
                        </span>
                      </div>
                      <span className="text-[10px] bg-emerald-500/20 text-emerald-300 px-1.5 py-0.5 rounded font-mono font-medium">
                        ~325 ms
                      </span>
                    </label>

                    {/* Radio Option 2: AMD Ryzen CPU (Multi-Core Allocator) */}
                    <label
                      className={`flex items-center justify-between p-2.5 rounded-xl border cursor-pointer text-xs transition-all ${
                        compareDevice === 'cpu'
                          ? 'bg-sky-950/40 border-sky-500/60 text-white font-semibold ring-1 ring-sky-500/30'
                          : 'bg-slate-900/60 border-slate-800 text-slate-400 hover:border-slate-700'
                      }`}
                    >
                      <div className="flex items-center gap-2">
                        <input
                          type="radio"
                          name="compareDevice"
                          checked={compareDevice === 'cpu'}
                          onChange={() => setCompareDevice('cpu')}
                          className="text-sky-500 focus:ring-sky-500"
                        />
                        <span className="flex items-center gap-1.5">
                          <Cpu className="w-3.5 h-3.5 text-sky-400" />
                          AMD Ryzen CPU (Multi-Core Allocator)
                        </span>
                      </div>
                      <span className="text-[10px] bg-sky-500/20 text-sky-300 px-1.5 py-0.5 rounded font-mono font-medium">
                        {getCpuLatency(compareCpuCores)}
                      </span>
                    </label>

                    {/* Interactive CPU Core Selector (Shown when AMD Ryzen CPU is active) */}
                    {compareDevice === 'cpu' && (
                      <div className="p-3 bg-slate-900/90 border border-sky-500/30 rounded-xl flex flex-col gap-2.5 shadow-inner">
                        {/* Core Count & Dynamic Latency Header */}
                        <div className="flex items-center justify-between text-xs">
                          <div className="flex items-center gap-1.5 text-slate-300 font-semibold">
                            <Sliders className="w-3.5 h-3.5 text-sky-400" />
                            <span>Cores:</span>
                            <span className="font-mono text-sky-400 font-bold">
                              {compareCpuCores} {compareCpuCores === 1 ? 'Core' : compareCpuCores === 12 ? 'Threads (Max)' : 'Cores'}
                            </span>
                          </div>
                          <div className="flex items-center gap-1.5">
                            <span className="text-[10px] text-slate-400">Latency:</span>
                            <span className="text-[10px] font-mono font-bold bg-sky-500/20 text-sky-300 px-2 py-0.5 rounded border border-sky-500/30">
                              {getCpuLatency(compareCpuCores)}
                            </span>
                          </div>
                        </div>

                        {/* Slider + Number Input */}
                        <div className="flex items-center gap-3">
                          <input
                            type="range"
                            min="1"
                            max="12"
                            step="1"
                            value={compareCpuCores}
                            onChange={(e) => setCompareCpuCores(parseInt(e.target.value, 10))}
                            className="w-full h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-sky-400"
                          />
                          <input
                            type="number"
                            min="1"
                            max="12"
                            value={compareCpuCores}
                            onChange={(e) => {
                              const val = Math.max(1, Math.min(12, parseInt(e.target.value, 10) || 1));
                              setCompareCpuCores(val);
                            }}
                            className="w-12 bg-slate-800 border border-slate-700 text-center text-xs font-mono font-bold text-white py-1 rounded-lg focus:outline-none focus:border-sky-400"
                          />
                        </div>

                        {/* Quick Presets Pills */}
                        <div className="flex flex-wrap gap-1 pt-1.5 border-t border-slate-800/80">
                          <span className="text-[10px] text-slate-400 self-center mr-1">Presets:</span>
                          {CPU_PRESETS.map((preset) => (
                            <button
                              key={preset.cores}
                              type="button"
                              onClick={() => setCompareCpuCores(preset.cores)}
                              className={`px-2 py-0.5 rounded text-[10px] font-medium transition-all ${
                                compareCpuCores === preset.cores
                                  ? 'bg-sky-500 text-slate-950 font-bold shadow-sm shadow-sky-500/30'
                                  : 'bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700/60'
                              }`}
                            >
                              {preset.label}
                            </button>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>

                  {/* ⚡ Multi-Core Turbo Mode Toggle */}
                  <div className="p-2.5 bg-amber-950/25 border border-amber-500/30 rounded-xl flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Zap className={`w-4 h-4 ${compareFastMode ? 'text-amber-400 animate-pulse' : 'text-slate-500'}`} />
                      <div>
                        <div className="text-[11px] font-bold text-slate-200 flex items-center gap-1.5">
                          ⚡ Multi-Core Turbo Mode
                          <span className="text-[9px] bg-amber-500/20 text-amber-300 px-1 rounded font-mono font-normal">18x Speedup</span>
                        </div>
                        <div className="text-[10px] text-slate-400">
                          640px input + 100 candidate ROIs + blur pre-filter + early exit
                        </div>
                      </div>
                    </div>
                    <button
                      type="button"
                      onClick={() => setCompareFastMode(!compareFastMode)}
                      className={`px-2.5 py-1 rounded-lg text-[10px] font-bold tracking-wider transition-all ${
                        compareFastMode
                          ? 'bg-amber-500 text-slate-950 shadow-md shadow-amber-500/20 hover:bg-amber-400'
                          : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
                      }`}
                    >
                      {compareFastMode ? 'ACTIVE' : 'OFF'}
                    </button>
                  </div>
                </div>

                {/* 2. Super-Resolution & Enhancement Algorithm */}
                <div className="flex flex-col gap-2.5">
                  <label className="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
                    <Sparkles className="w-3.5 h-3.5 text-teal-400" />
                    Super-Resolution & Enhancement
                  </label>
                  <div className="flex flex-col gap-2">
                    <select
                      value={compareUpscale}
                      onChange={(e) => setCompareUpscale(e.target.value)}
                      className="bg-slate-900 border border-slate-800 rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-teal-500 font-medium"
                    >
                      <option value="lanczos4">
                        Lanczos-4 Sinc Interpolation (Recommended - Preserves Muzzle Beads)
                      </option>
                      <option value="bicubic">Bicubic Interpolation</option>
                      <option value="bilinear">Bilinear Interpolation</option>
                    </select>

                    <div className="p-2.5 rounded-xl bg-slate-900/80 border border-slate-800/80 text-[11px] text-slate-300 flex flex-col gap-1">
                      <div className="flex items-center justify-between">
                        <span className="text-slate-400">Scale Factor:</span>
                        <span className="font-mono text-emerald-400 font-bold">2x (224x224 → 448x448)</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-slate-400">Ridge Enhancement:</span>
                        <span className="text-teal-300 font-semibold">CLAHE Grayscale + Unsharp</span>
                      </div>
                      <div className="text-[10px] text-slate-400 italic pt-1 border-t border-slate-800/60">
                        Lanczos 8-point sinc preserves fine dermatoglyphic ridge groove micro-features.
                      </div>
                    </div>
                  </div>
                </div>

                {/* 3. Decision Threshold & Sampling Controls */}
                <div className="flex flex-col gap-2.5">
                  <div className="flex items-center justify-between">
                    <label className="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
                      <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
                      Decision Threshold (statusprint.py)
                    </label>
                    <span className="font-mono text-emerald-400 font-bold text-xs">
                      {compareThreshold} matches
                    </span>
                  </div>

                  <div className="flex flex-col gap-2">
                    <input
                      type="range"
                      min="15"
                      max="75"
                      step="1"
                      value={compareThreshold}
                      onChange={(e) => setCompareThreshold(Number(e.target.value))}
                      className="w-full accent-emerald-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
                    />

                    <div className="flex items-center justify-between text-[11px] text-slate-400">
                      <span>Lenient (20)</span>
                      <span className="text-emerald-400 font-mono font-bold">Default: 35</span>
                      <span>Strict (60)</span>
                    </div>

                    <div className="grid grid-cols-2 gap-2 pt-1 border-t border-slate-800/80">
                      <div>
                        <label className="text-[11px] text-slate-400 block mb-0.5">Frame Interval</label>
                        <select
                          value={compareInterval}
                          onChange={(e) => setCompareInterval(Number(e.target.value))}
                          className="w-full bg-slate-900 border border-slate-800 rounded-lg px-2 py-1 text-xs text-white font-mono"
                        >
                          <option value={5}>Every 5th frame</option>
                          <option value={10}>Every 10th frame</option>
                          <option value={15}>Every 15th frame</option>
                          <option value={20}>Every 20th frame</option>
                        </select>
                      </div>
                      <div>
                        <label className="text-[11px] text-slate-400 block mb-0.5">Max Candidate Frames</label>
                        <select
                          value={compareMaxFrames}
                          onChange={(e) => setCompareMaxFrames(Number(e.target.value))}
                          className="w-full bg-slate-900 border border-slate-800 rounded-lg px-2 py-1 text-xs text-white font-mono"
                        >
                          <option value={4}>4 Frames (Fast)</option>
                          <option value={6}>6 Frames (Balanced)</option>
                          <option value={8}>8 Frames (Deep)</option>
                          <option value={12}>12 Frames (Exhaustive)</option>
                        </select>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Action Button: Compare 2 Cattle Videos */}
              <div className="pt-2">
                {!compareRunning ? (
                  <button
                    onClick={startTwoVideoCompare}
                    className="w-full py-4 px-6 rounded-xl bg-gradient-to-r from-emerald-500 via-teal-500 to-emerald-400 hover:from-emerald-400 hover:to-teal-300 text-slate-950 font-extrabold flex items-center justify-center gap-3 text-base shadow-xl shadow-emerald-500/25 transition-all cursor-pointer"
                  >
                    <GitCompare className="w-5 h-5 fill-current" />
                    <span>Compare 2 Cattle Videos (Extract, Upscale & ORB Match)</span>
                  </button>
                ) : (
                  <button
                    onClick={stopTwoVideoCompare}
                    className="w-full py-4 px-6 rounded-xl bg-rose-600 hover:bg-rose-500 text-white font-extrabold flex items-center justify-center gap-3 text-base shadow-xl shadow-rose-600/25 transition-all cursor-pointer"
                  >
                    <Square className="w-5 h-5 fill-current" />
                    <span>Stop Pipeline Execution</span>
                  </button>
                )}
              </div>
            </div>

            {/* Pipeline Step Progress Bar (when running or completed) */}
            {(compareRunning || compareStep > 0) && (
              <div className="glass-panel p-4 rounded-xl flex flex-col gap-2">
                <div className="flex items-center justify-between text-xs font-semibold text-slate-300">
                  <span>Verification Execution Progress:</span>
                  <span className="font-mono text-emerald-400">
                    {compareStep === 1 && 'Step 1/4: Video A Extraction...'}
                    {compareStep === 2 && 'Step 2/4: Video B Extraction...'}
                    {compareStep === 3 && 'Step 3/4: Lanczos-4 Super-Resolution...'}
                    {compareStep === 4 && 'Step 4/4: ORB Hamming Matching...'}
                    {compareStep === 5 && 'Completed'}
                  </span>
                </div>
                <div className="grid grid-cols-4 gap-2 text-xs">
                  <div className={`p-2 rounded-lg border text-center font-medium ${
                    compareStep >= 1 ? 'bg-emerald-950/60 border-emerald-500/60 text-emerald-300' : 'bg-slate-900 border-slate-800 text-slate-500'
                  }`}>
                    1. Video A Muzzle
                  </div>
                  <div className={`p-2 rounded-lg border text-center font-medium ${
                    compareStep >= 2 ? 'bg-emerald-950/60 border-emerald-500/60 text-emerald-300' : 'bg-slate-900 border-slate-800 text-slate-500'
                  }`}>
                    2. Video B Muzzle
                  </div>
                  <div className={`p-2 rounded-lg border text-center font-medium ${
                    compareStep >= 3 ? 'bg-emerald-950/60 border-emerald-500/60 text-emerald-300' : 'bg-slate-900 border-slate-800 text-slate-500'
                  }`}>
                    3. 2x Super-Resolution
                  </div>
                  <div className={`p-2 rounded-lg border text-center font-medium ${
                    compareStep >= 4 ? 'bg-emerald-950/60 border-emerald-500/60 text-emerald-300' : 'bg-slate-900 border-slate-800 text-slate-500'
                  }`}>
                    4. ORB Verification
                  </div>
                </div>
              </div>
            )}

            {/* ========================================================= */}
            {/* STEP 3: MATCH VERDICT BANNER (PROMINENT RESULT DISPLAY)   */}
            {/* ========================================================= */}
            {compareResult && (
              <div
                className={`p-6 rounded-2xl border-2 transition-all shadow-2xl ${
                  compareResult.match.is_match
                    ? 'bg-emerald-950/80 border-emerald-500 shadow-emerald-500/20'
                    : 'bg-rose-950/80 border-rose-500 shadow-rose-500/20'
                }`}
              >
                <div className="flex flex-col lg:flex-row items-center justify-between gap-6">
                  {/* Left: Verdict Title & Badge */}
                  <div className="flex items-center gap-4 text-center lg:text-left">
                    <div
                      className={`w-16 h-16 rounded-2xl flex items-center justify-center shrink-0 ${
                        compareResult.match.is_match
                          ? 'bg-emerald-500/20 text-emerald-400 ring-2 ring-emerald-500/40'
                          : 'bg-rose-500/20 text-rose-400 ring-2 ring-rose-500/40'
                      }`}
                    >
                      {compareResult.match.is_match ? (
                        <CheckCircle2 className="w-10 h-10 animate-bounce" />
                      ) : (
                        <XCircle className="w-10 h-10" />
                      )}
                    </div>
                    <div>
                      <div className="flex flex-wrap items-center gap-2 justify-center lg:justify-start">
                        <span
                          className={`text-2xl font-black tracking-tight ${
                            compareResult.match.is_match ? 'text-emerald-300' : 'text-rose-300'
                          }`}
                        >
                          {compareResult.match.is_match
                            ? 'MATCH: SAME CATTLE'
                            : 'DIFFERENT CATTLE'}
                        </span>
                        <span
                          className={`text-xs font-bold uppercase px-2.5 py-0.5 rounded-full border ${
                            compareResult.match.is_match
                              ? 'bg-emerald-500/30 text-emerald-200 border-emerald-400/50'
                              : 'bg-rose-500/30 text-rose-200 border-rose-400/50'
                          }`}
                        >
                          {compareResult.match.is_match ? 'Duplicate Verified' : 'Distinct Animal'}
                        </span>
                      </div>
                      <p className="text-xs text-slate-300 mt-1 max-w-xl">
                        {compareResult.match.is_match
                          ? `Biometric nose ridge keypoint alignments confirm Video A and Video B belong to the exact same cattle (statusprint.py threshold: ${compareResult.match.threshold}).`
                          : `Keypoint descriptor distance analysis confirms Video A and Video B are distinct cattle with separate dermatoglyphic nose patterns.`}
                      </p>
                    </div>
                  </div>

                  {/* Right: Key Metrics Grid */}
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 w-full lg:w-auto">
                    {/* Good Matches */}
                    <div className="bg-slate-950/60 p-3.5 rounded-xl border border-slate-800 text-center min-w-[120px]">
                      <div className="text-[10px] uppercase font-bold text-slate-400">Good Matches</div>
                      <div
                        className={`text-2xl font-black font-mono mt-0.5 ${
                          compareResult.match.is_match ? 'text-emerald-400' : 'text-rose-400'
                        }`}
                      >
                        {compareResult.match.good_matches}
                      </div>
                      <div className="text-[10px] text-slate-400">
                        Threshold: <span className="font-bold text-white">{compareResult.match.threshold}</span>
                      </div>
                    </div>

                    {/* Similarity % */}
                    <div className="bg-slate-950/60 p-3.5 rounded-xl border border-slate-800 text-center min-w-[120px]">
                      <div className="text-[10px] uppercase font-bold text-slate-400">Similarity</div>
                      <div
                        className={`text-2xl font-black font-mono mt-0.5 ${
                          compareResult.match.is_match ? 'text-emerald-300' : 'text-amber-400'
                        }`}
                      >
                        {compareResult.match.similarity_percent}%
                      </div>
                      <div className="text-[10px] text-slate-400">Lowe Ratio &lt; 0.7</div>
                    </div>

                    {/* Keypoints Count */}
                    <div className="bg-slate-950/60 p-3.5 rounded-xl border border-slate-800 text-center min-w-[120px]">
                      <div className="text-[10px] uppercase font-bold text-slate-400">ORB Keypoints</div>
                      <div className="text-sm font-bold text-slate-200 font-mono mt-1">
                        A: {compareResult.match.kp1_count}
                      </div>
                      <div className="text-sm font-bold text-slate-200 font-mono">
                        B: {compareResult.match.kp2_count}
                      </div>
                    </div>

                    {/* Latency */}
                    <div className="bg-slate-950/60 p-3.5 rounded-xl border border-slate-800 text-center min-w-[120px]">
                      <div className="text-[10px] uppercase font-bold text-slate-400">Total Latency</div>
                      <div className="text-2xl font-black font-mono text-teal-400 mt-0.5">
                        {compareResult.duration_seconds}s
                      </div>
                      <div className="text-[10px] text-slate-400">End-to-End Pipeline</div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* ========================================================= */}
            {/* STEP 4: LARGE VISUAL KEYPOINT MATCH DIAGRAM               */}
            {/* ========================================================= */}
            {compareResult?.match?.visual_url && (
              <div className="glass-panel p-5 rounded-2xl flex flex-col gap-4 border border-slate-800">
                <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                  <div className="flex items-center gap-2">
                    <ScanLine className="w-5 h-5 text-emerald-400" />
                    <div>
                      <h3 className="text-sm font-bold text-white">
                        Step 4: Visual ORB Keypoint Match Diagram
                      </h3>
                      <p className="text-xs text-slate-400">
                        Connecting lines denote matching dermatoglyphic feature descriptors across Video A &amp; Video B
                      </p>
                    </div>
                  </div>
                  <a
                    href={compareResult.match.visual_url}
                    download="cattle_muzzle_match_diagram.jpg"
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold transition-all"
                  >
                    <Download className="w-4 h-4" />
                    Save Full Diagram
                  </a>
                </div>

                <div className="w-full bg-slate-900/90 rounded-xl overflow-hidden border border-slate-800 flex items-center justify-center p-2 relative">
                  <img
                    src={compareResult.match.visual_url}
                    alt="Visual Keypoint Match Diagram"
                    className="max-h-[500px] w-auto max-w-full object-contain rounded-lg shadow-2xl"
                  />
                </div>

                <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-slate-400 bg-slate-900/50 p-3 rounded-xl border border-slate-800/80">
                  <div className="flex items-center gap-4">
                    <div className="flex items-center gap-1.5">
                      <span className="w-3 h-3 rounded-full bg-emerald-500 inline-block"></span>
                      <span>Green lines: Accepted ORB Keypoint Matches (Lowe distance ratio &lt; 0.7)</span>
                    </div>
                  </div>
                  <span className="font-mono text-emerald-400">
                    Displaying top 40 best Hamming-distance matches
                  </span>
                </div>
              </div>
            )}

            {/* ========================================================= */}
            {/* 5-STAGE BIOMETRIC PREVIEW & GROOVE TUNING CONTROL PANEL   */}
            {/* ========================================================= */}
            {compareResult && (
              <GrooveTuningSection compareResult={compareResult} />
            )}

            {/* ========================================================= */}
            {/* STEP 5: REAL-TIME LOGS TERMINAL                           */}
            {/* ========================================================= */}
            <div className="terminal-panel rounded-2xl p-4 flex flex-col gap-2.5">
              <div className="flex items-center justify-between border-b border-slate-800/80 pb-2">
                <div className="flex items-center gap-2 text-xs font-bold text-slate-300">
                  <Terminal className="w-4 h-4 text-emerald-400" />
                  <span>Real-Time Biometric Verification Execution Stream</span>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setCompareLogs([])}
                    className="text-[11px] text-slate-400 hover:text-slate-200 transition-all px-2 py-0.5 rounded bg-slate-800/60"
                  >
                    Clear Terminal
                  </button>
                  <span className="text-[10px] text-slate-500 font-mono">
                    {compareRunning ? 'SSE Streaming...' : 'Idle'}
                  </span>
                </div>
              </div>

              <div className="h-44 overflow-y-auto font-mono text-xs text-slate-300 space-y-1 pr-2">
                {compareLogs.length === 0 ? (
                  <div className="text-slate-600 italic py-2">
                    Ready. Click "Compare 2 Cattle Videos" to stream real-time microsecond detection, super-resolution &amp; ORB matching logs...
                  </div>
                ) : (
                  compareLogs.map((l, i) => {
                    const isStep = l.text.includes('[Step');
                    const isResult = l.text.includes('RESULT:') || l.text.includes('MATCH:');
                    const isErr = l.text.includes('ERROR') || l.text.includes('Failed');
                    const isConfig = l.text.includes('[CONFIG]') || l.text.includes('[INIT]');

                    let textColor = 'text-slate-400';
                    if (isResult) textColor = 'text-emerald-300 font-bold';
                    else if (isStep) textColor = 'text-cyan-300 font-semibold';
                    else if (isConfig) textColor = 'text-purple-300';
                    else if (isErr) textColor = 'text-rose-400 font-bold';

                    return (
                      <div key={i} className={`leading-relaxed break-all ${textColor}`}>
                        <span className="text-slate-600 select-none mr-2">[{l.time}]</span>
                        {l.text}
                      </div>
                    );
                  })
                )}
                <div ref={compareLogsEndRef} />
              </div>
            </div>
          </div>
        )}

        {/* ========================================================= */}
        {/* TAB 2: SINGLE-VIDEO FRAME EXTRACTION & DETECTION           */}
        {/* ========================================================= */}
        {activeTab === 'viewer' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Left Controls Column (4 cols) */}
            <div className="lg:col-span-4 flex flex-col gap-5">
              {/* Hardware Selection Card */}
              <div className="glass-panel p-5 rounded-2xl flex flex-col gap-3">
                <div className="flex items-center justify-between">
                  <h2 className="text-sm font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
                    <Cpu className="w-4 h-4 text-emerald-400" />
                    Target Hardware
                  </h2>
                  <span className="text-[11px] text-slate-400">Select execution engine</span>
                </div>

                <div className="grid grid-cols-1 gap-2.5">
                  {/* GPU Option */}
                  <label
                    className={`relative flex items-center justify-between p-3.5 rounded-xl border cursor-pointer transition-all ${
                      device === 'gpu'
                        ? 'bg-emerald-950/40 border-emerald-500/50 shadow-sm shadow-emerald-500/10 ring-1 ring-emerald-500/30'
                        : 'bg-slate-900/50 border-slate-800 hover:border-slate-700'
                    }`}
                  >
                    <div className="flex items-center gap-3">
                      <input
                        type="radio"
                        name="hardware"
                        value="gpu"
                        checked={device === 'gpu'}
                        onChange={() => setDevice('gpu')}
                        className="text-emerald-500 focus:ring-emerald-500"
                      />
                      <div>
                        <div className="text-sm font-bold text-white flex items-center gap-2">
                          NVIDIA RTX 3050 GPU
                          <span className="text-[10px] font-semibold uppercase bg-emerald-500/20 text-emerald-300 px-1.5 py-0.5 rounded">
                            Fastest
                          </span>
                        </div>
                        <div className="text-xs text-slate-400">~325 ms latency • 1.9 FPS throughput</div>
                      </div>
                    </div>
                    <Zap className="w-5 h-5 text-emerald-400" />
                  </label>

                  {/* AMD Ryzen CPU (Multi-Core Allocator) */}
                  <label
                    className={`relative flex items-center justify-between p-3.5 rounded-xl border cursor-pointer transition-all ${
                      device === 'cpu'
                        ? 'bg-sky-950/40 border-sky-500/50 shadow-sm shadow-sky-500/10 ring-1 ring-sky-500/30'
                        : 'bg-slate-900/50 border-slate-800 hover:border-slate-700'
                    }`}
                  >
                    <div className="flex items-center gap-3">
                      <input
                        type="radio"
                        name="hardware"
                        value="cpu"
                        checked={device === 'cpu'}
                        onChange={() => setDevice('cpu')}
                        className="text-sky-500 focus:ring-sky-500"
                      />
                      <div>
                        <div className="text-sm font-bold text-white flex items-center gap-2">
                          AMD Ryzen CPU (Multi-Core Allocator)
                          <span className="text-[10px] font-semibold uppercase bg-sky-500/20 text-sky-300 px-1.5 py-0.5 rounded font-mono">
                            {getCpuLatency(cpuCores)}
                          </span>
                        </div>
                        <div className="text-xs text-slate-400">
                          {cpuCores} {cpuCores === 1 ? 'Core' : cpuCores === 12 ? 'Threads' : 'Cores'} allocated • Dynamic affinity
                        </div>
                      </div>
                    </div>
                    <Cpu className="w-5 h-5 text-sky-400" />
                  </label>

                  {/* CPU Multi-Core Allocator Controls (Shown when AMD Ryzen CPU is active) */}
                  {device === 'cpu' && (
                    <div className="p-3.5 bg-slate-900/90 border border-sky-500/30 rounded-xl flex flex-col gap-3 shadow-inner">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <Sliders className="w-4 h-4 text-sky-400" />
                          <span className="text-xs font-semibold text-slate-200">Allocated CPU Cores:</span>
                          <span className="text-xs font-mono font-bold text-sky-400">
                            {cpuCores} {cpuCores === 1 ? 'Core' : cpuCores === 12 ? 'Threads (Max Boost)' : 'Cores'}
                          </span>
                        </div>
                        <div className="flex items-center gap-1.5">
                          <span className="text-[10px] text-slate-400">Est. Latency:</span>
                          <span className="text-xs font-mono font-bold bg-sky-500/20 text-sky-300 px-2 py-0.5 rounded border border-sky-500/30">
                            {getCpuLatency(cpuCores)}
                          </span>
                        </div>
                      </div>

                      <div className="flex items-center gap-3">
                        <input
                          type="range"
                          min="1"
                          max="12"
                          step="1"
                          value={cpuCores}
                          onChange={(e) => setCpuCores(parseInt(e.target.value, 10))}
                          className="w-full h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-sky-400"
                        />
                        <input
                          type="number"
                          min="1"
                          max="12"
                          value={cpuCores}
                          onChange={(e) => {
                            const val = Math.max(1, Math.min(12, parseInt(e.target.value, 10) || 1));
                            setCpuCores(val);
                          }}
                          className="w-12 bg-slate-800 border border-slate-700 text-center text-xs font-mono font-bold text-white py-1 rounded-lg focus:outline-none focus:border-sky-400"
                        />
                      </div>

                      <div className="flex flex-wrap gap-1.5 pt-1.5 border-t border-slate-800">
                        <span className="text-[10px] text-slate-400 self-center mr-1">Presets:</span>
                        {CPU_PRESETS.map((preset) => (
                          <button
                            key={preset.cores}
                            type="button"
                            onClick={() => setCpuCores(preset.cores)}
                            className={`px-2.5 py-1 rounded-lg text-xs font-medium transition-all ${
                              cpuCores === preset.cores
                                ? 'bg-sky-500 text-slate-950 font-bold shadow-sm shadow-sky-500/30'
                                : 'bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700/60'
                            }`}
                          >
                            {preset.label}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* Source Input Card */}
              <div className="glass-panel p-5 rounded-2xl flex flex-col gap-4">
                <div className="flex items-center justify-between">
                  <h2 className="text-sm font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
                    <Video className="w-4 h-4 text-emerald-400" />
                    Input Source
                  </h2>
                  <span className="text-[11px] text-slate-400">Video / Image</span>
                </div>

                {/* Input Mode Selector */}
                <div className="grid grid-cols-3 gap-1 bg-slate-900 p-1 rounded-xl border border-slate-800 text-xs">
                  <button
                    onClick={() => setInputMode('sample_video')}
                    className={`py-1.5 px-2 rounded-lg font-medium transition-all ${
                      inputMode === 'sample_video' ? 'bg-slate-800 text-white font-bold' : 'text-slate-400 hover:text-white'
                    }`}
                  >
                    Sample Video
                  </button>
                  <button
                    onClick={() => setInputMode('upload_video')}
                    className={`py-1.5 px-2 rounded-lg font-medium transition-all ${
                      inputMode === 'upload_video' ? 'bg-slate-800 text-white font-bold' : 'text-slate-400 hover:text-white'
                    }`}
                  >
                    Upload Video
                  </button>
                  <button
                    onClick={() => setInputMode('sample_image')}
                    className={`py-1.5 px-2 rounded-lg font-medium transition-all ${
                      inputMode === 'sample_image' ? 'bg-slate-800 text-white font-bold' : 'text-slate-400 hover:text-white'
                    }`}
                  >
                    Still Photo
                  </button>
                </div>

                {/* Mode: Sample Video */}
                {inputMode === 'sample_video' && (
                  <div className="flex flex-col gap-2">
                    <label className="text-xs text-slate-400">Choose Sample Video:</label>
                    <select
                      value={selectedSampleVideo}
                      onChange={(e) => setSelectedSampleVideo(e.target.value)}
                      className="bg-slate-900 border border-slate-800 rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-emerald-500 font-mono"
                    >
                      {sampleVideos.map((v, i) => (
                        <option key={i} value={v.path}>
                          {v.name} ({v.size_mb} MB)
                        </option>
                      ))}
                    </select>
                  </div>
                )}

                {/* Mode: Upload Video */}
                {inputMode === 'upload_video' && (
                  <div>
                    <input
                      ref={fileInputRefSingle}
                      type="file"
                      accept="video/*,image/*"
                      onChange={handleFileUploadSingle}
                      className="hidden"
                    />
                    <button
                      onClick={() => fileInputRefSingle.current?.click()}
                      className="w-full border-2 border-dashed border-slate-700 hover:border-emerald-500/60 p-4 rounded-xl flex flex-col items-center justify-center gap-2 text-slate-400 hover:text-white transition-all bg-slate-900/30"
                    >
                      <Upload className="w-5 h-5 text-emerald-400" />
                      <span className="text-xs font-semibold">
                        {uploadedFile ? uploadedFile.name : 'Click to upload MP4/AVI cattle video'}
                      </span>
                    </button>
                  </div>
                )}

                {/* Mode: Sample Image */}
                {inputMode === 'sample_image' && (
                  <div className="flex flex-col gap-2">
                    <label className="text-xs text-slate-400">Select Test Cattle Image:</label>
                    <select
                      value={selectedSampleImage}
                      onChange={(e) => setSelectedSampleImage(e.target.value)}
                      className="bg-slate-900 border border-slate-800 rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-emerald-500"
                    >
                      {status?.sample_images?.map((img) => (
                        <option key={img} value={img}>
                          {img}
                        </option>
                      ))}
                    </select>
                  </div>
                )}

                {/* Frame sampling controls for video */}
                {(inputMode === 'sample_video' || inputMode === 'upload_video') && (
                  <div className="grid grid-cols-2 gap-3 pt-2 border-t border-slate-800/80">
                    <div>
                      <label className="text-xs text-slate-400 block mb-1">Frame Interval</label>
                      <select
                        value={frameInterval}
                        onChange={(e) => setFrameInterval(Number(e.target.value))}
                        className="w-full bg-slate-900 border border-slate-800 rounded-lg px-2 py-1.5 text-xs text-white"
                      >
                        <option value={5}>Every 5th frame</option>
                        <option value={10}>Every 10th frame</option>
                        <option value={15}>Every 15th frame</option>
                        <option value={30}>Every 30th frame</option>
                      </select>
                    </div>
                    <div>
                      <label className="text-xs text-slate-400 block mb-1">Max Sample Frames</label>
                      <select
                        value={maxFrames}
                        onChange={(e) => setMaxFrames(Number(e.target.value))}
                        className="w-full bg-slate-900 border border-slate-800 rounded-lg px-2 py-1.5 text-xs text-white"
                      >
                        <option value={3}>3 Frames (Quick)</option>
                        <option value={6}>6 Frames</option>
                        <option value={12}>12 Frames</option>
                        <option value={25}>25 Frames</option>
                      </select>
                    </div>
                  </div>
                )}

                {/* Crop & Confidence Controls */}
                <div className="pt-2 border-t border-slate-800/80 flex flex-col gap-3">
                  <div>
                    <div className="flex items-center justify-between text-xs text-slate-400 mb-1">
                      <span>Detection Confidence:</span>
                      <span className="font-mono text-emerald-400 font-bold">{Math.round(confidence * 100)}%</span>
                    </div>
                    <input
                      type="range"
                      min="0.5"
                      max="0.95"
                      step="0.05"
                      value={confidence}
                      onChange={(e) => setConfidence(Number(e.target.value))}
                      className="w-full accent-emerald-500 h-1.5 bg-slate-800 rounded-lg cursor-pointer"
                    />
                  </div>

                  <div className="flex items-center justify-between text-xs pt-1">
                    <label className="flex items-center gap-2 cursor-pointer text-slate-300">
                      <input
                        type="checkbox"
                        checked={square}
                        onChange={(e) => setSquare(e.target.checked)}
                        className="rounded text-emerald-500 focus:ring-0"
                      />
                      <span>1:1 Square Crop (Padding 8%)</span>
                    </label>
                    <span className="text-[11px] font-mono text-slate-400">224x224 px</span>
                  </div>
                </div>

                {/* Primary Action Button */}
                <div className="pt-2">
                  {!isProcessing ? (
                    <button
                      onClick={startDetection}
                      className="w-full py-3.5 px-4 rounded-xl bg-gradient-to-r from-emerald-500 to-teal-500 hover:from-emerald-400 hover:to-teal-400 text-slate-950 font-bold flex items-center justify-center gap-2 text-sm shadow-lg shadow-emerald-500/20 transition-all cursor-pointer"
                    >
                      <Play className="w-4 h-4 fill-current" />
                      Extract Frames &amp; Crop Muzzle
                    </button>
                  ) : (
                    <button
                      onClick={stopProcessing}
                      className="w-full py-3.5 px-4 rounded-xl bg-rose-600 hover:bg-rose-500 text-white font-bold flex items-center justify-center gap-2 text-sm shadow-lg shadow-rose-600/20 transition-all cursor-pointer"
                    >
                      <Square className="w-4 h-4 fill-current" />
                      Stop Pipeline Execution
                    </button>
                  )}
                </div>
              </div>
            </div>

            {/* Right Display & Logs Column (8 cols) */}
            <div className="lg:col-span-8 flex flex-col gap-5">
              {/* Live Metric Banner */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <div className="glass-panel p-4 rounded-xl">
                  <div className="text-[11px] font-bold uppercase tracking-wider text-slate-400">Processed Frames</div>
                  <div className="text-2xl font-extrabold text-white mt-1">{metrics.processedFrames}</div>
                  <div className="text-[10px] text-slate-500 mt-0.5">Sampled from video</div>
                </div>
                <div className="glass-panel p-4 rounded-xl">
                  <div className="text-[11px] font-bold uppercase tracking-wider text-slate-400">Muzzles Extracted</div>
                  <div className="text-2xl font-extrabold text-emerald-400 mt-1">{metrics.totalMuzzles}</div>
                  <div className="text-[10px] text-slate-500 mt-0.5">Confidence &gt; {Math.round(confidence * 100)}%</div>
                </div>
                <div className="glass-panel p-4 rounded-xl">
                  <div className="text-[11px] font-bold uppercase tracking-wider text-slate-400">Inference Latency</div>
                  <div className="text-2xl font-extrabold text-sky-400 mt-1 font-mono">
                    {metrics.latestInferenceMs.toFixed(1)} <span className="text-xs font-normal text-slate-400">ms</span>
                  </div>
                  <div className="text-[10px] text-slate-500 mt-0.5">
                    {device === 'gpu' ? 'RTX 3050' : `CPU (${cpuCores === 12 ? '12 Threads' : `${cpuCores} Cores`})`}
                  </div>
                </div>
                <div className="glass-panel p-4 rounded-xl">
                  <div className="text-[11px] font-bold uppercase tracking-wider text-slate-400">Processing Speed</div>
                  <div className="text-2xl font-extrabold text-teal-400 mt-1 font-mono">
                    {metrics.latestFps.toFixed(2)} <span className="text-xs font-normal text-slate-400">FPS</span>
                  </div>
                  <div className="text-[10px] text-slate-500 mt-0.5">End-to-end frame rate</div>
                </div>
              </div>

              {/* Main Detection & Crop Display */}
              <div className="glass-panel p-5 rounded-2xl flex flex-col gap-4">
                <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                  <h3 className="text-sm font-bold text-white flex items-center gap-2">
                    <Maximize2 className="w-4 h-4 text-emerald-400" />
                    Detection &amp; Cropped Cow Muzzle Output
                  </h3>
                  {latestCrop && (
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 font-mono">
                        Conf: {(latestCrop.confidence * 100).toFixed(1)}%
                      </span>
                      <a
                        href={latestCrop.crop_url}
                        download
                        className="text-xs flex items-center gap-1 bg-slate-800 hover:bg-slate-700 px-2.5 py-1 rounded-lg text-slate-200 transition-all"
                      >
                        <Download className="w-3.5 h-3.5" />
                        Download Crop
                      </a>
                    </div>
                  )}
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {/* Left: Annotated Input Frame */}
                  <div className="flex flex-col gap-2">
                    <div className="text-xs font-bold text-slate-400 flex items-center justify-between">
                      <span>Full Video Frame (Annotated Detection)</span>
                      <span className="text-[11px] text-slate-500 font-mono">BBox &amp; Mask</span>
                    </div>
                    <div className="aspect-video bg-slate-900 rounded-xl overflow-hidden border border-slate-800 flex items-center justify-center relative">
                      {latestFrame ? (
                        <img
                          src={latestFrame}
                          alt="Annotated Detection"
                          className="w-full h-full object-contain"
                        />
                      ) : (
                        <div className="flex flex-col items-center justify-center gap-2 text-slate-600 p-6 text-center">
                          <Video className="w-8 h-8 stroke-1" />
                          <span className="text-xs">Extracted video frames with detection box will appear here</span>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Right: Cropped Cow Muzzle */}
                  <div className="flex flex-col gap-2">
                    <div className="text-xs font-bold text-slate-400 flex items-center justify-between">
                      <span>Extracted Cow Muzzle (1:1 Biometric Crop)</span>
                      <span className="text-[11px] text-emerald-400 font-mono">224x224 ArcFace</span>
                    </div>
                    <div className="aspect-video bg-slate-900 rounded-xl overflow-hidden border border-slate-800 flex items-center justify-center relative">
                      {latestCrop ? (
                        <div className="flex items-center justify-center w-full h-full p-2">
                          <img
                            src={latestCrop.crop_url}
                            alt="Cropped Muzzle"
                            className="max-h-full max-w-full aspect-square object-cover rounded-lg shadow-xl border border-emerald-500/30"
                          />
                        </div>
                      ) : (
                        <div className="flex flex-col items-center justify-center gap-2 text-slate-600 p-6 text-center">
                          <div className="text-2xl">🐮</div>
                          <span className="text-xs">Cropped muzzle pattern (noseprint) will display here</span>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>

              {/* Real-Time Millisecond Logs Terminal */}
              <div className="terminal-panel rounded-2xl p-4 flex flex-col gap-2.5">
                <div className="flex items-center justify-between border-b border-slate-800/80 pb-2">
                  <div className="flex items-center gap-2 text-xs font-bold text-slate-300">
                    <Terminal className="w-4 h-4 text-emerald-400" />
                    <span>Real-Time Latency &amp; Execution Logs</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => setLogs([])}
                      className="text-[11px] text-slate-400 hover:text-slate-200 transition-all px-2 py-0.5 rounded bg-slate-800/60"
                    >
                      Clear
                    </button>
                    <span className="text-[10px] text-slate-500 font-mono">Streaming active</span>
                  </div>
                </div>

                <div className="h-44 overflow-y-auto font-mono text-xs text-slate-300 space-y-1 pr-2">
                  {logs.length === 0 ? (
                    <div className="text-slate-600 italic py-2">
                      Ready. Click "Extract Frames &amp; Crop Muzzle" to view microsecond timing logs...
                    </div>
                  ) : (
                    logs.map((l, i) => {
                      const isTiming = l.text.includes('[TIMING]');
                      const isMuzzle = l.text.includes('Detected Muzzle');
                      const isHw = l.text.includes('[Hardware]');
                      const isErr = l.text.includes('ERROR') || l.text.includes('Failed');

                      let textColor = 'text-slate-400';
                      if (isTiming) textColor = 'text-cyan-300 font-medium';
                      else if (isMuzzle) textColor = 'text-emerald-300 font-bold';
                      else if (isHw) textColor = 'text-purple-300';
                      else if (isErr) textColor = 'text-rose-400 font-bold';

                      return (
                        <div key={i} className={`leading-relaxed break-all ${textColor}`}>
                          <span className="text-slate-600 select-none mr-2">[{l.time}]</span>
                          {l.text}
                        </div>
                      );
                    })
                  )}
                  <div ref={logsEndRef} />
                </div>
              </div>

              {/* Extracted Muzzles Gallery */}
              {allCrops.length > 0 && (
                <div className="glass-panel p-5 rounded-2xl flex flex-col gap-3">
                  <div className="flex items-center justify-between">
                    <h3 className="text-sm font-bold text-white flex items-center gap-2">
                      <Layers className="w-4 h-4 text-emerald-400" />
                      All Extracted Muzzles ({allCrops.length})
                    </h3>
                    <span className="text-xs text-slate-400">Click to preview or download</span>
                  </div>

                  <div className="grid grid-cols-3 sm:grid-cols-6 gap-3">
                    {allCrops.map((crop, idx) => (
                      <div
                        key={idx}
                        className="group relative bg-slate-900 rounded-xl overflow-hidden border border-slate-800 hover:border-emerald-500/50 transition-all p-1 flex flex-col items-center"
                      >
                        <img
                          src={crop.crop_url}
                          alt={`Muzzle ${idx}`}
                          className="w-full aspect-square object-cover rounded-lg"
                        />
                        <div className="text-[10px] font-mono text-emerald-400 mt-1 font-semibold">
                          {(crop.confidence * 100).toFixed(1)}%
                        </div>
                        <a
                          href={crop.crop_url}
                          download
                          className="absolute inset-0 bg-slate-950/80 backdrop-blur-xs flex items-center justify-center opacity-0 group-hover:opacity-100 transition-all text-xs font-semibold text-emerald-400 gap-1"
                        >
                          <Download className="w-3.5 h-3.5" />
                          Save
                        </a>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* ========================================================= */}
        {/* TAB 3: HARDWARE PERFORMANCE BENCHMARK                     */}
        {/* ========================================================= */}
        {activeTab === 'benchmark' && (
          <div className="glass-panel p-6 rounded-2xl flex flex-col gap-6">
            <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800 pb-4">
              <div>
                <h2 className="text-lg font-bold text-white flex items-center gap-2">
                  <BarChart3 className="w-5 h-5 text-emerald-400" />
                  Comparative Hardware Benchmark Suite
                </h2>
                <p className="text-xs text-slate-400 mt-0.5">
                  Benchmarks the cow muzzle model across GPU (RTX 3050), Ryzen 12 Threads, and Ryzen 2 Cores.
                </p>
              </div>

              <button
                onClick={runBenchmark}
                disabled={benchmarkRunning}
                className={`py-2.5 px-5 rounded-xl font-bold text-sm flex items-center gap-2 shadow-lg transition-all cursor-pointer ${
                  benchmarkRunning
                    ? 'bg-slate-800 text-slate-500 cursor-not-allowed'
                    : 'bg-emerald-500 hover:bg-emerald-400 text-slate-950 shadow-emerald-500/20'
                }`}
              >
                <RefreshCw className={`w-4 h-4 ${benchmarkRunning ? 'animate-spin' : ''}`} />
                {benchmarkRunning ? 'Running Benchmark...' : 'Run Comparative Benchmark'}
              </button>
            </div>

            {/* Benchmark Live Results Table */}
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead>
                  <tr className="border-b border-slate-800 text-xs font-bold text-slate-400 uppercase tracking-wider">
                    <th className="py-3 px-4">Hardware Profile</th>
                    <th className="py-3 px-4">Mean Inference Time</th>
                    <th className="py-3 px-4">Median (p50)</th>
                    <th className="py-3 px-4">End-to-End Latency</th>
                    <th className="py-3 px-4">Throughput (FPS)</th>
                    <th className="py-3 px-4">Speedup Factor</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 font-mono text-xs">
                  {(benchmarkResults || [
                    {
                      profile: 'GPU (RTX 3050 Laptop)',
                      mean_inference_ms: 325.1,
                      median_inference_ms: 325.1,
                      mean_total_ms: 524.7,
                      steady_fps: 1.91,
                      speedup: '20.3x faster',
                      badge: 'Fastest (Recommended)',
                    },
                    {
                      profile: 'CPU (Kernel Default - 12 Threads)',
                      mean_inference_ms: 2905.3,
                      median_inference_ms: 2905.3,
                      mean_total_ms: 3124.8,
                      steady_fps: 0.32,
                      speedup: '3.4x faster',
                      badge: 'Ryzen CPU',
                    },
                    {
                      profile: 'CPU (2 Cores Restricted)',
                      mean_inference_ms: 10346.9,
                      median_inference_ms: 10346.9,
                      mean_total_ms: 10643.6,
                      steady_fps: 0.09,
                      speedup: '1.0x (Baseline)',
                      badge: 'Constrained',
                    },
                  ]).map((item, idx) => (
                    <tr key={idx} className="hover:bg-slate-900/50 transition-colors">
                      <td className="py-3.5 px-4 font-sans font-bold text-white flex items-center gap-2">
                        {item.profile.includes('GPU') ? (
                          <Zap className="w-4 h-4 text-emerald-400" />
                        ) : (
                          <Cpu className="w-4 h-4 text-sky-400" />
                        )}
                        <span>{item.profile}</span>
                        {item.badge && (
                          <span className="text-[10px] font-sans font-semibold px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                            {item.badge}
                          </span>
                        )}
                      </td>
                      <td className="py-3.5 px-4 text-emerald-400 font-bold">
                        {item.mean_inference_ms ? item.mean_inference_ms.toFixed(1) : '-'} ms
                      </td>
                      <td className="py-3.5 px-4 text-slate-300">
                        {item.median_inference_ms ? item.median_inference_ms.toFixed(1) : '-'} ms
                      </td>
                      <td className="py-3.5 px-4 text-slate-300">
                        {item.mean_total_ms ? item.mean_total_ms.toFixed(1) : '-'} ms
                      </td>
                      <td className="py-3.5 px-4 text-teal-300 font-bold">
                        {item.steady_fps ? item.steady_fps.toFixed(2) : '-'} FPS
                      </td>
                      <td className="py-3.5 px-4 text-purple-300 font-bold">
                        {item.speedup || (idx === 0 ? '20.3x' : idx === 1 ? '3.4x' : '1.0x')}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Benchmark Console Logs */}
            <div className="terminal-panel rounded-xl p-4 flex flex-col gap-2">
              <div className="text-xs font-bold text-slate-400 flex items-center gap-2">
                <Terminal className="w-3.5 h-3.5 text-emerald-400" />
                Benchmark Execution Stream
              </div>
              <div className="h-44 overflow-y-auto font-mono text-xs text-slate-300 space-y-1">
                {logs.map((l, i) => (
                  <div key={i} className="text-slate-400">
                    <span className="text-slate-600 mr-2">[{l.time}]</span>
                    {l.text}
                  </div>
                ))}
                <div ref={logsEndRef} />
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-900 px-6 py-4 text-center text-xs text-slate-500">
        Matterport Mask R-CNN Cattle Muzzle Detection &amp; statusprint.py ORB Verification • Lanczos-4 Sinc Super-Resolution
      </footer>
    </div>
  );
}
