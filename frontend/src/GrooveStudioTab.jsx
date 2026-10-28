import React, { useState, useEffect, useCallback } from 'react';
import {
  ScanLine,
  Sliders,
  Play,
  RotateCcw,
  Sparkles,
  Maximize2,
  CheckCircle2,
  Activity,
  Layers,
  Clock,
  Download,
  Eye,
  FileVideo,
  Upload,
  Cpu,
  Zap,
  ZoomIn,
  ZoomOut,
  RefreshCw,
  X,
  ChevronLeft,
  ChevronRight
} from 'lucide-react';

const KERNEL_PRESETS = {
  fine: { label: 'Fine [7, 11, 15]', scales: [7, 11, 15], desc: 'Narrow micro-capillary grooves' },
  standard: { label: 'Standard [9, 15, 21]', scales: [9, 15, 21], desc: 'Optimal balance for cow muzzle valleys' },
  broad: { label: 'Broad [11, 19, 27]', scales: [11, 19, 27], desc: 'Wide inter-bead boundary valleys' },
};

export default function GrooveStudioTab({ sampleVideos = [] }) {
  // Video selection
  const [selectedVideo, setSelectedVideo] = useState('/home/adis_mayur/F_CODES/muzzleVideo.mp4');
  const [customVideoPath, setCustomVideoPath] = useState('');
  const [isProcessingVideo, setIsProcessingVideo] = useState(false);
  const [isTuning, setIsTuning] = useState(false);
  const [errorMsg, setErrorMsg] = useState(null);

  // Active groove parameters
  const [method, setMethod] = useState('blackhat'); // 'blackhat' | 'frangi'
  const [thresholdMethod, setThresholdMethod] = useState('otsu'); // 'otsu' | 'adaptive'
  const [presetKey, setPresetKey] = useState('standard');
  const [scales, setScales] = useState([9, 15, 21]);
  const [minComponentSize, setMinComponentSize] = useState(15);
  const [skeletonMode, setSkeletonMode] = useState('centerline'); // 'centerline' | 'overlay'

  // Groove result state
  const [grooveData, setGrooveData] = useState(null);
  const [viewMode, setViewMode] = useState('grid'); // 'grid' | 'focus'
  const [selectedStage, setSelectedStage] = useState(4); // 0..4

  // Fullscreen lightbox
  const [lightboxOpen, setLightboxOpen] = useState(false);
  const [lightboxStage, setLightboxStage] = useState(4);
  const [lightboxZoom, setLightboxZoom] = useState(100);

  // Pre-fetch sample result on mount
  useEffect(() => {
    fetchSampleResult();
  }, []);

  const fetchSampleResult = async () => {
    try {
      const res = await fetch('/api/grooves/sample-result');
      if (res.ok) {
        const data = await res.json();
        setGrooveData(data);
      }
    } catch (err) {
      console.warn('Could not load precomputed sample result:', err);
    }
  };

  // Run full video processing
  const handleProcessVideo = async () => {
    setIsProcessingVideo(true);
    setErrorMsg(null);
    try {
      const videoToUse = customVideoPath.trim() || selectedVideo;
      const res = await fetch('/api/grooves/process-video', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          video_path: videoToUse,
          method,
          scales,
          threshold_method: thresholdMethod,
          min_component_size: minComponentSize,
        }),
      });

      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.detail || `Server returned ${res.status}`);
      }

      const data = await res.json();
      setGrooveData(data);
    } catch (err) {
      setErrorMsg(err.message || 'Video processing failed.');
    } finally {
      setIsProcessingVideo(false);
    }
  };

  // Real-time tune parameters on active crop (~30ms)
  const handleTune = async (overrideParams = {}) => {
    if (!grooveData?.crop_path) return;
    setIsTuning(true);
    setErrorMsg(null);

    const m = overrideParams.method || method;
    const t = overrideParams.thresholdMethod || thresholdMethod;
    const s = overrideParams.scales || scales;
    const minS = overrideParams.minComponentSize !== undefined ? overrideParams.minComponentSize : minComponentSize;

    try {
      const res = await fetch('/api/grooves/tune', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_path: grooveData.crop_path,
          method: m,
          scales: s,
          threshold_method: t,
          min_component_size: minS,
          skeleton_enabled: true,
        }),
      });

      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.detail || 'Tuning failed');
      }

      const tuneRes = await res.json();
      setGrooveData((prev) => ({
        ...prev,
        method: m,
        threshold_method: t,
        scales: s,
        min_component_size: minS,
        groove_pixel_count: tuneRes.groove_pixel_count,
        skeleton_pixel_count: tuneRes.skeleton_pixel_count,
        timings: {
          ...(prev?.timings || {}),
          groove_ms: tuneRes.timings.total_ms,
        },
        artifacts: {
          ...(prev?.artifacts || {}),
          groove_response: `${tuneRes.groove_response_url}?t=${Date.now()}`,
          groove_mask: `${tuneRes.groove_mask_url}?t=${Date.now()}`,
          groove_skeleton: `${tuneRes.groove_skeleton_url}?t=${Date.now()}`,
          groove_overlay: `${tuneRes.groove_overlay_url}?t=${Date.now()}`,
        },
      }));
    } catch (err) {
      setErrorMsg(err.message || 'Tuning failed');
    } finally {
      setIsTuning(false);
    }
  };

  const stages = [
    {
      id: 0,
      title: 'Stage 1: Native Muzzle Crop',
      shortTitle: '1. Native Crop',
      url: grooveData?.artifacts?.native_crop,
      badge: 'Mask R-CNN',
      badgeColor: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
      desc: 'Raw biometric ROI crop preserved at native sensor resolution.',
    },
    {
      id: 1,
      title: 'Stage 2: Grayscale Ridge Enhanced',
      shortTitle: '2. Ridge Enhanced',
      url: grooveData?.artifacts?.enhanced_ridge,
      badge: 'Lanczos-4 + CLAHE',
      badgeColor: 'bg-cyan-500/20 text-cyan-400 border-cyan-500/30',
      desc: '2x Lanczos super-resolution + local adaptive contrast enhancement.',
    },
    {
      id: 2,
      title: 'Stage 3: Multi-Scale Groove Response',
      shortTitle: '3. Groove Response',
      url: grooveData?.artifacts?.groove_response,
      badge: method === 'blackhat' ? 'Morphological Black-Hat' : 'Frangi Hessian',
      badgeColor: 'bg-teal-500/20 text-teal-400 border-teal-500/30',
      desc: 'Differential response highlighting dark line-shaped groove depressions.',
    },
    {
      id: 3,
      title: 'Stage 4: Binary Groove Mask',
      shortTitle: '4. Binary Mask',
      url: grooveData?.artifacts?.groove_mask,
      badge: `${thresholdMethod.toUpperCase()} + CC Pruning`,
      badgeColor: 'bg-indigo-500/20 text-indigo-400 border-indigo-500/30',
      desc: 'White pixels = detected grooves, black = skin beads, background & nostrils.',
    },
    {
      id: 4,
      title: 'Stage 5: 1px Topological Skeleton',
      shortTitle: '5. Thin Skeleton',
      url: skeletonMode === 'centerline' ? grooveData?.artifacts?.groove_skeleton : grooveData?.artifacts?.groove_overlay,
      badge: skeletonMode === 'centerline' ? '1-px Medial Axis' : 'Color Overlay',
      badgeColor: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30',
      desc: 'Thin 1-pixel-wide centerline network preserving bifurcations and branch topology.',
    },
  ];

  return (
    <div className="flex flex-col gap-6">
      {/* ========================================================= */}
      {/* HERO BANNER: NOSE_LINES PROJECT BRANDING                   */}
      {/* ========================================================= */}
      <div className="glass-panel p-6 rounded-2xl border border-emerald-500/30 bg-gradient-to-r from-slate-900/90 via-slate-900/70 to-emerald-950/40 shadow-2xl relative overflow-hidden">
        <div className="absolute top-0 right-0 w-96 h-96 bg-emerald-500/5 rounded-full blur-3xl pointer-events-none" />
        <div className="flex flex-wrap items-center justify-between gap-4 relative z-10">
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 rounded-2xl bg-gradient-to-tr from-emerald-500 to-teal-400 flex items-center justify-center text-slate-950 shadow-lg shadow-emerald-500/30 font-bold text-2xl">
              <ScanLine className="w-7 h-7 text-slate-950" />
            </div>
            <div>
              <div className="flex items-center gap-3">
                <h2 className="text-xl font-bold tracking-tight text-white">
                  Cattle Muzzle Groove Extraction &amp; Ridge Skeletonization
                </h2>
                <span className="px-2.5 py-0.5 rounded-full text-xs font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 font-mono">
                  nose_lines Studio
                </span>
              </div>
              <p className="text-xs text-slate-400 mt-1">
                Multi-Scale Morphological Black-Hat &amp; Frangi Hessian Filtering • Binary Valley Segmentation • 1-px Topological Ridge Skeletonization
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-xs font-mono text-slate-400 px-3 py-1.5 rounded-lg bg-slate-950/60 border border-slate-800 flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              Engine: 2 CPU Cores / Python OpenCV
            </span>
          </div>
        </div>
      </div>

      {/* ========================================================= */}
      {/* CONTROL PANEL: VIDEO INPUT & PARAMETER TUNING             */}
      {/* ========================================================= */}
      <div className="glass-panel p-5 rounded-2xl border border-slate-800 flex flex-col gap-5 shadow-xl">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-3">
          <div className="flex items-center gap-2">
            <Sliders className="w-4 h-4 text-emerald-400" />
            <h3 className="text-sm font-bold text-white uppercase tracking-wider">
              1. Video Input &amp; Groove Extraction Parameters
            </h3>
          </div>
          {grooveData && (
            <div className="flex items-center gap-3 text-xs font-mono text-slate-400">
              <span className="text-emerald-400 font-bold">
                {grooveData.groove_pixel_count?.toLocaleString()} groove px ({grooveData.groove_coverage_pct || '22.4'}%)
              </span>
              <span>•</span>
              <span className="text-teal-300">
                {grooveData.skeleton_pixel_count?.toLocaleString()} skeleton px
              </span>
              <span>•</span>
              <span className="text-sky-300">
                {grooveData.timings?.groove_ms || 14} ms
              </span>
            </div>
          )}
        </div>

        {/* Top Controls Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* 1. Video Selector */}
          <div className="flex flex-col gap-2 p-3 rounded-xl bg-slate-950/50 border border-slate-800">
            <label className="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
              <FileVideo className="w-3.5 h-3.5 text-emerald-400" />
              <span>Input Cattle Video</span>
            </label>
            <select
              value={selectedVideo}
              onChange={(e) => setSelectedVideo(e.target.value)}
              className="w-full bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-emerald-500 font-mono"
            >
              {sampleVideos.length > 0 ? (
                sampleVideos.map((v) => (
                  <option key={v.path} value={v.path}>
                    {v.name} ({v.size_mb} MB)
                  </option>
                ))
              ) : (
                <>
                  <option value="/home/adis_mayur/F_CODES/muzzleVideo.mp4">muzzleVideo.mp4 (2.55 MB)</option>
                  <option value="/home/adis_mayur/F_CODES/muzzleVideo2.mp4">muzzleVideo2.mp4 (2.59 MB)</option>
                  <option value="/home/adis_mayur/F_CODES/muzzleVideo3.mp4">muzzleVideo3.mp4 (2.63 MB)</option>
                  <option value="/home/adis_mayur/F_CODES/muzzleVideo5.mp4">muzzleVideo5.mp4 (4.25 MB)</option>
                </>
              )}
            </select>
            <span className="text-[10px] text-slate-500 font-mono truncate">
              Path: {selectedVideo}
            </span>
          </div>

          {/* 2. Enhancement Method */}
          <div className="flex flex-col gap-2 p-3 rounded-xl bg-slate-950/50 border border-slate-800">
            <label className="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5 text-teal-400" />
              <span>Groove Filter Method</span>
            </label>
            <div className="grid grid-cols-2 gap-1 p-0.5 bg-slate-900 rounded-lg border border-slate-800">
              <button
                type="button"
                onClick={() => {
                  setMethod('blackhat');
                  handleTune({ method: 'blackhat' });
                }}
                className={`py-1.5 px-2 rounded text-xs font-medium transition-all ${
                  method === 'blackhat'
                    ? 'bg-emerald-600 text-white font-bold shadow'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                Black-Hat
              </button>
              <button
                type="button"
                onClick={() => {
                  setMethod('frangi');
                  handleTune({ method: 'frangi' });
                }}
                className={`py-1.5 px-2 rounded text-xs font-medium transition-all ${
                  method === 'frangi'
                    ? 'bg-teal-600 text-white font-bold shadow'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                Frangi Hessian
              </button>
            </div>
            <span className="text-[10px] text-slate-500">
              {method === 'blackhat' ? 'Multi-scale morphological closing difference' : 'Hessian eigenvalue curvilinear vesselness'}
            </span>
          </div>

          {/* 3. Thresholding Mode */}
          <div className="flex flex-col gap-2 p-3 rounded-xl bg-slate-950/50 border border-slate-800">
            <label className="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
              <Layers className="w-3.5 h-3.5 text-cyan-400" />
              <span>Binarization Threshold</span>
            </label>
            <div className="grid grid-cols-2 gap-1 p-0.5 bg-slate-900 rounded-lg border border-slate-800">
              <button
                type="button"
                onClick={() => {
                  setThresholdMethod('otsu');
                  handleTune({ thresholdMethod: 'otsu' });
                }}
                className={`py-1.5 px-2 rounded text-xs font-medium transition-all ${
                  thresholdMethod === 'otsu'
                    ? 'bg-emerald-600 text-white font-bold shadow'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                Otsu (Bi-modal)
              </button>
              <button
                type="button"
                onClick={() => {
                  setThresholdMethod('adaptive');
                  handleTune({ thresholdMethod: 'adaptive' });
                }}
                className={`py-1.5 px-2 rounded text-xs font-medium transition-all ${
                  thresholdMethod === 'adaptive'
                    ? 'bg-teal-600 text-white font-bold shadow'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                Adaptive Gaussian
              </button>
            </div>
            <span className="text-[10px] text-slate-500">
              {thresholdMethod === 'otsu' ? 'Global bimodal threshold on masked skin' : 'Local neighborhood Gaussian threshold'}
            </span>
          </div>

          {/* 4. Multi-Scale Kernel Preset */}
          <div className="flex flex-col gap-2 p-3 rounded-xl bg-slate-950/50 border border-slate-800">
            <label className="text-xs font-semibold text-slate-300 flex items-center justify-between">
              <span>Kernel Scales</span>
              <span className="text-[10px] font-mono text-emerald-400">[{scales.join(', ')}] px</span>
            </label>
            <div className="grid grid-cols-3 gap-1 p-0.5 bg-slate-900 rounded-lg border border-slate-800">
              {Object.entries(KERNEL_PRESETS).map(([k, p]) => (
                <button
                  key={k}
                  type="button"
                  onClick={() => {
                    setPresetKey(k);
                    setScales(p.scales);
                    handleTune({ scales: p.scales });
                  }}
                  className={`py-1 px-1.5 rounded text-[11px] font-medium transition-all ${
                    presetKey === k
                      ? 'bg-emerald-600 text-white font-bold shadow'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  {k === 'fine' ? 'Fine' : k === 'standard' ? 'Standard' : 'Broad'}
                </button>
              ))}
            </div>
            <span className="text-[10px] text-slate-500 truncate">
              {KERNEL_PRESETS[presetKey]?.desc}
            </span>
          </div>
        </div>

        {/* Secondary Row: Sliders & Actions */}
        <div className="flex flex-wrap items-center justify-between gap-4 pt-2 border-t border-slate-800/80">
          <div className="flex flex-wrap items-center gap-6">
            {/* Noise sensitivity slider */}
            <div className="flex items-center gap-3">
              <span className="text-xs text-slate-400 font-medium whitespace-nowrap">
                Noise Pruning:
              </span>
              <input
                type="range"
                min="5"
                max="50"
                step="1"
                value={minComponentSize}
                onChange={(e) => {
                  const val = parseInt(e.target.value, 10);
                  setMinComponentSize(val);
                }}
                onMouseUp={() => handleTune({ minComponentSize })}
                onTouchEnd={() => handleTune({ minComponentSize })}
                className="w-32 h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-emerald-500"
              />
              <span className="text-xs font-mono text-emerald-300 w-16">
                min {minComponentSize} px
              </span>
            </div>

            {/* Skeleton display mode */}
            <div className="flex items-center gap-2">
              <span className="text-xs text-slate-400 font-medium">Skeleton Preview:</span>
              <div className="flex items-center gap-1 p-0.5 bg-slate-900 rounded-lg border border-slate-800">
                <button
                  type="button"
                  onClick={() => setSkeletonMode('centerline')}
                  className={`px-2 py-1 rounded text-xs ${
                    skeletonMode === 'centerline'
                      ? 'bg-emerald-600 text-white font-bold'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  1px Centerline
                </button>
                <button
                  type="button"
                  onClick={() => setSkeletonMode('overlay')}
                  className={`px-2 py-1 rounded text-xs ${
                    skeletonMode === 'overlay'
                      ? 'bg-emerald-600 text-white font-bold'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  Color Overlay
                </button>
              </div>
            </div>
          </div>

          {/* Action buttons */}
          <div className="flex items-center gap-3">
            <button
              type="button"
              disabled={isTuning || !grooveData}
              onClick={() => handleTune()}
              className="px-3.5 py-2 rounded-xl text-xs font-semibold text-slate-300 bg-slate-800 hover:bg-slate-700 border border-slate-700 transition flex items-center gap-1.5 disabled:opacity-50"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${isTuning ? 'animate-spin text-emerald-400' : ''}`} />
              <span>{isTuning ? 'Tuning (~30ms)...' : 'Re-Tune Active Crop'}</span>
            </button>

            <button
              type="button"
              disabled={isProcessingVideo}
              onClick={handleProcessVideo}
              className="px-5 py-2 rounded-xl text-xs font-bold text-slate-950 bg-gradient-to-r from-emerald-400 to-teal-400 hover:from-emerald-300 hover:to-teal-300 transition shadow-lg shadow-emerald-500/20 flex items-center gap-2 disabled:opacity-50"
            >
              <Play className={`w-4 h-4 fill-slate-950 ${isProcessingVideo ? 'animate-pulse' : ''}`} />
              <span>{isProcessingVideo ? 'Extracting from Video...' : 'Extract Grooves From Selected Video'}</span>
            </button>
          </div>
        </div>

        {errorMsg && (
          <div className="p-3 rounded-xl bg-rose-950/60 border border-rose-500/40 text-rose-300 text-xs flex items-center gap-2">
            <X className="w-4 h-4 text-rose-400 flex-shrink-0" />
            <span>{errorMsg}</span>
          </div>
        )}
      </div>

      {/* ========================================================= */}
      {/* 5-STAGE PROGRESSION VIEW (GRID & FOCUS MODES)             */}
      {/* ========================================================= */}
      <div className="flex flex-col gap-4">
        {/* Gallery Mode Switcher Bar */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <h3 className="text-sm font-bold text-white flex items-center gap-2">
              <Layers className="w-4 h-4 text-emerald-400" />
              <span>5-Stage Biometric Groove &amp; Skeleton Transformation</span>
            </h3>
            <span className="text-xs text-slate-400">
              (Native Crop → Lanczos Ridge → Groove Response → Binary Mask → 1px Centerline)
            </span>
          </div>

          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1 p-0.5 bg-slate-950 rounded-lg border border-slate-800 text-xs">
              <button
                type="button"
                onClick={() => setViewMode('grid')}
                className={`px-3 py-1 rounded font-medium transition ${
                  viewMode === 'grid'
                    ? 'bg-emerald-600 text-white font-bold'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                All 5 Stages Grid
              </button>
              <button
                type="button"
                onClick={() => setViewMode('focus')}
                className={`px-3 py-1 rounded font-medium transition ${
                  viewMode === 'focus'
                    ? 'bg-emerald-600 text-white font-bold'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                HD Focus Viewer
              </button>
            </div>

            {grooveData?.artifacts?.summary_panel && (
              <a
                href={grooveData.artifacts.summary_panel}
                download="00_groove_5stages_summary.jpg"
                className="px-3 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs text-slate-300 border border-slate-700 flex items-center gap-1.5 transition"
              >
                <Download className="w-3.5 h-3.5 text-emerald-400" />
                <span>Composite Summary</span>
              </a>
            )}
          </div>
        </div>

        {/* VIEW MODE 1: ALL 5 STAGES GRID */}
        {viewMode === 'grid' && (
          <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-4">
            {stages.map((stg) => (
              <div
                key={stg.id}
                className="glass-panel p-3 rounded-2xl border border-slate-800 flex flex-col gap-2.5 hover:border-emerald-500/40 transition group relative"
              >
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-white tracking-tight">{stg.shortTitle}</span>
                  <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded border ${stg.badgeColor}`}>
                    {stg.badge}
                  </span>
                </div>

                <div
                  className="aspect-square bg-slate-950 rounded-xl overflow-hidden border border-slate-800 relative cursor-pointer flex items-center justify-center p-1"
                  onClick={() => {
                    setLightboxStage(stg.id);
                    setLightboxOpen(true);
                  }}
                >
                  {stg.url ? (
                    <img
                      src={stg.url}
                      alt={stg.title}
                      className="w-full h-full object-cover rounded-lg group-hover:scale-105 transition duration-300"
                    />
                  ) : (
                    <div className="flex flex-col items-center justify-center text-slate-600 gap-1">
                      <ScanLine className="w-6 h-6 animate-pulse" />
                      <span className="text-[11px]">Processing...</span>
                    </div>
                  )}

                  <div className="absolute inset-0 bg-slate-950/40 opacity-0 group-hover:opacity-100 transition flex items-center justify-center">
                    <span className="px-2.5 py-1 rounded-lg bg-slate-900/90 text-white text-[11px] font-medium flex items-center gap-1 border border-slate-700 shadow-xl">
                      <Maximize2 className="w-3 h-3 text-emerald-400" />
                      Inspect HD
                    </span>
                  </div>
                </div>

                <p className="text-[11px] text-slate-400 leading-tight">
                  {stg.desc}
                </p>

                {stg.url && (
                  <a
                    href={stg.url}
                    download
                    className="text-[10px] text-slate-400 hover:text-emerald-300 flex items-center gap-1 font-mono pt-1 border-t border-slate-800/60"
                  >
                    <Download className="w-3 h-3 text-slate-400" />
                    Download stage image
                  </a>
                )}
              </div>
            ))}
          </div>
        )}

        {/* VIEW MODE 2: HD FOCUS VIEWER */}
        {viewMode === 'focus' && (
          <div className="glass-panel p-5 rounded-2xl border border-slate-800 flex flex-col gap-4">
            {/* Stage tabs */}
            <div className="flex flex-wrap items-center gap-2 border-b border-slate-800 pb-3">
              {stages.map((stg) => (
                <button
                  key={stg.id}
                  type="button"
                  onClick={() => setSelectedStage(stg.id)}
                  className={`px-3.5 py-1.5 rounded-xl text-xs font-semibold transition flex items-center gap-1.5 ${
                    selectedStage === stg.id
                      ? 'bg-emerald-600 text-white shadow-lg shadow-emerald-600/20'
                      : 'bg-slate-900 text-slate-400 hover:text-slate-200 border border-slate-800'
                  }`}
                >
                  <span>{stg.shortTitle}</span>
                </button>
              ))}
            </div>

            {/* Focused stage display */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-center">
              <div className="lg:col-span-2 aspect-[4/3] max-h-[500px] bg-slate-950 rounded-2xl overflow-hidden border border-slate-800 flex items-center justify-center p-2 relative group">
                {stages[selectedStage]?.url ? (
                  <img
                    src={stages[selectedStage].url}
                    alt={stages[selectedStage].title}
                    className="w-full h-full object-contain rounded-xl"
                  />
                ) : (
                  <span className="text-xs text-slate-500 font-mono">No stage image loaded</span>
                )}
                <button
                  type="button"
                  onClick={() => {
                    setLightboxStage(selectedStage);
                    setLightboxOpen(true);
                  }}
                  className="absolute bottom-4 right-4 px-3 py-1.5 rounded-xl bg-slate-900/90 text-white text-xs font-semibold flex items-center gap-1.5 border border-slate-700 shadow-xl opacity-0 group-hover:opacity-100 transition"
                >
                  <Maximize2 className="w-3.5 h-3.5 text-emerald-400" />
                  Full Screen Zoom
                </button>
              </div>

              {/* Stage metadata card */}
              <div className="flex flex-col gap-4 p-5 rounded-2xl bg-slate-950/70 border border-slate-800">
                <div>
                  <span className={`text-[11px] font-mono px-2 py-0.5 rounded border ${stages[selectedStage]?.badgeColor}`}>
                    {stages[selectedStage]?.badge}
                  </span>
                  <h4 className="text-base font-bold text-white mt-2">
                    {stages[selectedStage]?.title}
                  </h4>
                  <p className="text-xs text-slate-400 mt-1">
                    {stages[selectedStage]?.desc}
                  </p>
                </div>

                <div className="flex flex-col gap-2 pt-2 border-t border-slate-800 text-xs font-mono">
                  <div className="flex justify-between text-slate-400">
                    <span>Input Resolution:</span>
                    <span className="text-white font-bold">{grooveData?.resolution || '392x392 px'}</span>
                  </div>
                  <div className="flex justify-between text-slate-400">
                    <span>Active Method:</span>
                    <span className="text-emerald-400 font-bold">{method.toUpperCase()}</span>
                  </div>
                  <div className="flex justify-between text-slate-400">
                    <span>Scales Applied:</span>
                    <span className="text-teal-300 font-bold">[{scales.join(', ')}]</span>
                  </div>
                  <div className="flex justify-between text-slate-400">
                    <span>Threshold Mode:</span>
                    <span className="text-cyan-300 font-bold">{thresholdMethod.toUpperCase()}</span>
                  </div>
                  <div className="flex justify-between text-slate-400">
                    <span>Detected Groove Pixels:</span>
                    <span className="text-white font-bold">{grooveData?.groove_pixel_count?.toLocaleString()} px</span>
                  </div>
                  <div className="flex justify-between text-slate-400">
                    <span>1px Skeleton Pixels:</span>
                    <span className="text-emerald-300 font-bold">{grooveData?.skeleton_pixel_count?.toLocaleString()} px</span>
                  </div>
                </div>

                {stages[selectedStage]?.url && (
                  <a
                    href={stages[selectedStage].url}
                    download
                    className="mt-2 py-2 px-4 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white font-bold text-xs flex items-center justify-center gap-2 transition shadow-lg shadow-emerald-600/20"
                  >
                    <Download className="w-4 h-4" />
                    Download This Stage Image
                  </a>
                )}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* ========================================================= */}
      {/* LIGHTBOX MODAL FOR FULLSCREEN HD INSPECTION               */}
      {/* ========================================================= */}
      {lightboxOpen && (
        <div className="fixed inset-0 z-50 bg-slate-950/90 backdrop-blur-md flex flex-col p-4 items-center justify-center">
          <div className="w-full max-w-5xl flex items-center justify-between pb-3 text-white border-b border-slate-800">
            <div className="flex items-center gap-3">
              <span className="font-bold text-sm">{stages[lightboxStage]?.title}</span>
              <span className="text-xs text-slate-400 font-mono">Zoom: {lightboxZoom}%</span>
            </div>

            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => setLightboxZoom((z) => Math.max(50, z - 25))}
                className="p-1.5 rounded-lg bg-slate-800 text-slate-300 hover:bg-slate-700"
              >
                <ZoomOut className="w-4 h-4" />
              </button>
              <button
                type="button"
                onClick={() => setLightboxZoom((z) => Math.min(300, z + 25))}
                className="p-1.5 rounded-lg bg-slate-800 text-slate-300 hover:bg-slate-700"
              >
                <ZoomIn className="w-4 h-4" />
              </button>
              <button
                type="button"
                onClick={() => setLightboxOpen(false)}
                className="p-1.5 rounded-lg bg-rose-500/20 text-rose-300 border border-rose-500/30 hover:bg-rose-500/30"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          </div>

          <div className="flex-1 w-full max-w-5xl flex items-center justify-center overflow-auto p-4 relative">
            <button
              type="button"
              onClick={() => setLightboxStage((s) => (s > 0 ? s - 1 : stages.length - 1))}
              className="absolute left-2 p-2 rounded-full bg-slate-900/80 border border-slate-700 text-white hover:bg-slate-800 transition"
            >
              <ChevronLeft className="w-5 h-5" />
            </button>

            {stages[lightboxStage]?.url && (
              <img
                src={stages[lightboxStage].url}
                alt={stages[lightboxStage].title}
                style={{ transform: `scale(${lightboxZoom / 100})` }}
                className="max-h-[75vh] max-w-full object-contain rounded-xl shadow-2xl transition-transform duration-200"
              />
            )}

            <button
              type="button"
              onClick={() => setLightboxStage((s) => (s < stages.length - 1 ? s + 1 : 0))}
              className="absolute right-2 p-2 rounded-full bg-slate-900/80 border border-slate-700 text-white hover:bg-slate-800 transition"
            >
              <ChevronRight className="w-5 h-5" />
            </button>
          </div>

          <div className="flex items-center gap-2 pt-2 border-t border-slate-800">
            {stages.map((stg) => (
              <button
                key={stg.id}
                type="button"
                onClick={() => setLightboxStage(stg.id)}
                className={`px-3 py-1 rounded-lg text-xs font-semibold ${
                  lightboxStage === stg.id
                    ? 'bg-emerald-600 text-white font-bold'
                    : 'bg-slate-900 text-slate-400 hover:text-slate-200'
                }`}
              >
                {stg.shortTitle}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
