import React, { useState, useEffect, useCallback } from 'react';
import {
  SlidersHorizontal,
  Sparkles,
  Filter,
  ScanLine,
  Maximize2,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Sun,
  Moon,
  Download,
  Check,
  AlertCircle,
  RefreshCw,
  Eye,
  Split,
  Layers,
  Activity,
  Info,
  CheckCircle2,
  AlertTriangle,
  X,
  ChevronLeft,
  ChevronRight,
  Zap,
  Sliders
} from 'lucide-react';

// Preset kernel configurations for multi-scale groove line enhancement
const KERNEL_PRESETS = {
  fine: {
    id: 'fine',
    label: 'Fine [7, 11, 15]',
    shortLabel: 'Fine',
    scales: [7, 11, 15],
    desc: 'Sensitive to micro-creases & shallow surface groove boundaries',
  },
  standard: {
    id: 'standard',
    label: 'Standard [9, 15, 21]',
    shortLabel: 'Standard',
    scales: [9, 15, 21],
    desc: 'Balanced multi-scale extraction (recommended for cattle muzzles)',
  },
  broad: {
    id: 'broad',
    label: 'Broad [11, 19, 27]',
    shortLabel: 'Broad',
    scales: [11, 19, 27],
    desc: 'Highlights deep structural valley fissures and broad clefts',
  },
  custom: {
    id: 'custom',
    label: 'Custom Scales',
    shortLabel: 'Custom',
    scales: null,
    desc: 'Custom structuring element kernel diameters',
  },
};

// URL normalizer helper: converts local filesystem output paths to accessible web routes
const toWebUrl = (pathOrUrl) => {
  if (!pathOrUrl || typeof pathOrUrl !== 'string') return null;
  if (
    pathOrUrl.startsWith('http://') ||
    pathOrUrl.startsWith('https://') ||
    pathOrUrl.startsWith('data:') ||
    pathOrUrl.startsWith('blob:')
  ) {
    return pathOrUrl;
  }
  if (
    pathOrUrl.startsWith('/output/') ||
    pathOrUrl.startsWith('/api/') ||
    pathOrUrl.startsWith('/test_samples/')
  ) {
    return pathOrUrl;
  }
  const outIdx = pathOrUrl.indexOf('/output/');
  if (outIdx !== -1) return pathOrUrl.substring(outIdx);
  const outIdx2 = pathOrUrl.indexOf('output/');
  if (outIdx2 !== -1) return '/' + pathOrUrl.substring(outIdx2);
  if (!pathOrUrl.startsWith('/')) return '/' + pathOrUrl;
  return pathOrUrl;
};

export default function GrooveTuningSection({ compareResult }) {
  // -------------------------------------------------------------
  // Tuning Control Parameters State
  // -------------------------------------------------------------
  const [grooveMethod, setGrooveMethod] = useState('blackhat'); // 'blackhat' | 'frangi'
  const [grooveThreshold, setGrooveThreshold] = useState('otsu'); // 'otsu' | 'adaptive'
  const [groovePreset, setGroovePreset] = useState('standard'); // 'fine' | 'standard' | 'broad' | 'custom'
  const [customScales, setCustomScales] = useState('9, 15, 21');
  const [grooveMinSize, setGrooveMinSize] = useState(15); // 5 to 50 px, default 15
  const [skeletonDisplayMode, setSkeletonDisplayMode] = useState('pure'); // 'pure' (1px centerline) | 'overlay' (color overlay on muzzle)
  const [inputResolutionSource, setInputResolutionSource] = useState('crop'); // 'crop' (224x224 native) | 'enhanced' (448x448 super-res)

  // -------------------------------------------------------------
  // Tuning Execution & Results State
  // -------------------------------------------------------------
  const [isTuningA, setIsTuningA] = useState(false);
  const [isTuningB, setIsTuningB] = useState(false);
  const [tuningError, setTuningError] = useState(null);
  const [tunedGrooves, setTunedGrooves] = useState({
    video_a: null,
    video_b: null,
  });

  // -------------------------------------------------------------
  // Stage Navigation & Display Mode State
  // -------------------------------------------------------------
  const [activeStageA, setActiveStageA] = useState(1); // 1 to 5
  const [activeStageB, setActiveStageB] = useState(1); // 1 to 5
  const [syncStages, setSyncStages] = useState(true); // synchronize stage switches between A and B
  const [viewMode, setViewMode] = useState('focus'); // 'focus' (stage carousel/focus) | 'grid' (all 5 stages)
  const [panelCollapsed, setPanelCollapsed] = useState(false);

  // -------------------------------------------------------------
  // High-Definition Lightbox Modal State
  // -------------------------------------------------------------
  const [lightbox, setLightbox] = useState(null);
  // { videoKey: 'video_a'|'video_b', stageId: 1..5, zoom: 1, inverted: false }

  // Prevent stale results when switching videos or crops
  useEffect(() => {
    setTunedGrooves({ video_a: null, video_b: null });
    setTuningError(null);
  }, [compareResult?.video_a?.crop_path, compareResult?.video_b?.crop_path]);

  // Synchronize stage switches when enabled
  const handleStageChange = (videoKey, newStage) => {
    if (syncStages) {
      setActiveStageA(newStage);
      setActiveStageB(newStage);
    } else {
      if (videoKey === 'video_a') setActiveStageA(newStage);
      else setActiveStageB(newStage);
    }
  };

  // -------------------------------------------------------------
  // Core API Caller: POST /api/grooves/tune
  // -------------------------------------------------------------
  const tuneSingleVideo = async (videoKey) => {
    const videoData = compareResult?.[videoKey];
    if (!videoData) return;

    // Resolve input path based on source selection
    const selectedImagePath =
      inputResolutionSource === 'enhanced' && videoData.enhanced_path
        ? videoData.enhanced_path
        : videoData.crop_path || videoData.enhanced_path;

    if (!selectedImagePath) {
      throw new Error(`Crop image path not found for ${videoKey === 'video_a' ? 'Video A' : 'Video B'}`);
    }

    // Resolve scales array
    let activeScales = KERNEL_PRESETS[groovePreset]?.scales;
    if (groovePreset === 'custom') {
      activeScales = customScales
        .split(',')
        .map((s) => parseInt(s.trim(), 10))
        .filter((n) => !isNaN(n) && n > 0);
      if (!activeScales || activeScales.length === 0) {
        activeScales = [9, 15, 21];
      }
    }

    const payload = {
      image_path: selectedImagePath,
      method: grooveMethod,
      scales: activeScales,
      threshold_method: grooveThreshold,
      min_component_size: Number(grooveMinSize),
      skeleton_enabled: true,
    };

    const startTime = performance.now();
    const response = await fetch('/api/grooves/tune', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    const clientLatencyMs = Math.round(performance.now() - startTime);

    if (!response.ok) {
      let errorDetail = `HTTP ${response.status} (${response.statusText})`;
      try {
        const errorJson = await response.json();
        if (errorJson.detail) {
          errorDetail = typeof errorJson.detail === 'string' ? errorJson.detail : JSON.stringify(errorJson.detail);
        } else if (errorJson.message) {
          errorDetail = errorJson.message;
        }
      } catch (_) {
        // use fallback text
      }
      throw new Error(`${videoKey === 'video_a' ? 'Video A' : 'Video B'}: ${errorDetail}`);
    }

    const json = await response.json();
    const data = json.result || json.data || json;

    const parsedResult = {
      groove_response_url: toWebUrl(
        data.groove_response_url ||
        data.response_url ||
        data.groove_response_path ||
        data.response_path ||
        data.saved_paths?.groove_response ||
        data.saved_paths?.response
      ),
      groove_mask_url: toWebUrl(
        data.groove_mask_url ||
        data.mask_url ||
        data.groove_mask_path ||
        data.mask_path ||
        data.saved_paths?.groove_mask ||
        data.saved_paths?.mask
      ),
      groove_skeleton_url: toWebUrl(
        data.groove_skeleton_url ||
        data.skeleton_url ||
        data.groove_skeleton_path ||
        data.skeleton_path ||
        data.saved_paths?.groove_skeleton ||
        data.saved_paths?.skeleton
      ),
      groove_overlay_url: toWebUrl(
        data.groove_overlay_url ||
        data.overlay_url ||
        data.groove_overlay_path ||
        data.overlay_path ||
        data.saved_paths?.groove_overlay ||
        data.saved_paths?.overlay
      ),
      processing_time_ms: Math.round(
        data.processing_time_ms ??
        data.timing_ms ??
        data.duration_ms ??
        data.elapsed_ms ??
        clientLatencyMs
      ),
      resolution: Array.isArray(data.resolution)
        ? `${data.resolution[0]}x${data.resolution[1]}`
        : typeof data.resolution === 'string'
        ? data.resolution
        : data.width && data.height
        ? `${data.width}x${data.height}`
        : inputResolutionSource === 'enhanced'
        ? '448x448'
        : '224x224',
      groove_pixel_count:
        data.groove_pixel_count ??
        data.num_groove_pixels ??
        data.pixel_count ??
        data.groove_pixels ??
        0,
      active_params: data.active_params || {
        method: grooveMethod,
        scales: activeScales,
        threshold_method: grooveThreshold,
        min_component_size: Number(grooveMinSize),
      },
      timestamp: Date.now(),
    };

    setTunedGrooves((prev) => ({
      ...prev,
      [videoKey]: parsedResult,
    }));
  };

  const handleTune = async (target = 'both') => {
    if (!compareResult) return;
    setTuningError(null);

    const targets = target === 'both' ? ['video_a', 'video_b'] : [target];
    if (targets.includes('video_a')) setIsTuningA(true);
    if (targets.includes('video_b')) setIsTuningB(true);

    try {
      const promises = targets.map((key) => tuneSingleVideo(key));
      const settled = await Promise.allSettled(promises);
      const errors = settled
        .filter((r) => r.status === 'rejected')
        .map((r) => r.reason?.message || 'Tuning request failed');

      if (errors.length > 0) {
        setTuningError(errors.join(' | '));
      } else {
        // If current stage is Stage 1 or 2, automatically advance to Stage 3 (Groove Response) to showcase result
        if (activeStageA < 3 && targets.includes('video_a')) {
          setActiveStageA(3);
        }
        if (activeStageB < 3 && targets.includes('video_b')) {
          setActiveStageB(3);
        }
      }
    } catch (err) {
      setTuningError(err.message || 'An unexpected error occurred while communicating with /api/grooves/tune.');
    } finally {
      if (targets.includes('video_a')) setIsTuningA(false);
      if (targets.includes('video_b')) setIsTuningB(false);
    }
  };

  // Reset parameters to defaults
  const handleResetDefaults = () => {
    setGrooveMethod('blackhat');
    setGrooveThreshold('otsu');
    setGroovePreset('standard');
    setCustomScales('9, 15, 21');
    setGrooveMinSize(15);
    setSkeletonDisplayMode('pure');
    setInputResolutionSource('crop');
    setTuningError(null);
  };

  // -------------------------------------------------------------
  // Stage Data Builder for a Given Video ('video_a' or 'video_b')
  // -------------------------------------------------------------
  const buildStages = (videoKey) => {
    const videoData = compareResult?.[videoKey];
    if (!videoData) return [];

    const tuned = tunedGrooves[videoKey];

    // Priority: Tuned URL > Initial comparison URL > null
    const responseUrl =
      tuned?.groove_response_url ||
      toWebUrl(videoData.groove_response_url || videoData.groove_response_path);

    const maskUrl =
      tuned?.groove_mask_url ||
      toWebUrl(videoData.groove_mask_url || videoData.groove_mask_path);

    const pureSkeletonUrl =
      tuned?.groove_skeleton_url ||
      toWebUrl(videoData.groove_skeleton_url || videoData.groove_skeleton_path);

    const overlayUrl =
      tuned?.groove_overlay_url ||
      toWebUrl(videoData.groove_overlay_url || videoData.groove_overlay_path);

    const selectedSkeletonUrl =
      skeletonDisplayMode === 'overlay'
        ? overlayUrl || pureSkeletonUrl
        : pureSkeletonUrl || overlayUrl;

    return [
      {
        id: 1,
        title: 'Native Muzzle Crop',
        shortLabel: '1: Native Crop',
        tag: 'Stage 1',
        url: toWebUrl(videoData.crop_url),
        path: videoData.crop_path,
        badge: `${(videoData.confidence * 100).toFixed(1)}% Conf`,
        badgeColor: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
        subtitle: 'Mask R-CNN Detection Window',
        resolution: '224x224 (Padded)',
        metrics: [
          { label: 'Quality Score', val: `${videoData.quality?.composite ?? '--'}/100` },
          { label: 'Sharpness', val: videoData.quality?.sharpness ?? '--' },
          { label: 'Frame Sampled', val: `#${videoData.frame_idx}` },
        ],
        description: 'Normalized 224x224 biometric ROI extracted around bovine nostrils & dermal muzzle pad.',
        isAvailable: Boolean(videoData.crop_url),
      },
      {
        id: 2,
        title: 'Grayscale Ridge Enhanced',
        shortLabel: '2: Ridge Enhanced',
        tag: 'Stage 2',
        url: toWebUrl(videoData.enhanced_url),
        path: videoData.enhanced_path,
        badge: 'Lanczos-4 2x',
        badgeColor: 'bg-teal-500/20 text-teal-300 border-teal-500/30',
        subtitle: 'Lanczos-4 Sinc Interpolation + CLAHE',
        resolution: '448x448 Super-Res',
        metrics: [
          { label: 'Interpolation', val: 'Lanczos-4 (8-pt Sinc)' },
          { label: 'Contrast', val: 'CLAHE Local Grid' },
          { label: 'Filter', val: 'Dermatoglyphic Beaded' },
        ],
        description: 'Super-resolved 448x448 grayscale print enhancing minute ridge beads and groove valleys.',
        isAvailable: Boolean(videoData.enhanced_url),
      },
      {
        id: 3,
        title: 'Groove Response Map',
        shortLabel: '3: Groove Response',
        tag: 'Stage 3',
        url: responseUrl,
        badge: tuned ? `${tuned.processing_time_ms} ms` : 'Valley Response',
        badgeColor: 'bg-purple-500/20 text-purple-300 border-purple-500/30',
        subtitle: `${(tuned?.active_params?.method || grooveMethod).toUpperCase()} Multi-Scale Filter`,
        resolution: tuned?.resolution || (inputResolutionSource === 'enhanced' ? '448x448' : '224x224'),
        metrics: [
          {
            label: 'Method',
            val:
              (tuned?.active_params?.method || grooveMethod) === 'blackhat'
                ? 'Morphological Black-Hat'
                : 'Frangi Hessian',
          },
          {
            label: 'Scales',
            val: `[${(
              tuned?.active_params?.scales ||
              (KERNEL_PRESETS[groovePreset]?.scales || [9, 15, 21])
            ).join(', ')}]`,
          },
          {
            label: 'Latency',
            val: tuned?.processing_time_ms ? `${tuned.processing_time_ms} ms` : 'Pending tuning',
          },
        ],
        description: 'Differential dark valley operator isolating low-luminance groove paths from raised beaded ridges.',
        isAvailable: Boolean(responseUrl),
        canTune: true,
      },
      {
        id: 4,
        title: 'Binary Groove Mask',
        shortLabel: '4: Binary Mask',
        tag: 'Stage 4',
        url: maskUrl,
        badge: tuned?.groove_pixel_count ? `${tuned.groove_pixel_count.toLocaleString()} px` : 'Grooves Mask',
        badgeColor: 'bg-amber-500/20 text-amber-300 border-amber-500/30',
        subtitle: `${(tuned?.active_params?.threshold_method || grooveThreshold).toUpperCase()} Threshold + Speckle Filter`,
        resolution: tuned?.resolution || (inputResolutionSource === 'enhanced' ? '448x448' : '224x224'),
        metrics: [
          {
            label: 'Threshold',
            val:
              (tuned?.active_params?.threshold_method || grooveThreshold) === 'otsu'
                ? 'Adaptive Otsu'
                : 'Gaussian Adaptive',
          },
          {
            label: 'Min Component',
            val: `${tuned?.active_params?.min_component_size ?? grooveMinSize} px`,
          },
          {
            label: 'Groove Pixels',
            val: tuned?.groove_pixel_count ? `${tuned.groove_pixel_count.toLocaleString()} px` : 'Pending tuning',
          },
        ],
        description: 'Thresholded binary mask (white = grooves, black = skin) with noise components pruned.',
        isAvailable: Boolean(maskUrl),
        canTune: true,
      },
      {
        id: 5,
        title: skeletonDisplayMode === 'overlay' ? 'Color Overlay on Muzzle' : 'Thin Skeletonized Network',
        shortLabel: '5: Thin Skeleton',
        tag: 'Stage 5',
        url: selectedSkeletonUrl,
        badge: skeletonDisplayMode === 'overlay' ? 'Green Overlay' : '1px Skeleton',
        badgeColor: 'bg-cyan-500/20 text-cyan-300 border-cyan-500/30',
        subtitle:
          skeletonDisplayMode === 'overlay'
            ? 'Topological Centerlines on Muzzle ROI'
            : '1-Pixel Wide Medial Axis Centerline Graph',
        resolution: tuned?.resolution || (inputResolutionSource === 'enhanced' ? '448x448' : '224x224'),
        metrics: [
          { label: 'Topology', val: '1px Medial Axis' },
          { label: 'Display Mode', val: skeletonDisplayMode === 'overlay' ? 'Color Overlay' : 'Centerline Only' },
          { label: 'Minutiae', val: 'Bifurcations Preserved' },
        ],
        description:
          skeletonDisplayMode === 'overlay'
            ? 'Skeletonized green groove trajectories superimposed directly over the bovine muzzle crop.'
            : 'Thinned 1-pixel wide skeleton network preserving topological branching points and minutiae.',
        isAvailable: Boolean(selectedSkeletonUrl),
        canTune: true,
      },
    ];
  };

  const stagesA = buildStages('video_a');
  const stagesB = buildStages('video_b');

  const currentStageDataA = stagesA.find((s) => s.id === activeStageA) || stagesA[0];
  const currentStageDataB = stagesB.find((s) => s.id === activeStageB) || stagesB[0];

  // -------------------------------------------------------------
  // Lightbox Keyboard Navigation Handler
  // -------------------------------------------------------------
  const handleLightboxKeyDown = useCallback(
    (e) => {
      if (!lightbox) return;
      if (e.key === 'Escape') {
        setLightbox(null);
      } else if (e.key === 'ArrowLeft') {
        setLightbox((prev) => ({
          ...prev,
          stageId: prev.stageId > 1 ? prev.stageId - 1 : 5,
        }));
      } else if (e.key === 'ArrowRight') {
        setLightbox((prev) => ({
          ...prev,
          stageId: prev.stageId < 5 ? prev.stageId + 1 : 1,
        }));
      }
    },
    [lightbox]
  );

  useEffect(() => {
    window.addEventListener('keydown', handleLightboxKeyDown);
    return () => window.removeEventListener('keydown', handleLightboxKeyDown);
  }, [handleLightboxKeyDown]);

  if (!compareResult) return null;

  return (
    <div className="flex flex-col gap-6">
      {/* ========================================================= */}
      {/* SECTION HEADER & CONTROL PANEL                            */}
      {/* ========================================================= */}
      <div className="glass-panel p-5 rounded-2xl border border-slate-800 flex flex-col gap-4 shadow-xl">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-3">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-emerald-600 to-teal-500 text-white flex items-center justify-center shadow-lg shadow-emerald-500/10">
              <SlidersHorizontal className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="text-base font-bold text-white tracking-tight">
                  Biometric Groove Extraction &amp; Filter Tuning Engine
                </h3>
                <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                  Interactive CV Pipeline
                </span>
              </div>
              <p className="text-xs text-slate-400">
                Tune multi-scale morphological response, adaptive binarization, and centerline skeletonization on muzzle crops in real-time.
              </p>
            </div>
          </div>

          {/* Quick Action Buttons */}
          <div className="flex items-center gap-2">
            <button
              onClick={() => handleTune('both')}
              disabled={isTuningA || isTuningB}
              className="flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-bold text-white bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 active:scale-95 transition-all shadow-md shadow-emerald-900/30 disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer"
            >
              {isTuningA || isTuningB ? (
                <RefreshCw className="w-4 h-4 animate-spin text-white" />
              ) : (
                <Sparkles className="w-4 h-4 text-emerald-200" />
              )}
              <span>Tune Both Videos (A &amp; B)</span>
            </button>

            <button
              onClick={handleResetDefaults}
              title="Reset parameters to recommended defaults"
              className="p-2 rounded-xl text-slate-400 hover:text-slate-200 bg-slate-800/80 hover:bg-slate-700 transition-all border border-slate-700/60 cursor-pointer"
            >
              <RotateCcw className="w-4 h-4" />
            </button>

            <button
              onClick={() => setPanelCollapsed(!panelCollapsed)}
              className="px-3 py-2 rounded-xl text-xs font-semibold text-slate-300 bg-slate-800/80 hover:bg-slate-700 transition-all border border-slate-700/60 cursor-pointer"
            >
              {panelCollapsed ? 'Expand Controls' : 'Collapse'}
            </button>
          </div>
        </div>

        {/* Collapsible Controls Body */}
        {!panelCollapsed && (
          <div className="flex flex-col gap-5 pt-1">
            {/* Top Grid of Parameters */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {/* Parameter 1: Enhancement Method */}
              <div className="flex flex-col gap-2 p-3 rounded-xl bg-slate-900/60 border border-slate-800/90">
                <div className="flex items-center justify-between text-xs font-semibold text-slate-300">
                  <div className="flex items-center gap-1.5">
                    <Filter className="w-3.5 h-3.5 text-purple-400" />
                    <span>Enhancement Method</span>
                  </div>
                  <span className="text-[10px] text-purple-400 font-mono">Stage 3</span>
                </div>
                <div className="grid grid-cols-2 gap-1.5 p-1 rounded-lg bg-slate-950 border border-slate-800">
                  <button
                    type="button"
                    onClick={() => setGrooveMethod('blackhat')}
                    className={`py-1.5 px-2 rounded text-xs font-medium transition-all cursor-pointer ${
                      grooveMethod === 'blackhat'
                        ? 'bg-purple-600 text-white font-bold shadow'
                        : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
                    }`}
                  >
                    Black-Hat
                  </button>
                  <button
                    type="button"
                    onClick={() => setGrooveMethod('frangi')}
                    className={`py-1.5 px-2 rounded text-xs font-medium transition-all cursor-pointer ${
                      grooveMethod === 'frangi'
                        ? 'bg-purple-600 text-white font-bold shadow'
                        : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
                    }`}
                  >
                    Frangi Hessian
                  </button>
                </div>
                <p className="text-[11px] text-slate-400 leading-tight">
                  {grooveMethod === 'blackhat'
                    ? 'Morphological closing difference isolating dark furrow valleys.'
                    : 'Hessian matrix multi-scale eigenvalues for vesselness curves.'}
                </p>
              </div>

              {/* Parameter 2: Thresholding Method */}
              <div className="flex flex-col gap-2 p-3 rounded-xl bg-slate-900/60 border border-slate-800/90">
                <div className="flex items-center justify-between text-xs font-semibold text-slate-300">
                  <div className="flex items-center gap-1.5">
                    <ScanLine className="w-3.5 h-3.5 text-amber-400" />
                    <span>Thresholding Method</span>
                  </div>
                  <span className="text-[10px] text-amber-400 font-mono">Stage 4</span>
                </div>
                <div className="grid grid-cols-2 gap-1.5 p-1 rounded-lg bg-slate-950 border border-slate-800">
                  <button
                    type="button"
                    onClick={() => setGrooveThreshold('otsu')}
                    className={`py-1.5 px-2 rounded text-xs font-medium transition-all cursor-pointer ${
                      grooveThreshold === 'otsu'
                        ? 'bg-amber-600 text-white font-bold shadow'
                        : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
                    }`}
                  >
                    Adaptive Otsu
                  </button>
                  <button
                    type="button"
                    onClick={() => setGrooveThreshold('adaptive')}
                    className={`py-1.5 px-2 rounded text-xs font-medium transition-all cursor-pointer ${
                      grooveThreshold === 'adaptive'
                        ? 'bg-amber-600 text-white font-bold shadow'
                        : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
                    }`}
                  >
                    Gaussian Adapt
                  </button>
                </div>
                <p className="text-[11px] text-slate-400 leading-tight">
                  {grooveThreshold === 'otsu'
                    ? 'Optimal bimodal threshold separating dark groove floor from bead.'
                    : 'Local neighborhood window compensating for lighting gradients.'}
                </p>
              </div>

              {/* Parameter 3: Multi-Scale Kernels */}
              <div className="flex flex-col gap-2 p-3 rounded-xl bg-slate-900/60 border border-slate-800/90">
                <div className="flex items-center justify-between text-xs font-semibold text-slate-300">
                  <div className="flex items-center gap-1.5">
                    <Layers className="w-3.5 h-3.5 text-emerald-400" />
                    <span>Multi-Scale Kernels</span>
                  </div>
                  <span className="text-[10px] text-emerald-400 font-mono">Filter Span</span>
                </div>
                <div className="grid grid-cols-4 gap-1 p-1 rounded-lg bg-slate-950 border border-slate-800">
                  {['fine', 'standard', 'broad', 'custom'].map((presetKey) => (
                    <button
                      key={presetKey}
                      type="button"
                      onClick={() => setGroovePreset(presetKey)}
                      className={`py-1 px-1 rounded text-[10px] font-medium transition-all cursor-pointer ${
                        groovePreset === presetKey
                          ? 'bg-emerald-600 text-white font-bold shadow'
                          : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
                      }`}
                    >
                      {KERNEL_PRESETS[presetKey].shortLabel}
                    </button>
                  ))}
                </div>
                {groovePreset === 'custom' ? (
                  <input
                    type="text"
                    value={customScales}
                    onChange={(e) => setCustomScales(e.target.value)}
                    placeholder="e.g. 7, 13, 21"
                    className="w-full bg-slate-950 border border-slate-700 rounded px-2 py-1 text-xs text-slate-200 font-mono focus:border-emerald-500 focus:outline-none"
                  />
                ) : (
                  <p className="text-[11px] text-slate-400 leading-tight">
                    {KERNEL_PRESETS[groovePreset]?.desc || 'Custom multi-scale structuring element diameters.'}
                  </p>
                )}
              </div>

              {/* Parameter 4: Noise Removal Sensitivity */}
              <div className="flex flex-col gap-2 p-3 rounded-xl bg-slate-900/60 border border-slate-800/90">
                <div className="flex items-center justify-between text-xs font-semibold text-slate-300">
                  <div className="flex items-center gap-1.5">
                    <Activity className="w-3.5 h-3.5 text-teal-400" />
                    <span>Noise Sensitivity</span>
                  </div>
                  <span className="px-1.5 py-0.5 rounded text-[10px] font-mono bg-teal-500/20 text-teal-300 border border-teal-500/30">
                    min {grooveMinSize} px
                  </span>
                </div>
                <div className="flex items-center gap-2 pt-1">
                  <input
                    type="range"
                    min="5"
                    max="50"
                    step="1"
                    value={grooveMinSize}
                    onChange={(e) => setGrooveMinSize(parseInt(e.target.value, 10))}
                    className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-teal-400"
                  />
                </div>
                <p className="text-[11px] text-slate-400 leading-tight">
                  Prunes disconnected speckles &lt; {grooveMinSize} px while retaining connected ridge lines.
                </p>
              </div>
            </div>

            {/* Bottom Row: Skeleton Display Mode, Source Resolution, & Individual Action Buttons */}
            <div className="flex flex-wrap items-center justify-between gap-4 p-3 rounded-xl bg-slate-900/40 border border-slate-800/80">
              {/* Skeleton Display Toggle */}
              <div className="flex items-center gap-3">
                <span className="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
                  <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
                  Skeleton Mode:
                </span>
                <div className="inline-flex rounded-lg bg-slate-950 p-1 border border-slate-800">
                  <button
                    type="button"
                    onClick={() => setSkeletonDisplayMode('pure')}
                    className={`px-3 py-1 rounded text-xs font-medium transition-all cursor-pointer ${
                      skeletonDisplayMode === 'pure'
                        ? 'bg-cyan-600 text-white font-bold shadow'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    Pure 1px Centerline
                  </button>
                  <button
                    type="button"
                    onClick={() => setSkeletonDisplayMode('overlay')}
                    className={`px-3 py-1 rounded text-xs font-medium transition-all cursor-pointer ${
                      skeletonDisplayMode === 'overlay'
                        ? 'bg-cyan-600 text-white font-bold shadow'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    Color Overlay on Muzzle
                  </button>
                </div>
              </div>

              {/* Input Source Toggle */}
              <div className="flex items-center gap-3">
                <span className="text-xs font-semibold text-slate-300">Input Source:</span>
                <div className="inline-flex rounded-lg bg-slate-950 p-1 border border-slate-800">
                  <button
                    type="button"
                    onClick={() => setInputResolutionSource('crop')}
                    className={`px-3 py-1 rounded text-xs font-medium transition-all cursor-pointer ${
                      inputResolutionSource === 'crop'
                        ? 'bg-slate-700 text-white font-bold'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    Native Crop (224x224)
                  </button>
                  <button
                    type="button"
                    onClick={() => setInputResolutionSource('enhanced')}
                    className={`px-3 py-1 rounded text-xs font-medium transition-all cursor-pointer ${
                      inputResolutionSource === 'enhanced'
                        ? 'bg-slate-700 text-white font-bold'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    2x Super-Res (448x448)
                  </button>
                </div>
              </div>

              {/* Individual Tune Buttons */}
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => handleTune('video_a')}
                  disabled={isTuningA}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold text-emerald-300 bg-emerald-500/10 hover:bg-emerald-500/20 border border-emerald-500/30 transition-all cursor-pointer disabled:opacity-50"
                >
                  {isTuningA ? (
                    <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                  ) : (
                    <Zap className="w-3.5 h-3.5 text-emerald-400" />
                  )}
                  <span>Tune Video A</span>
                </button>

                <button
                  type="button"
                  onClick={() => handleTune('video_b')}
                  disabled={isTuningB}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold text-teal-300 bg-teal-500/10 hover:bg-teal-500/20 border border-teal-500/30 transition-all cursor-pointer disabled:opacity-50"
                >
                  {isTuningB ? (
                    <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                  ) : (
                    <Zap className="w-3.5 h-3.5 text-teal-400" />
                  )}
                  <span>Tune Video B</span>
                </button>
              </div>
            </div>

            {/* Diagnostics & Active Parameter Summary Bar */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs bg-slate-950/70 p-3 rounded-xl border border-slate-800">
              {/* Video A Diagnostics */}
              <div className="flex flex-wrap items-center justify-between gap-2 border-r-0 md:border-r border-slate-800 pr-0 md:pr-3">
                <div className="flex items-center gap-2">
                  <div className="w-5 h-5 rounded bg-emerald-500/20 text-emerald-400 flex items-center justify-center font-bold text-[10px]">
                    A
                  </div>
                  <span className="font-semibold text-slate-200">Video A Grooves:</span>
                  {tunedGrooves.video_a ? (
                    <span className="text-emerald-400 font-mono text-[11px] flex items-center gap-1">
                      <CheckCircle2 className="w-3.5 h-3.5" /> Tuned in {tunedGrooves.video_a.processing_time_ms}ms
                    </span>
                  ) : (
                    <span className="text-slate-400 text-[11px]">Default / Pending tune</span>
                  )}
                </div>

                <div className="flex items-center gap-2 font-mono text-[11px] text-slate-400">
                  <span className="bg-slate-900 px-2 py-0.5 rounded border border-slate-800">
                    Res: {tunedGrooves.video_a?.resolution || (inputResolutionSource === 'enhanced' ? '448x448' : '224x224')}
                  </span>
                  <span className="bg-slate-900 px-2 py-0.5 rounded border border-slate-800 text-teal-400">
                    {tunedGrooves.video_a?.groove_pixel_count
                      ? `${tunedGrooves.video_a.groove_pixel_count.toLocaleString()} px`
                      : '-- px'}
                  </span>
                </div>
              </div>

              {/* Video B Diagnostics */}
              <div className="flex flex-wrap items-center justify-between gap-2 pl-0 md:pl-2">
                <div className="flex items-center gap-2">
                  <div className="w-5 h-5 rounded bg-teal-500/20 text-teal-400 flex items-center justify-center font-bold text-[10px]">
                    B
                  </div>
                  <span className="font-semibold text-slate-200">Video B Grooves:</span>
                  {tunedGrooves.video_b ? (
                    <span className="text-teal-400 font-mono text-[11px] flex items-center gap-1">
                      <CheckCircle2 className="w-3.5 h-3.5" /> Tuned in {tunedGrooves.video_b.processing_time_ms}ms
                    </span>
                  ) : (
                    <span className="text-slate-400 text-[11px]">Default / Pending tune</span>
                  )}
                </div>

                <div className="flex items-center gap-2 font-mono text-[11px] text-slate-400">
                  <span className="bg-slate-900 px-2 py-0.5 rounded border border-slate-800">
                    Res: {tunedGrooves.video_b?.resolution || (inputResolutionSource === 'enhanced' ? '448x448' : '224x224')}
                  </span>
                  <span className="bg-slate-900 px-2 py-0.5 rounded border border-slate-800 text-teal-400">
                    {tunedGrooves.video_b?.groove_pixel_count
                      ? `${tunedGrooves.video_b.groove_pixel_count.toLocaleString()} px`
                      : '-- px'}
                  </span>
                </div>
              </div>
            </div>

            {/* Error Alert Banner */}
            {tuningError && (
              <div className="flex items-center justify-between p-3 rounded-xl bg-rose-950/40 border border-rose-800/80 text-rose-300 text-xs">
                <div className="flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0" />
                  <span>{tuningError}</span>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => handleTune('both')}
                    className="px-2.5 py-1 rounded bg-rose-800/60 hover:bg-rose-700 text-rose-100 font-semibold cursor-pointer"
                  >
                    Retry
                  </button>
                  <button
                    type="button"
                    onClick={() => setTuningError(null)}
                    className="p-1 rounded text-rose-400 hover:text-rose-200 cursor-pointer"
                  >
                    <X className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* ========================================================= */}
      {/* 5-STAGE PREVIEWS CONTROLS (SYNC & VIEW MODE TOGGLE)       */}
      {/* ========================================================= */}
      <div className="flex flex-wrap items-center justify-between gap-3 px-1">
        <div className="flex items-center gap-2">
          <h4 className="text-sm font-bold text-white flex items-center gap-2">
            <Layers className="w-4 h-4 text-emerald-400" />
            <span>Full 5-Stage Biometric Progression</span>
          </h4>
          <span className="text-xs text-slate-400 font-mono">
            [1: Native Crop &rarr; 2: CLAHE &rarr; 3: Groove Response &rarr; 4: Binary Mask &rarr; 5: 1px Skeleton]
          </span>
        </div>

        <div className="flex items-center gap-2">
          {/* Synchronize Stage Selection Toggle */}
          <button
            type="button"
            onClick={() => setSyncStages(!syncStages)}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold transition-all border cursor-pointer ${
              syncStages
                ? 'bg-emerald-500/15 text-emerald-300 border-emerald-500/40 shadow-sm'
                : 'bg-slate-800/70 text-slate-400 border-slate-700 hover:text-slate-200'
            }`}
          >
            <Split className="w-3.5 h-3.5" />
            <span>Sync Stages (A &amp; B)</span>
            <span
              className={`w-2 h-2 rounded-full ${
                syncStages ? 'bg-emerald-400' : 'bg-slate-500'
              }`}
            />
          </button>

          {/* View Mode: Focus Carousel vs All 5 Stages Grid */}
          <div className="inline-flex rounded-xl bg-slate-900 p-0.5 border border-slate-800">
            <button
              type="button"
              onClick={() => setViewMode('focus')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
                viewMode === 'focus'
                  ? 'bg-slate-800 text-white shadow'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Eye className="w-3.5 h-3.5 text-emerald-400" />
              <span>Focus View</span>
            </button>
            <button
              type="button"
              onClick={() => setViewMode('grid')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
                viewMode === 'grid'
                  ? 'bg-slate-800 text-white shadow'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Layers className="w-3.5 h-3.5 text-teal-400" />
              <span>All 5 Stages Grid</span>
            </button>
          </div>
        </div>
      </div>

      {/* ========================================================= */}
      {/* 5-STAGE DISPLAY FOR VIDEO A AND VIDEO B                   */}
      {/* ========================================================= */}
      {viewMode === 'focus' ? (
        /* FOCUS MODE: Large HD Inspector with Top Tab Strip & Bottom Thumbnails */
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* ---------------- Video A Card ---------------- */}
          <div className="glass-panel p-5 rounded-2xl flex flex-col gap-4 border border-slate-800 relative shadow-xl">
            {/* Header */}
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div className="flex items-center gap-2">
                <div className="w-7 h-7 rounded-lg bg-emerald-500/20 text-emerald-400 flex items-center justify-center font-bold text-xs border border-emerald-500/30">
                  A
                </div>
                <div>
                  <h4 className="text-sm font-bold text-white flex items-center gap-1.5">
                    <span>Video A: {compareResult.video_a.name}</span>
                  </h4>
                  <p className="text-[11px] text-slate-400 font-mono">
                    Frame #{compareResult.video_a.frame_idx} • Quality:{' '}
                    {compareResult.video_a.quality?.composite ?? '--'}/100
                  </p>
                </div>
              </div>

              <span className={`px-2.5 py-1 rounded-full text-xs font-mono font-semibold border ${currentStageDataA.badgeColor}`}>
                {currentStageDataA.badge}
              </span>
            </div>

            {/* Stage Selector Tab Strip */}
            <div className="grid grid-cols-5 gap-1 p-1 rounded-xl bg-slate-950 border border-slate-800">
              {stagesA.map((stg) => {
                const isActive = stg.id === activeStageA;
                return (
                  <button
                    key={stg.id}
                    type="button"
                    onClick={() => handleStageChange('video_a', stg.id)}
                    className={`flex flex-col items-center justify-center py-2 px-1 rounded-lg text-center transition-all cursor-pointer ${
                      isActive
                        ? 'bg-emerald-600 text-white font-bold shadow-md shadow-emerald-950/60'
                        : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/80'
                    }`}
                  >
                    <span className="text-[10px] uppercase tracking-wider opacity-80">{stg.tag}</span>
                    <span className="text-[11px] font-semibold truncate max-w-full">
                      {stg.shortLabel.split(':')[1]?.trim() || stg.shortLabel}
                    </span>
                    <span
                      className={`w-1.5 h-1.5 rounded-full mt-1 ${
                        stg.isAvailable ? 'bg-emerald-400' : 'bg-slate-600'
                      }`}
                    />
                  </button>
                );
              })}
            </div>

            {/* Main Stage Display Area */}
            <div className="aspect-square bg-slate-950 rounded-2xl overflow-hidden border border-slate-800 relative flex items-center justify-center group">
              {currentStageDataA.isAvailable ? (
                <>
                  <img
                    src={currentStageDataA.url}
                    alt={`Video A ${currentStageDataA.title}`}
                    className={`w-full h-full object-contain transition-transform duration-300 group-hover:scale-105 ${
                      currentStageDataA.id === 2 ? 'filter contrast-125' : ''
                    }`}
                  />

                  {/* Overlay Controls */}
                  <div className="absolute top-3 right-3 flex items-center gap-1.5 opacity-90 group-hover:opacity-100 transition-opacity">
                    <button
                      type="button"
                      onClick={() =>
                        setLightbox({
                          videoKey: 'video_a',
                          stageId: currentStageDataA.id,
                          zoom: 1,
                          inverted: false,
                        })
                      }
                      title="Open High-Definition Zoom Inspector"
                      className="p-2 rounded-xl bg-slate-900/80 hover:bg-slate-800 text-slate-200 backdrop-blur-md border border-slate-700/80 transition-all cursor-pointer shadow-lg"
                    >
                      <Maximize2 className="w-4 h-4 text-emerald-400" />
                    </button>
                    {currentStageDataA.url && (
                      <a
                        href={currentStageDataA.url}
                        download={`video_A_stage_${currentStageDataA.id}_${currentStageDataA.title.replace(/\s+/g, '_')}.png`}
                        title="Download Stage Image"
                        className="p-2 rounded-xl bg-slate-900/80 hover:bg-slate-800 text-slate-200 backdrop-blur-md border border-slate-700/80 transition-all cursor-pointer shadow-lg"
                      >
                        <Download className="w-4 h-4" />
                      </a>
                    )}
                  </div>

                  {/* Resolution & Stage Info Overlay */}
                  <div className="absolute bottom-3 left-3 right-3 flex items-center justify-between pointer-events-none">
                    <span className="px-2.5 py-1 rounded-lg bg-slate-900/85 text-[11px] font-mono text-emerald-300 backdrop-blur-md border border-slate-800">
                      {currentStageDataA.resolution}
                    </span>
                    <span className="px-2.5 py-1 rounded-lg bg-slate-900/85 text-[11px] font-mono text-slate-300 backdrop-blur-md border border-slate-800">
                      {currentStageDataA.subtitle}
                    </span>
                  </div>
                </>
              ) : (
                /* Pending Groove Tuning Placeholder */
                <div className="flex flex-col items-center justify-center p-6 text-center gap-3">
                  <div className="w-12 h-12 rounded-2xl bg-slate-900 border border-slate-800 flex items-center justify-center text-slate-400">
                    <Filter className="w-6 h-6 text-purple-400" />
                  </div>
                  <div>
                    <h5 className="text-sm font-bold text-white">{currentStageDataA.title} Pending</h5>
                    <p className="text-xs text-slate-400 max-w-xs mt-1">
                      Groove response and skeleton networks have not been extracted for Video A yet.
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={() => handleTune('video_a')}
                    disabled={isTuningA}
                    className="flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-bold text-white bg-emerald-600 hover:bg-emerald-500 transition-all cursor-pointer shadow-md disabled:opacity-50"
                  >
                    {isTuningA ? (
                      <RefreshCw className="w-4 h-4 animate-spin" />
                    ) : (
                      <Sparkles className="w-4 h-4" />
                    )}
                    <span>Tune Video A Grooves</span>
                  </button>
                </div>
              )}
            </div>

            {/* Stage Metrics Info Bar */}
            <div className="grid grid-cols-3 gap-2 p-2.5 rounded-xl bg-slate-900/60 border border-slate-800 text-xs font-mono">
              {currentStageDataA.metrics.map((m, idx) => (
                <div key={idx} className="flex flex-col">
                  <span className="text-[10px] text-slate-400">{m.label}</span>
                  <span className="font-semibold text-slate-200 truncate">{m.val}</span>
                </div>
              ))}
            </div>

            {/* Explanation Note */}
            <p className="text-xs text-slate-400 leading-relaxed bg-slate-950/40 p-2.5 rounded-xl border border-slate-800/80">
              <span className="text-emerald-400 font-semibold">{currentStageDataA.title}: </span>
              {currentStageDataA.description}
            </p>

            {/* Clickable 5-Stage Thumbnail Strip */}
            <div className="flex flex-col gap-1.5 pt-1">
              <span className="text-[10px] uppercase font-bold text-slate-400 tracking-wider">
                Stage Navigation Strip (Click to Switch)
              </span>
              <div className="grid grid-cols-5 gap-2">
                {stagesA.map((stg) => {
                  const isActive = stg.id === activeStageA;
                  return (
                    <button
                      key={stg.id}
                      type="button"
                      onClick={() => handleStageChange('video_a', stg.id)}
                      className={`flex flex-col gap-1 p-1 rounded-xl transition-all border text-left cursor-pointer ${
                        isActive
                          ? 'border-emerald-500 bg-emerald-500/10 ring-2 ring-emerald-500/20'
                          : 'border-slate-800 bg-slate-950/60 hover:border-slate-700'
                      }`}
                    >
                      <div className="aspect-square bg-slate-900 rounded-lg overflow-hidden flex items-center justify-center">
                        {stg.isAvailable ? (
                          <img
                            src={stg.url}
                            alt={stg.title}
                            className="w-full h-full object-cover"
                          />
                        ) : (
                          <div className="text-[9px] text-slate-600 font-mono">Empty</div>
                        )}
                      </div>
                      <span className="text-[10px] font-semibold text-slate-300 truncate px-0.5">
                        {stg.tag}
                      </span>
                    </button>
                  );
                })}
              </div>
            </div>
          </div>

          {/* ---------------- Video B Card ---------------- */}
          <div className="glass-panel p-5 rounded-2xl flex flex-col gap-4 border border-slate-800 relative shadow-xl">
            {/* Header */}
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div className="flex items-center gap-2">
                <div className="w-7 h-7 rounded-lg bg-teal-500/20 text-teal-400 flex items-center justify-center font-bold text-xs border border-teal-500/30">
                  B
                </div>
                <div>
                  <h4 className="text-sm font-bold text-white flex items-center gap-1.5">
                    <span>Video B: {compareResult.video_b.name}</span>
                  </h4>
                  <p className="text-[11px] text-slate-400 font-mono">
                    Frame #{compareResult.video_b.frame_idx} • Quality:{' '}
                    {compareResult.video_b.quality?.composite ?? '--'}/100
                  </p>
                </div>
              </div>

              <span className={`px-2.5 py-1 rounded-full text-xs font-mono font-semibold border ${currentStageDataB.badgeColor}`}>
                {currentStageDataB.badge}
              </span>
            </div>

            {/* Stage Selector Tab Strip */}
            <div className="grid grid-cols-5 gap-1 p-1 rounded-xl bg-slate-950 border border-slate-800">
              {stagesB.map((stg) => {
                const isActive = stg.id === activeStageB;
                return (
                  <button
                    key={stg.id}
                    type="button"
                    onClick={() => handleStageChange('video_b', stg.id)}
                    className={`flex flex-col items-center justify-center py-2 px-1 rounded-lg text-center transition-all cursor-pointer ${
                      isActive
                        ? 'bg-teal-600 text-white font-bold shadow-md shadow-teal-950/60'
                        : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/80'
                    }`}
                  >
                    <span className="text-[10px] uppercase tracking-wider opacity-80">{stg.tag}</span>
                    <span className="text-[11px] font-semibold truncate max-w-full">
                      {stg.shortLabel.split(':')[1]?.trim() || stg.shortLabel}
                    </span>
                    <span
                      className={`w-1.5 h-1.5 rounded-full mt-1 ${
                        stg.isAvailable ? 'bg-teal-400' : 'bg-slate-600'
                      }`}
                    />
                  </button>
                );
              })}
            </div>

            {/* Main Stage Display Area */}
            <div className="aspect-square bg-slate-950 rounded-2xl overflow-hidden border border-slate-800 relative flex items-center justify-center group">
              {currentStageDataB.isAvailable ? (
                <>
                  <img
                    src={currentStageDataB.url}
                    alt={`Video B ${currentStageDataB.title}`}
                    className={`w-full h-full object-contain transition-transform duration-300 group-hover:scale-105 ${
                      currentStageDataB.id === 2 ? 'filter contrast-125' : ''
                    }`}
                  />

                  {/* Overlay Controls */}
                  <div className="absolute top-3 right-3 flex items-center gap-1.5 opacity-90 group-hover:opacity-100 transition-opacity">
                    <button
                      type="button"
                      onClick={() =>
                        setLightbox({
                          videoKey: 'video_b',
                          stageId: currentStageDataB.id,
                          zoom: 1,
                          inverted: false,
                        })
                      }
                      title="Open High-Definition Zoom Inspector"
                      className="p-2 rounded-xl bg-slate-900/80 hover:bg-slate-800 text-slate-200 backdrop-blur-md border border-slate-700/80 transition-all cursor-pointer shadow-lg"
                    >
                      <Maximize2 className="w-4 h-4 text-teal-400" />
                    </button>
                    {currentStageDataB.url && (
                      <a
                        href={currentStageDataB.url}
                        download={`video_B_stage_${currentStageDataB.id}_${currentStageDataB.title.replace(/\s+/g, '_')}.png`}
                        title="Download Stage Image"
                        className="p-2 rounded-xl bg-slate-900/80 hover:bg-slate-800 text-slate-200 backdrop-blur-md border border-slate-700/80 transition-all cursor-pointer shadow-lg"
                      >
                        <Download className="w-4 h-4" />
                      </a>
                    )}
                  </div>

                  {/* Resolution & Stage Info Overlay */}
                  <div className="absolute bottom-3 left-3 right-3 flex items-center justify-between pointer-events-none">
                    <span className="px-2.5 py-1 rounded-lg bg-slate-900/85 text-[11px] font-mono text-teal-300 backdrop-blur-md border border-slate-800">
                      {currentStageDataB.resolution}
                    </span>
                    <span className="px-2.5 py-1 rounded-lg bg-slate-900/85 text-[11px] font-mono text-slate-300 backdrop-blur-md border border-slate-800">
                      {currentStageDataB.subtitle}
                    </span>
                  </div>
                </>
              ) : (
                /* Pending Groove Tuning Placeholder */
                <div className="flex flex-col items-center justify-center p-6 text-center gap-3">
                  <div className="w-12 h-12 rounded-2xl bg-slate-900 border border-slate-800 flex items-center justify-center text-slate-400">
                    <Filter className="w-6 h-6 text-purple-400" />
                  </div>
                  <div>
                    <h5 className="text-sm font-bold text-white">{currentStageDataB.title} Pending</h5>
                    <p className="text-xs text-slate-400 max-w-xs mt-1">
                      Groove response and skeleton networks have not been extracted for Video B yet.
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={() => handleTune('video_b')}
                    disabled={isTuningB}
                    className="flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-bold text-white bg-teal-600 hover:bg-teal-500 transition-all cursor-pointer shadow-md disabled:opacity-50"
                  >
                    {isTuningB ? (
                      <RefreshCw className="w-4 h-4 animate-spin" />
                    ) : (
                      <Sparkles className="w-4 h-4" />
                    )}
                    <span>Tune Video B Grooves</span>
                  </button>
                </div>
              )}
            </div>

            {/* Stage Metrics Info Bar */}
            <div className="grid grid-cols-3 gap-2 p-2.5 rounded-xl bg-slate-900/60 border border-slate-800 text-xs font-mono">
              {currentStageDataB.metrics.map((m, idx) => (
                <div key={idx} className="flex flex-col">
                  <span className="text-[10px] text-slate-400">{m.label}</span>
                  <span className="font-semibold text-slate-200 truncate">{m.val}</span>
                </div>
              ))}
            </div>

            {/* Explanation Note */}
            <p className="text-xs text-slate-400 leading-relaxed bg-slate-950/40 p-2.5 rounded-xl border border-slate-800/80">
              <span className="text-teal-400 font-semibold">{currentStageDataB.title}: </span>
              {currentStageDataB.description}
            </p>

            {/* Clickable 5-Stage Thumbnail Strip */}
            <div className="flex flex-col gap-1.5 pt-1">
              <span className="text-[10px] uppercase font-bold text-slate-400 tracking-wider">
                Stage Navigation Strip (Click to Switch)
              </span>
              <div className="grid grid-cols-5 gap-2">
                {stagesB.map((stg) => {
                  const isActive = stg.id === activeStageB;
                  return (
                    <button
                      key={stg.id}
                      type="button"
                      onClick={() => handleStageChange('video_b', stg.id)}
                      className={`flex flex-col gap-1 p-1 rounded-xl transition-all border text-left cursor-pointer ${
                        isActive
                          ? 'border-teal-500 bg-teal-500/10 ring-2 ring-teal-500/20'
                          : 'border-slate-800 bg-slate-950/60 hover:border-slate-700'
                      }`}
                    >
                      <div className="aspect-square bg-slate-900 rounded-lg overflow-hidden flex items-center justify-center">
                        {stg.isAvailable ? (
                          <img
                            src={stg.url}
                            alt={stg.title}
                            className="w-full h-full object-cover"
                          />
                        ) : (
                          <div className="text-[9px] text-slate-600 font-mono">Empty</div>
                        )}
                      </div>
                      <span className="text-[10px] font-semibold text-slate-300 truncate px-0.5">
                        {stg.tag}
                      </span>
                    </button>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      ) : (
        /* GRID MODE: All 5 Stages Displayed Concurrently for Direct Visual Comparison */
        <div className="flex flex-col gap-6">
          {/* Row A: Video A 5-Stage Sequence */}
          <div className="glass-panel p-5 rounded-2xl border border-slate-800 flex flex-col gap-3 shadow-xl">
            <div className="flex items-center justify-between border-b border-slate-800 pb-2">
              <div className="flex items-center gap-2">
                <div className="w-6 h-6 rounded-md bg-emerald-500/20 text-emerald-400 flex items-center justify-center font-bold text-xs">
                  A
                </div>
                <h4 className="text-sm font-bold text-white">Video A: {compareResult.video_a.name}</h4>
                <span className="text-xs text-slate-400 font-mono">
                  (Frame #{compareResult.video_a.frame_idx})
                </span>
              </div>
              <button
                type="button"
                onClick={() => handleTune('video_a')}
                disabled={isTuningA}
                className="flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-bold text-emerald-300 bg-emerald-500/10 hover:bg-emerald-500/20 border border-emerald-500/30 transition-all cursor-pointer"
              >
                {isTuningA ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Sparkles className="w-3.5 h-3.5" />}
                <span>Re-Tune A</span>
              </button>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3">
              {stagesA.map((stg) => (
                <div
                  key={stg.id}
                  className="flex flex-col gap-2 p-2.5 rounded-xl bg-slate-950/70 border border-slate-800 hover:border-slate-700 transition-all"
                >
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-bold text-slate-300">{stg.tag}</span>
                    <span className="text-[10px] text-slate-400 font-mono">{stg.resolution}</span>
                  </div>

                  <div className="aspect-square bg-slate-900 rounded-lg overflow-hidden border border-slate-800/80 relative flex items-center justify-center group">
                    {stg.isAvailable ? (
                      <>
                        <img
                          src={stg.url}
                          alt={stg.title}
                          className="w-full h-full object-contain"
                        />
                        <button
                          type="button"
                          onClick={() =>
                            setLightbox({
                              videoKey: 'video_a',
                              stageId: stg.id,
                              zoom: 1,
                              inverted: false,
                            })
                          }
                          className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center text-white cursor-pointer"
                        >
                          <Maximize2 className="w-5 h-5 text-emerald-400 drop-shadow" />
                        </button>
                      </>
                    ) : (
                      <div className="text-[10px] text-slate-500 text-center p-2 font-mono">
                        Not Tuned
                      </div>
                    )}
                  </div>

                  <div className="text-[11px] font-semibold text-slate-300 truncate">{stg.title}</div>
                  <div className="text-[10px] text-slate-400 line-clamp-2 leading-tight">
                    {stg.subtitle}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Row B: Video B 5-Stage Sequence */}
          <div className="glass-panel p-5 rounded-2xl border border-slate-800 flex flex-col gap-3 shadow-xl">
            <div className="flex items-center justify-between border-b border-slate-800 pb-2">
              <div className="flex items-center gap-2">
                <div className="w-6 h-6 rounded-md bg-teal-500/20 text-teal-400 flex items-center justify-center font-bold text-xs">
                  B
                </div>
                <h4 className="text-sm font-bold text-white">Video B: {compareResult.video_b.name}</h4>
                <span className="text-xs text-slate-400 font-mono">
                  (Frame #{compareResult.video_b.frame_idx})
                </span>
              </div>
              <button
                type="button"
                onClick={() => handleTune('video_b')}
                disabled={isTuningB}
                className="flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-bold text-teal-300 bg-teal-500/10 hover:bg-teal-500/20 border border-teal-500/30 transition-all cursor-pointer"
              >
                {isTuningB ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Sparkles className="w-3.5 h-3.5" />}
                <span>Re-Tune B</span>
              </button>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3">
              {stagesB.map((stg) => (
                <div
                  key={stg.id}
                  className="flex flex-col gap-2 p-2.5 rounded-xl bg-slate-950/70 border border-slate-800 hover:border-slate-700 transition-all"
                >
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-bold text-slate-300">{stg.tag}</span>
                    <span className="text-[10px] text-slate-400 font-mono">{stg.resolution}</span>
                  </div>

                  <div className="aspect-square bg-slate-900 rounded-lg overflow-hidden border border-slate-800/80 relative flex items-center justify-center group">
                    {stg.isAvailable ? (
                      <>
                        <img
                          src={stg.url}
                          alt={stg.title}
                          className="w-full h-full object-contain"
                        />
                        <button
                          type="button"
                          onClick={() =>
                            setLightbox({
                              videoKey: 'video_b',
                              stageId: stg.id,
                              zoom: 1,
                              inverted: false,
                            })
                          }
                          className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center text-white cursor-pointer"
                        >
                          <Maximize2 className="w-5 h-5 text-teal-400 drop-shadow" />
                        </button>
                      </>
                    ) : (
                      <div className="text-[10px] text-slate-500 text-center p-2 font-mono">
                        Not Tuned
                      </div>
                    )}
                  </div>

                  <div className="text-[11px] font-semibold text-slate-300 truncate">{stg.title}</div>
                  <div className="text-[10px] text-slate-400 line-clamp-2 leading-tight">
                    {stg.subtitle}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* ========================================================= */}
      {/* HIGH-DEFINITION ZOOM MODAL LIGHTBOX                       */}
      {/* ========================================================= */}
      {lightbox && (() => {
        const lbStages = lightbox.videoKey === 'video_a' ? stagesA : stagesB;
        const curStage = lbStages.find((s) => s.id === lightbox.stageId) || lbStages[0];
        const isVideoA = lightbox.videoKey === 'video_a';

        return (
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/85 backdrop-blur-md animate-in fade-in duration-200">
            <div className="relative w-full max-w-5xl max-h-[92vh] flex flex-col rounded-3xl bg-slate-900/95 border border-slate-800 shadow-2xl overflow-hidden">
              {/* Lightbox Header */}
              <div className="flex flex-wrap items-center justify-between gap-3 p-4 border-b border-slate-800 bg-slate-950/80">
                <div className="flex items-center gap-3">
                  <div
                    className={`w-7 h-7 rounded-lg flex items-center justify-center font-bold text-xs ${
                      isVideoA
                        ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                        : 'bg-teal-500/20 text-teal-400 border border-teal-500/30'
                    }`}
                  >
                    {isVideoA ? 'A' : 'B'}
                  </div>
                  <div>
                    <h3 className="text-sm font-bold text-white flex items-center gap-2">
                      <span>{curStage.title}</span>
                      <span className="text-xs text-slate-400 font-mono">
                        ({curStage.resolution})
                      </span>
                    </h3>
                    <p className="text-[11px] text-slate-400">
                      {isVideoA ? compareResult.video_a.name : compareResult.video_b.name} •{' '}
                      {curStage.subtitle}
                    </p>
                  </div>
                </div>

                {/* Lightbox Controls: Stage Selector, Invert, Zoom, Download, Close */}
                <div className="flex items-center gap-2">
                  {/* Switch Video inside modal */}
                  <div className="inline-flex rounded-lg bg-slate-900 p-0.5 border border-slate-800">
                    <button
                      type="button"
                      onClick={() => setLightbox({ ...lightbox, videoKey: 'video_a' })}
                      className={`px-2.5 py-1 rounded text-xs font-semibold transition-all cursor-pointer ${
                        isVideoA ? 'bg-emerald-600 text-white shadow' : 'text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      Video A
                    </button>
                    <button
                      type="button"
                      onClick={() => setLightbox({ ...lightbox, videoKey: 'video_b' })}
                      className={`px-2.5 py-1 rounded text-xs font-semibold transition-all cursor-pointer ${
                        !isVideoA ? 'bg-teal-600 text-white shadow' : 'text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      Video B
                    </button>
                  </div>

                  {/* Stage Switcher inside modal */}
                  <div className="hidden sm:inline-flex rounded-lg bg-slate-900 p-0.5 border border-slate-800">
                    {[1, 2, 3, 4, 5].map((id) => (
                      <button
                        key={id}
                        type="button"
                        onClick={() => setLightbox({ ...lightbox, stageId: id })}
                        className={`w-7 h-7 rounded text-xs font-semibold transition-all flex items-center justify-center cursor-pointer ${
                          lightbox.stageId === id
                            ? isVideoA
                              ? 'bg-emerald-600 text-white font-bold'
                              : 'bg-teal-600 text-white font-bold'
                            : 'text-slate-400 hover:text-slate-200'
                        }`}
                      >
                        {id}
                      </button>
                    ))}
                  </div>

                  {/* Color Inversion Toggle */}
                  <button
                    type="button"
                    onClick={() => setLightbox({ ...lightbox, inverted: !lightbox.inverted })}
                    title="Invert Image Colors (Negative View)"
                    className={`p-2 rounded-xl border transition-all cursor-pointer ${
                      lightbox.inverted
                        ? 'bg-amber-500/20 text-amber-300 border-amber-500/40'
                        : 'bg-slate-800 text-slate-300 border-slate-700 hover:bg-slate-700'
                    }`}
                  >
                    {lightbox.inverted ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
                  </button>

                  {/* Zoom Controls */}
                  <div className="inline-flex rounded-lg bg-slate-900 p-0.5 border border-slate-800">
                    <button
                      type="button"
                      onClick={() =>
                        setLightbox({
                          ...lightbox,
                          zoom: Math.max(1, (lightbox.zoom || 1) - 0.5),
                        })
                      }
                      title="Zoom Out"
                      className="p-1.5 rounded text-slate-300 hover:text-white cursor-pointer"
                    >
                      <ZoomOut className="w-4 h-4" />
                    </button>
                    <span className="px-2 py-1 text-xs font-mono font-bold text-emerald-400 flex items-center">
                      {Math.round((lightbox.zoom || 1) * 100)}%
                    </span>
                    <button
                      type="button"
                      onClick={() =>
                        setLightbox({
                          ...lightbox,
                          zoom: Math.min(3, (lightbox.zoom || 1) + 0.5),
                        })
                      }
                      title="Zoom In"
                      className="p-1.5 rounded text-slate-300 hover:text-white cursor-pointer"
                    >
                      <ZoomIn className="w-4 h-4" />
                    </button>
                    <button
                      type="button"
                      onClick={() => setLightbox({ ...lightbox, zoom: 1 })}
                      title="Reset Zoom to 100%"
                      className="p-1.5 rounded text-slate-400 hover:text-slate-200 cursor-pointer"
                    >
                      <RotateCcw className="w-3.5 h-3.5" />
                    </button>
                  </div>

                  {/* Download Image */}
                  {curStage.url && (
                    <a
                      href={curStage.url}
                      download={`cattle_muzzle_${lightbox.videoKey}_stage_${curStage.id}.png`}
                      title="Download full resolution image"
                      className="p-2 rounded-xl bg-slate-800 text-slate-300 hover:text-white border border-slate-700 hover:bg-slate-700 transition-all cursor-pointer"
                    >
                      <Download className="w-4 h-4" />
                    </a>
                  )}

                  {/* Close Modal */}
                  <button
                    type="button"
                    onClick={() => setLightbox(null)}
                    title="Close Inspector (Esc)"
                    className="p-2 rounded-xl bg-slate-800/80 hover:bg-rose-900/60 text-slate-300 hover:text-rose-200 border border-slate-700 hover:border-rose-700 transition-all cursor-pointer ml-1"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
              </div>

              {/* Main Image Viewport */}
              <div className="relative flex-1 min-h-[50vh] max-h-[68vh] overflow-auto flex items-center justify-center p-6 bg-slate-950/90 select-none">
                {curStage.isAvailable ? (
                  <img
                    src={curStage.url}
                    alt={curStage.title}
                    style={{
                      transform: `scale(${lightbox.zoom || 1})`,
                      filter: lightbox.inverted ? 'invert(1) hue-rotate(180deg)' : 'none',
                    }}
                    className="max-h-[64vh] max-w-full object-contain transition-transform duration-150 rounded-xl shadow-2xl"
                  />
                ) : (
                  <div className="text-center p-8 text-slate-400 flex flex-col items-center gap-3">
                    <Filter className="w-10 h-10 text-purple-400 opacity-60" />
                    <p className="text-sm font-semibold">Stage {curStage.id} output is not yet available.</p>
                    <p className="text-xs text-slate-400 max-w-sm">
                      Close the inspector and click "Tune Video {isVideoA ? 'A' : 'B'} Grooves" to run extraction.
                    </p>
                  </div>
                )}

                {/* Left/Right Stage Navigation Chevrons */}
                <button
                  type="button"
                  onClick={() =>
                    setLightbox({
                      ...lightbox,
                      stageId: lightbox.stageId > 1 ? lightbox.stageId - 1 : 5,
                    })
                  }
                  title="Previous Stage (Left Arrow)"
                  className="absolute left-4 top-1/2 -translate-y-1/2 p-2.5 rounded-full bg-slate-900/80 hover:bg-slate-800 text-slate-200 border border-slate-700 backdrop-blur-md transition-all cursor-pointer shadow-xl"
                >
                  <ChevronLeft className="w-5 h-5" />
                </button>

                <button
                  type="button"
                  onClick={() =>
                    setLightbox({
                      ...lightbox,
                      stageId: lightbox.stageId < 5 ? lightbox.stageId + 1 : 1,
                    })
                  }
                  title="Next Stage (Right Arrow)"
                  className="absolute right-4 top-1/2 -translate-y-1/2 p-2.5 rounded-full bg-slate-900/80 hover:bg-slate-800 text-slate-200 border border-slate-700 backdrop-blur-md transition-all cursor-pointer shadow-xl"
                >
                  <ChevronRight className="w-5 h-5" />
                </button>
              </div>

              {/* Lightbox Footer Info */}
              <div className="flex flex-wrap items-center justify-between gap-3 px-5 py-3 border-t border-slate-800 bg-slate-950/90 text-xs text-slate-400 font-mono">
                <div className="flex items-center gap-4">
                  <span>
                    Stage {curStage.id} of 5: <strong className="text-white">{curStage.title}</strong>
                  </span>
                  <span>{curStage.description}</span>
                </div>
                <div className="flex items-center gap-3">
                  <span>Use &larr; &rarr; arrow keys to cycle stages</span>
                  <span className="px-2 py-0.5 rounded bg-slate-900 text-slate-300 border border-slate-800">
                    Esc to Close
                  </span>
                </div>
              </div>
            </div>
          </div>
        );
      })()}
    </div>
  );
}
