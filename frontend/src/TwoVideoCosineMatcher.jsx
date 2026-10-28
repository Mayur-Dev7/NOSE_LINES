import React, { useState, useEffect } from 'react';
import {
  Film,
  ScanLine,
  GitCompare,
  CheckCircle2,
  XCircle,
  Database,
  ArrowRight,
  Sparkles,
  RefreshCw,
  Upload,
  Layers,
  ChevronDown,
  Info,
  Trash2,
  Eye,
  Sliders,
  Maximize2,
  ShieldCheck,
  AlertCircle,
  Crosshair,
  Network,
  Activity,
  FileText
} from 'lucide-react';

const FEATURE_LABELS = {
  groove_pixels: 'Groove Pixels (Valleys)',
  skeleton_pixels: 'Ridge Skeleton (5th Image 1-px)',
  groove_coverage_pct: 'Groove Area Coverage',
  endpoints: 'Line Endpoints (Degree 1)',
  junctions: 'Line Junctions (Degree >= 3)',
  connected_components: 'Connected Line Clusters',
  mean_branch_length: 'Mean Branch Length',
  cycles: 'Loop Cycles (Bead Valleys)',
  density: 'Ridge Line Density',
  euler_characteristic: 'Euler Characteristic (χ)',
};

const FEATURE_UNITS = {
  groove_pixels: 'px',
  skeleton_pixels: 'px',
  groove_coverage_pct: '%',
  endpoints: 'ends',
  junctions: 'juncs',
  connected_components: 'clusters',
  mean_branch_length: 'px',
  cycles: 'loops',
  density: '',
  euler_characteristic: 'χ',
};

export default function TwoVideoCosineMatcher({ sampleVideos = [] }) {
  // Video selection
  const [videoA, setVideoA] = useState('/home/adis_mayur/F_CODES/muzzleVideo.mp4');
  const [videoB, setVideoB] = useState('/home/adis_mayur/F_CODES/muzzleVideo2.mp4');

  // Filter & Threshold Configuration
  const [filterMethod, setFilterMethod] = useState('frangi'); // 'frangi' | 'blackhat'
  const [thresholdMethod, setThresholdMethod] = useState('adaptive'); // 'adaptive' | 'otsu'
  const [adaptiveBlockSize, setAdaptiveBlockSize] = useState(15);
  const [adaptiveC, setAdaptiveC] = useState(2);
  const [excludeNostrils, setExcludeNostrils] = useState(true);
  const [minSpurLength, setMinSpurLength] = useState(8);
  const [minComponentSize, setMinComponentSize] = useState(30);

  // Calibration status from /api/video/matcher-config
  const [matcherConfig, setMatcherConfig] = useState(null);
  const [loadingConfig, setLoadingConfig] = useState(false);

  // Loading states
  const [loadingA, setLoadingA] = useState(false);
  const [loadingB, setLoadingB] = useState(false);
  const [extractLogA, setExtractLogA] = useState('');
  const [extractLogB, setExtractLogB] = useState('');

  // Extracted 5-frame results
  const [dataA, setDataA] = useState(null);
  const [dataB, setDataB] = useState(null);

  // Active view toggle on the 5 cards: 'skeleton' (5th Image default) | 'overlay' | 'crop' | 'enhanced' | 'response' | 'mask'
  const [viewTypeA, setViewTypeA] = useState('skeleton');
  const [viewTypeB, setViewTypeB] = useState('skeleton');

  // Full-size image modal for frame preview
  const [previewImage, setPreviewImage] = useState(null);

  // Full-size modal for Keypoint Match Visualization
  const [visualModalOpen, setVisualModalOpen] = useState(false);

  // Biometric Matcher Result
  const [biometricMatchResult, setBiometricMatchResult] = useState(null);
  const [matchingBiometric, setMatchingBiometric] = useState(false);
  const [matchError, setMatchError] = useState(null);

  // Secondary Cosine Calculation Result
  const [similarityResult, setSimilarityResult] = useState(null);
  const [calculatingSim, setCalculatingSim] = useState(false);
  const [showCosineDetails, setShowCosineDetails] = useState(false);

  // LocalStorage inspection modal
  const [showLocalStorageModal, setShowLocalStorageModal] = useState(false);
  const [storageData, setStorageData] = useState({ a: null, b: null });

  // Fetch matcher calibration configuration
  const fetchMatcherConfig = async () => {
    setLoadingConfig(true);
    try {
      const res = await fetch('/api/video/matcher-config');
      if (res.ok) {
        const config = await res.json();
        setMatcherConfig(config);
      } else {
        setMatcherConfig({ calibrated: false, status: 'UNCALIBRATED' });
      }
    } catch (err) {
      console.warn('Failed to fetch matcher config:', err);
      setMatcherConfig({ calibrated: false, status: 'UNCALIBRATED' });
    } finally {
      setLoadingConfig(false);
    }
  };

  useEffect(() => {
    fetchMatcherConfig();
  }, []);

  // On mount: check if parameters exist in LocalStorage
  useEffect(() => {
    try {
      const storedA = localStorage.getItem('muzzle_video_a_params');
      const storedB = localStorage.getItem('muzzle_video_b_params');
      if (storedA) {
        const parsedA = JSON.parse(storedA);
        setDataA(parsedA);
      }
      if (storedB) {
        const parsedB = JSON.parse(storedB);
        setDataB(parsedB);
      }
    } catch (e) {
      console.warn('LocalStorage read error:', e);
    }
  }, []);

  // Sync state to LocalStorage
  const saveToLocalStorage = (key, data) => {
    try {
      localStorage.setItem(key, JSON.stringify(data));
      setStorageData({
        a: localStorage.getItem('muzzle_video_a_params'),
        b: localStorage.getItem('muzzle_video_b_params'),
      });
    } catch (e) {
      console.warn('LocalStorage save error:', e);
    }
  };

  // Clear LocalStorage
  const handleClearStorage = () => {
    localStorage.removeItem('muzzle_video_a_params');
    localStorage.removeItem('muzzle_video_b_params');
    setDataA(null);
    setDataB(null);
    setBiometricMatchResult(null);
    setSimilarityResult(null);
    setMatchError(null);
    setStorageData({ a: null, b: null });
  };

  // Extract 5 images & groove lines for Video A
  const handleExtractVideoA = async () => {
    if (loadingA || loadingB) return;
    setLoadingA(true);
    setBiometricMatchResult(null);
    setSimilarityResult(null);
    setMatchError(null);
    setExtractLogA('Running Mask R-CNN detector with Canonical Alignment & Skeleton Pruning...');
    try {
      const res = await fetch('/api/video/extract-five-muzzle-parameters', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          video_path: videoA,
          method: filterMethod,
          threshold_method: thresholdMethod,
          adaptive_block_size: parseInt(adaptiveBlockSize, 10),
          adaptive_c: parseInt(adaptiveC, 10),
          exclude_nostrils: excludeNostrils,
          min_spur_length: parseInt(minSpurLength, 10),
          min_component_size: parseInt(minComponentSize, 10),
        }),
      });
      const data = await res.json();
      if (data.success) {
        setDataA(data);
        saveToLocalStorage('muzzle_video_a_params', data);
        setExtractLogA(`Extracted! 5th-stage skeleton lines (${data.median_parameters?.skeleton_pixels ?? 'N/A'} px, ${data.median_parameters?.endpoints ?? 'N/A'} endpoints). Stored in LocalStorage.`);
      } else {
        alert(data.detail || 'Extraction failed for Video A');
        setExtractLogA('Extraction failed');
      }
    } catch (err) {
      alert('Failed to connect to backend: ' + err.message);
      setExtractLogA('Backend error: ' + err.message);
    } finally {
      setLoadingA(false);
    }
  };

  // Extract 5 images & groove lines for Video B
  const handleExtractVideoB = async () => {
    if (loadingA || loadingB) return;
    setLoadingB(true);
    setBiometricMatchResult(null);
    setSimilarityResult(null);
    setMatchError(null);
    setExtractLogB('Running Mask R-CNN detector with Canonical Alignment & Skeleton Pruning...');
    try {
      const res = await fetch('/api/video/extract-five-muzzle-parameters', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          video_path: videoB,
          method: filterMethod,
          threshold_method: thresholdMethod,
          adaptive_block_size: parseInt(adaptiveBlockSize, 10),
          adaptive_c: parseInt(adaptiveC, 10),
          exclude_nostrils: excludeNostrils,
          min_spur_length: parseInt(minSpurLength, 10),
          min_component_size: parseInt(minComponentSize, 10),
        }),
      });
      const data = await res.json();
      if (data.success) {
        setDataB(data);
        saveToLocalStorage('muzzle_video_b_params', data);
        setExtractLogB(`Extracted! 5th-stage skeleton lines (${data.median_parameters?.skeleton_pixels ?? 'N/A'} px, ${data.median_parameters?.endpoints ?? 'N/A'} endpoints). Stored in LocalStorage.`);
      } else {
        alert(data.detail || 'Extraction failed for Video B');
        setExtractLogB('Extraction failed');
      }
    } catch (err) {
      alert('Failed to connect to backend: ' + err.message);
      setExtractLogB('Backend error: ' + err.message);
    } finally {
      setLoadingB(false);
    }
  };

  // Extract both videos sequentially
  const handleExtractBoth = async () => {
    if (loadingA || loadingB) return;
    await handleExtractVideoA();
    await handleExtractVideoB();
  };

  // RUN BIOMETRIC VERIFICATION (SIFT + MINUTIAE)
  const handleRunBiometricMatch = async () => {
    if (matchingBiometric) return;
    setMatchingBiometric(true);
    setMatchError(null);

    // Refresh config just before match
    fetchMatcherConfig();

    try {
      const payload = {
        video_a_path: videoA,
        video_b_path: videoB,
        filter_method: filterMethod,
        threshold_method: thresholdMethod,
        adaptive_block_size: parseInt(adaptiveBlockSize, 10),
        adaptive_c: parseInt(adaptiveC, 10),
        exclude_nostrils: excludeNostrils,
        min_spur_length: parseInt(minSpurLength, 10),
        min_component_size: parseInt(minComponentSize, 10),
        frames_a: dataA?.frames || null,
        frames_b: dataB?.frames || null,
      };

      const res = await fetch('/api/video/match', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server returned ${res.status}`);
      }

      const matchData = await res.json();
      setBiometricMatchResult(matchData);
    } catch (err) {
      console.error('Biometric matching error:', err);
      setMatchError('Biometric matching failed: ' + err.message);
    } finally {
      setMatchingBiometric(false);
    }
  };

  // Calculate Cosine Similarity from LocalStorage (Fallback / Secondary metric)
  const handleCalculateCosineSimilarity = async () => {
    let storedA = null;
    let storedB = null;
    try {
      const rawA = localStorage.getItem('muzzle_video_a_params');
      const rawB = localStorage.getItem('muzzle_video_b_params');
      if (rawA) storedA = JSON.parse(rawA);
      if (rawB) storedB = JSON.parse(rawB);
    } catch (e) {
      console.warn('LocalStorage parse error:', e);
    }

    const currentA = storedA || dataA;
    const currentB = storedB || dataB;

    if (!currentA || !currentB) {
      alert('Please extract 5 images for both Video A and Video B first to store their parameters in LocalStorage.');
      return;
    }

    setCalculatingSim(true);
    try {
      const vecA = currentA.vector_array || Object.values(currentA.median_parameters);
      const vecB = currentB.vector_array || Object.values(currentB.median_parameters);

      const res = await fetch('/api/video/cosine-similarity', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ vector_a: vecA, vector_b: vecB }),
      });
      const simData = await res.json();
      setSimilarityResult({
        ...simData,
        paramsA: currentA.median_parameters,
        paramsB: currentB.median_parameters,
        videoAName: currentA.video_name,
        videoBName: currentB.video_name,
      });
      setShowCosineDetails(true);
    } catch (err) {
      alert('Cosine similarity calculation failed: ' + err.message);
    } finally {
      setCalculatingSim(false);
    }
  };

  const getFrameImageUrl = (frame, viewType) => {
    switch (viewType) {
      case 'crop':
        return frame.crop_url;
      case 'enhanced':
        return frame.enhanced_url;
      case 'response':
        return frame.groove_response_url;
      case 'mask':
        return frame.groove_mask_url;
      case 'overlay':
        return frame.overlay_url;
      case 'skeleton':
      default:
        return frame.skeleton_url; // 5th Image: Thin Skeleton
    }
  };

  const isCalibrated = matcherConfig && matcherConfig.calibrated === true;
  const calibratedThreshold = isCalibrated && matcherConfig.threshold !== null && matcherConfig.threshold !== undefined
    ? Number(matcherConfig.threshold)
    : null;

  return (
    <div className="flex flex-col gap-6 text-slate-100 max-w-7xl mx-auto w-full pb-16">
      {/* Header Bar */}
      <div className="glass-panel p-6 rounded-2xl border border-slate-800 flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 flex-wrap">
            <h1 className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
              <ShieldCheck className="w-6 h-6 text-emerald-400" />
              2-Video Muzzle Biometric Matcher
            </h1>
            <span className="text-xs font-semibold px-2.5 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 font-mono">
              SIFT/ORB RANSAC &bull; Minutiae Topology &bull; Canonical Alignment
            </span>

            {/* Calibration Status Badge */}
            {isCalibrated ? (
              <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 font-mono shadow-sm">
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                Calibrated EER Threshold: {(calibratedThreshold * 100).toFixed(1)}% ({calibratedThreshold.toFixed(3)})
              </span>
            ) : (
              <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold bg-amber-500/20 text-amber-300 border border-amber-500/40 font-mono">
                <AlertCircle className="w-3.5 h-3.5 text-amber-400" />
                UNCALIBRATED
              </span>
            )}
          </div>

          <p className="text-xs text-slate-400 mt-1.5">
            Extracts canonical 448x448 muzzle crops, performs skeleton spur &amp; component pruning, runs 5×5 pairwise verification using SIFT/ORB RANSAC inliers and Minutiae graph topology.
          </p>
        </div>

        {/* Global Action Buttons */}
        <div className="flex items-center gap-2">
          <button
            onClick={fetchMatcherConfig}
            disabled={loadingConfig}
            className="flex items-center gap-1.5 px-3 py-2 rounded-xl text-xs font-semibold bg-slate-900 hover:bg-slate-800 border border-slate-700 text-slate-300 transition-colors"
            title="Refresh matcher configuration"
          >
            <RefreshCw className={`w-3.5 h-3.5 text-slate-400 ${loadingConfig ? 'animate-spin' : ''}`} />
            Refresh Config
          </button>
          <button
            onClick={() => {
              setStorageData({
                a: localStorage.getItem('muzzle_video_a_params'),
                b: localStorage.getItem('muzzle_video_b_params'),
              });
              setShowLocalStorageModal(true);
            }}
            className="flex items-center gap-1.5 px-3 py-2 rounded-xl text-xs font-semibold bg-slate-900 hover:bg-slate-800 border border-slate-700 text-slate-300 transition-colors"
          >
            <Database className="w-3.5 h-3.5 text-teal-400" />
            LocalStorage
          </button>
          <button
            onClick={handleClearStorage}
            className="flex items-center gap-1.5 px-3 py-2 rounded-xl text-xs font-semibold bg-slate-900 hover:bg-rose-950/40 border border-slate-800 hover:border-rose-800 text-slate-400 hover:text-rose-300 transition-colors"
            title="Clear stored parameters"
          >
            <Trash2 className="w-3.5 h-3.5" />
            Reset
          </button>
        </div>
      </div>

      {/* Uncalibrated Alert Banner (Helpful Tip) */}
      {!isCalibrated && (
        <div className="glass-panel p-3.5 rounded-xl border border-amber-500/40 bg-amber-950/20 flex items-center justify-between gap-3 text-xs">
          <div className="flex items-center gap-2.5 text-amber-200">
            <AlertCircle className="w-4 h-4 text-amber-400 shrink-0" />
            <span>
              <strong>Matcher Status: UNCALIBRATED.</strong> Run <code className="px-1.5 py-0.5 rounded bg-slate-900 border border-amber-500/30 text-amber-300 font-mono">tools/eval_matching.py</code> to calibrate decision threshold and generate ROC/EER operating point.
            </span>
          </div>
          <button
            onClick={fetchMatcherConfig}
            className="px-2.5 py-1 rounded-lg bg-amber-500/20 hover:bg-amber-500/30 text-amber-300 border border-amber-500/40 font-mono text-[11px] shrink-0"
          >
            Check Again
          </button>
        </div>
      )}

      {/* ========================================================= */}
      {/* THRESHOLD & SKELETON PRUNING CONFIGURATION OPTIONS        */}
      {/* ========================================================= */}
      <div className="glass-panel p-4 rounded-xl border border-slate-800 flex flex-col gap-3">
        <div className="flex items-center justify-between border-b border-slate-800/80 pb-2 flex-wrap gap-2">
          <div className="flex items-center gap-2 text-xs font-bold text-slate-200">
            <Sliders className="w-4 h-4 text-emerald-400" />
            <span>Groove Extraction, Pruning &amp; Matching Options:</span>
          </div>
          <span className="text-[10px] text-slate-400 font-mono">
            Pipeline: Quality Gating &rarr; Canonical Alignment (448px) &rarr; Spur/Component Pruning &rarr; SIFT + Minutiae
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-5 gap-4 text-xs">
          {/* Option 1: Filter Method */}
          <div className="flex flex-col gap-1.5">
            <label className="text-slate-400 font-semibold flex items-center justify-between">
              <span>Filter Method:</span>
              <span className="text-[10px] font-mono text-emerald-400 font-bold">{filterMethod === 'frangi' ? 'Frangi' : 'Black-Hat'}</span>
            </label>
            <select
              value={filterMethod}
              onChange={(e) => setFilterMethod(e.target.value)}
              className="bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-slate-200 focus:outline-hidden focus:border-emerald-500 font-medium"
            >
              <option value="frangi">Frangi Hessian (Eigenvalues)</option>
              <option value="blackhat">Multi-Scale Black-Hat</option>
            </select>
          </div>

          {/* Option 2: Threshold Method */}
          <div className="flex flex-col gap-1.5">
            <label className="text-slate-400 font-semibold flex items-center justify-between">
              <span>Threshold Method:</span>
              <span className="text-[10px] font-mono text-emerald-400 font-bold">{thresholdMethod === 'adaptive' ? 'Adaptive' : 'Otsu'}</span>
            </label>
            <select
              value={thresholdMethod}
              onChange={(e) => setThresholdMethod(e.target.value)}
              className="bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-slate-200 focus:outline-hidden focus:border-emerald-500 font-medium"
            >
              <option value="adaptive">Adaptive Gaussian</option>
              <option value="otsu">Standard / Otsu</option>
            </select>
          </div>

          {/* Option 3: Adaptive Gaussian Block Size & Constant C */}
          {thresholdMethod === 'adaptive' ? (
            <div className="flex items-center gap-2">
              <div className="flex-1 flex flex-col gap-1.5">
                <label className="text-slate-400 font-semibold flex items-center justify-between">
                  <span>Block:</span>
                  <span className="text-[10px] font-mono text-slate-300">{adaptiveBlockSize}px</span>
                </label>
                <select
                  value={adaptiveBlockSize}
                  onChange={(e) => setAdaptiveBlockSize(Number(e.target.value))}
                  className="bg-slate-900 border border-slate-700 rounded-lg px-2 py-1.5 text-slate-200 focus:outline-hidden focus:border-emerald-500 font-mono"
                >
                  <option value={9}>9px</option>
                  <option value={15}>15px</option>
                  <option value={21}>21px</option>
                  <option value={31}>31px</option>
                </select>
              </div>

              <div className="flex-1 flex flex-col gap-1.5">
                <label className="text-slate-400 font-semibold flex items-center justify-between">
                  <span>C:</span>
                  <span className="text-[10px] font-mono text-slate-300">+{adaptiveC}</span>
                </label>
                <select
                  value={adaptiveC}
                  onChange={(e) => setAdaptiveC(Number(e.target.value))}
                  className="bg-slate-900 border border-slate-700 rounded-lg px-2 py-1.5 text-slate-200 focus:outline-hidden focus:border-emerald-500 font-mono"
                >
                  <option value={1}>1</option>
                  <option value={2}>2</option>
                  <option value={3}>3</option>
                  <option value={5}>5</option>
                </select>
              </div>
            </div>
          ) : (
            <div className="flex flex-col gap-1.5">
              <label className="text-slate-400 font-semibold">Standard Threshold:</label>
              <div className="px-3 py-1.5 rounded-lg bg-slate-900/60 border border-slate-800 text-slate-400 text-xs truncate">
                Otsu auto-bimodal
              </div>
            </div>
          )}

          {/* Option 4: Skeleton Pruning Parameters */}
          <div className="flex items-center gap-2">
            <div className="flex-1 flex flex-col gap-1.5">
              <label className="text-slate-400 font-semibold flex items-center justify-between">
                <span>Spur Prune:</span>
                <span className="text-[10px] font-mono text-teal-400">&lt;{minSpurLength}px</span>
              </label>
              <select
                value={minSpurLength}
                onChange={(e) => setMinSpurLength(Number(e.target.value))}
                className="bg-slate-900 border border-slate-700 rounded-lg px-2 py-1.5 text-slate-200 focus:outline-hidden focus:border-teal-500 font-mono"
              >
                <option value={4}>4 px</option>
                <option value={8}>8 px (Default)</option>
                <option value={12}>12 px</option>
                <option value={16}>16 px</option>
              </select>
            </div>

            <div className="flex-1 flex flex-col gap-1.5">
              <label className="text-slate-400 font-semibold flex items-center justify-between">
                <span>Min Cluster:</span>
                <span className="text-[10px] font-mono text-teal-400">&ge;{minComponentSize}px</span>
              </label>
              <select
                value={minComponentSize}
                onChange={(e) => setMinComponentSize(Number(e.target.value))}
                className="bg-slate-900 border border-slate-700 rounded-lg px-2 py-1.5 text-slate-200 focus:outline-hidden focus:border-teal-500 font-mono"
              >
                <option value={15}>15 px</option>
                <option value={30}>30 px (Default)</option>
                <option value={50}>50 px</option>
              </select>
            </div>
          </div>

          {/* Option 5: Nostril Cavity Suppression */}
          <div className="flex flex-col gap-1.5 justify-center">
            <label className="text-slate-400 font-semibold">Nostril Cavity:</label>
            <label className="flex items-center gap-2 cursor-pointer text-slate-300 select-none py-1">
              <input
                type="checkbox"
                checked={excludeNostrils}
                onChange={(e) => setExcludeNostrils(e.target.checked)}
                className="w-4 h-4 rounded text-emerald-500 focus:ring-emerald-400 bg-slate-900 border-slate-700"
              />
              <span className="text-xs">Suppress Nostril Dark Holes</span>
            </label>
          </div>
        </div>
      </div>

      {/* Preset Quick Tests Bar */}
      <div className="glass-panel p-3.5 rounded-xl border border-slate-800 flex flex-wrap items-center justify-between gap-3 text-xs">
        <div className="flex items-center gap-2 text-slate-300 font-semibold">
          <Sparkles className="w-4 h-4 text-emerald-400" />
          <span>Quick Video Pairs:</span>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <button
            onClick={() => {
              setVideoA('/home/adis_mayur/F_CODES/muzzleVideo.mp4');
              setVideoB('/home/adis_mayur/F_CODES/muzzleVideo.mp4');
            }}
            className="px-3 py-1.5 rounded-lg bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 font-medium transition-colors"
          >
            🟢 Same Cattle (Video 1 vs Video 1)
          </button>
          <button
            onClick={() => {
              setVideoA('/home/adis_mayur/F_CODES/muzzleVideo.mp4');
              setVideoB('/home/adis_mayur/F_CODES/muzzleVideo2.mp4');
            }}
            className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 font-medium transition-colors"
          >
            🔍 Pair A/B (Video 1 vs Video 2)
          </button>
          <button
            onClick={() => {
              setVideoA('/home/adis_mayur/F_CODES/muzzleVideo2.mp4');
              setVideoB('/home/adis_mayur/F_CODES/muzzleVideo3.mp4');
            }}
            className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 font-medium transition-colors"
          >
            🔍 Pair B/C (Video 2 vs Video 3)
          </button>
          <button
            onClick={() => {
              setVideoA('/home/adis_mayur/F_CODES/muzzleVideo5.mp4');
              setVideoB('/home/adis_mayur/F_CODES/muzzleVideo6.mp4');
            }}
            className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 font-medium transition-colors"
          >
            🔍 High-Res (Video 5 vs Video 6)
          </button>
        </div>
      </div>

      {/* 2-Column Video Panels */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* ========================================================= */}
        {/* VIDEO A PANEL                                             */}
        {/* ========================================================= */}
        <div className="glass-panel p-5 rounded-2xl border border-slate-800 flex flex-col gap-4">
          <div className="flex items-center justify-between pb-3 border-b border-slate-800">
            <div className="flex items-center gap-2">
              <span className="w-7 h-7 rounded-lg bg-emerald-500/20 text-emerald-400 font-bold flex items-center justify-center text-sm border border-emerald-500/30">
                A
              </span>
              <div>
                <h2 className="text-base font-bold text-white">Cattle Video A (Reference Profile)</h2>
                <span className="text-[11px] text-slate-400">Extracts 5 canonical frames &bull; computes minutiae topology</span>
              </div>
            </div>
            {dataA && (
              <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-[11px] font-bold bg-emerald-500/15 text-emerald-300 border border-emerald-500/30">
                <Database className="w-3 h-3" />
                In LocalStorage
              </span>
            )}
          </div>

          {/* Video Selector Dropdown */}
          <div className="flex flex-col gap-1.5 text-xs">
            <label className="text-slate-400 font-medium">Select Video A:</label>
            <select
              value={videoA}
              onChange={(e) => setVideoA(e.target.value)}
              className="bg-slate-900 border border-slate-700 rounded-xl px-3 py-2 text-slate-200 focus:outline-hidden focus:border-emerald-500"
            >
              <option value="/home/adis_mayur/F_CODES/muzzleVideo.mp4">muzzleVideo.mp4 (2.55 MB)</option>
              <option value="/home/adis_mayur/F_CODES/muzzleVideo2.mp4">muzzleVideo2.mp4 (2.59 MB)</option>
              <option value="/home/adis_mayur/F_CODES/muzzleVideo3.mp4">muzzleVideo3.mp4 (2.63 MB)</option>
              <option value="/home/adis_mayur/F_CODES/muzzleVideo5.mp4">muzzleVideo5.mp4 (4.25 MB)</option>
              <option value="/home/adis_mayur/F_CODES/muzzleVideo6.mp4">muzzleVideo6.mp4 (4.26 MB)</option>
              <option value="/home/adis_mayur/F_CODES/muzzleVideo7.mp4">muzzleVideo7.mp4 (4.25 MB)</option>
            </select>
          </div>

          {/* Extraction Trigger */}
          <button
            onClick={handleExtractVideoA}
            disabled={loadingA || loadingB}
            className={`w-full py-2.5 px-4 rounded-xl font-bold text-xs flex items-center justify-center gap-2 transition-all shadow-md ${
              loadingA || loadingB
                ? 'bg-slate-800 text-slate-500 cursor-not-allowed'
                : 'bg-gradient-to-r from-emerald-500 to-teal-500 hover:from-emerald-400 hover:to-teal-400 text-slate-950 shadow-emerald-500/20'
            }`}
          >
            {loadingA ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin text-emerald-400" />
                <span>Aligning &amp; Extracting 5 Frames (Video A)...</span>
              </>
            ) : (
              <>
                <ScanLine className="w-4 h-4" />
                <span>{dataA ? 'Re-Extract 5 Images (Video A)' : 'Extract 5 Images & Store in LocalStorage'}</span>
              </>
            )}
          </button>

          {extractLogA && (
            <div className="text-[11px] font-mono text-emerald-400 bg-emerald-950/30 px-3 py-1.5 rounded-lg border border-emerald-500/30">
              {extractLogA}
            </div>
          )}

          {/* 5 Extracted Images Strip */}
          {dataA && (
            <div className="flex flex-col gap-3 pt-2">
              <div className="flex items-center justify-between text-xs text-slate-400">
                <span className="font-semibold text-slate-200">5 Extracted Muzzle Images:</span>
                <div className="flex items-center gap-1 bg-slate-900 p-0.5 rounded-lg border border-slate-800 text-[10px]">
                  <button
                    onClick={() => setViewTypeA('skeleton')}
                    className={`px-2 py-0.5 rounded ${viewTypeA === 'skeleton' ? 'bg-emerald-500/20 text-emerald-300 font-bold border border-emerald-500/40' : 'text-slate-400'}`}
                  >
                    5th Image (Lines)
                  </button>
                  <button
                    onClick={() => setViewTypeA('overlay')}
                    className={`px-2 py-0.5 rounded ${viewTypeA === 'overlay' ? 'bg-slate-800 text-emerald-300 font-bold' : 'text-slate-400'}`}
                  >
                    Overlay
                  </button>
                  <button
                    onClick={() => setViewTypeA('response')}
                    className={`px-2 py-0.5 rounded ${viewTypeA === 'response' ? 'bg-slate-800 text-emerald-300 font-bold' : 'text-slate-400'}`}
                  >
                    Filter Resp
                  </button>
                  <button
                    onClick={() => setViewTypeA('crop')}
                    className={`px-2 py-0.5 rounded ${viewTypeA === 'crop' ? 'bg-slate-800 text-emerald-300 font-bold' : 'text-slate-400'}`}
                  >
                    Mask R-CNN Crop
                  </button>
                </div>
              </div>

              {/* 5 Images Grid */}
              <div className="grid grid-cols-5 gap-2">
                {dataA.frames.map((f, i) => {
                  const imgUrl = getFrameImageUrl(f, viewTypeA);
                  return (
                    <div
                      key={i}
                      onClick={() => setPreviewImage({ url: imgUrl, title: `Video A - Frame #${f.frame_number} (${viewTypeA})`, params: f.parameters })}
                      className="group flex flex-col rounded-lg border border-slate-800 overflow-hidden bg-slate-950 cursor-pointer hover:border-emerald-500 transition-colors"
                    >
                      <div className="aspect-square relative overflow-hidden bg-slate-950">
                        <img
                          src={imgUrl}
                          alt={`Frame ${f.frame_number}`}
                          className="w-full h-full object-cover group-hover:scale-105 transition-transform"
                        />
                        <span className="absolute top-1 left-1 text-[9px] font-mono px-1 rounded bg-slate-950/80 text-slate-300">
                          #{f.frame_number}
                        </span>
                        <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity">
                          <Maximize2 className="w-3.5 h-3.5 text-white" />
                        </div>
                      </div>
                      <div className="p-1 text-[9px] font-mono text-center text-slate-400 truncate bg-slate-900/60">
                        {f.parameters?.skeleton_pixels ?? 0} skel px
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Parameters Summary Badge List (Display Only) */}
              <div className="p-3 bg-slate-950/60 rounded-xl border border-slate-800/80 grid grid-cols-2 gap-2 text-[11px] font-mono">
                <div>
                  <span className="text-slate-400">Ridge Skeleton: </span>
                  <span className="font-bold text-teal-300">{dataA.median_parameters?.skeleton_pixels ?? 'N/A'} px</span>
                </div>
                <div>
                  <span className="text-slate-400">Groove Valleys: </span>
                  <span className="font-bold text-teal-300">{dataA.median_parameters?.groove_pixels ?? 'N/A'} px</span>
                </div>
                <div>
                  <span className="text-slate-400">Endpoints: </span>
                  <span className="font-bold text-slate-200">{dataA.median_parameters?.endpoints ?? 'N/A'}</span>
                </div>
                <div>
                  <span className="text-slate-400">Junctions: </span>
                  <span className="font-bold text-slate-200">{dataA.median_parameters?.junctions ?? 'N/A'}</span>
                </div>
                <div>
                  <span className="text-slate-400">Cycles: </span>
                  <span className="font-bold text-slate-200">{dataA.median_parameters?.cycles ?? 'N/A'}</span>
                </div>
                <div>
                  <span className="text-slate-400">Euler (χ): </span>
                  <span className="font-bold text-slate-200">{dataA.median_parameters?.euler_characteristic ?? 'N/A'}</span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* ========================================================= */}
        {/* VIDEO B PANEL                                             */}
        {/* ========================================================= */}
        <div className="glass-panel p-5 rounded-2xl border border-slate-800 flex flex-col gap-4">
          <div className="flex items-center justify-between pb-3 border-b border-slate-800">
            <div className="flex items-center gap-2">
              <span className="w-7 h-7 rounded-lg bg-teal-500/20 text-teal-400 font-bold flex items-center justify-center text-sm border border-teal-500/30">
                B
              </span>
              <div>
                <h2 className="text-base font-bold text-white">Cattle Video B (Query Candidate)</h2>
                <span className="text-[11px] text-slate-400">Extracts 5 canonical frames &bull; computes minutiae topology</span>
              </div>
            </div>
            {dataB && (
              <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-[11px] font-bold bg-teal-500/15 text-teal-300 border border-teal-500/30">
                <Database className="w-3 h-3" />
                In LocalStorage
              </span>
            )}
          </div>

          {/* Video Selector Dropdown */}
          <div className="flex flex-col gap-1.5 text-xs">
            <label className="text-slate-400 font-medium">Select Video B:</label>
            <select
              value={videoB}
              onChange={(e) => setVideoB(e.target.value)}
              className="bg-slate-900 border border-slate-700 rounded-xl px-3 py-2 text-slate-200 focus:outline-hidden focus:border-teal-500"
            >
              <option value="/home/adis_mayur/F_CODES/muzzleVideo.mp4">muzzleVideo.mp4 (2.55 MB)</option>
              <option value="/home/adis_mayur/F_CODES/muzzleVideo2.mp4">muzzleVideo2.mp4 (2.59 MB)</option>
              <option value="/home/adis_mayur/F_CODES/muzzleVideo3.mp4">muzzleVideo3.mp4 (2.63 MB)</option>
              <option value="/home/adis_mayur/F_CODES/muzzleVideo5.mp4">muzzleVideo5.mp4 (4.25 MB)</option>
              <option value="/home/adis_mayur/F_CODES/muzzleVideo6.mp4">muzzleVideo6.mp4 (4.26 MB)</option>
              <option value="/home/adis_mayur/F_CODES/muzzleVideo7.mp4">muzzleVideo7.mp4 (4.25 MB)</option>
            </select>
          </div>

          {/* Extraction Trigger */}
          <button
            onClick={handleExtractVideoB}
            disabled={loadingA || loadingB}
            className={`w-full py-2.5 px-4 rounded-xl font-bold text-xs flex items-center justify-center gap-2 transition-all shadow-md ${
              loadingA || loadingB
                ? 'bg-slate-800 text-slate-500 cursor-not-allowed'
                : 'bg-gradient-to-r from-teal-500 to-cyan-500 hover:from-teal-400 hover:to-cyan-400 text-slate-950 shadow-teal-500/20'
            }`}
          >
            {loadingB ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin text-teal-400" />
                <span>Aligning &amp; Extracting 5 Frames (Video B)...</span>
              </>
            ) : (
              <>
                <ScanLine className="w-4 h-4" />
                <span>{dataB ? 'Re-Extract 5 Images (Video B)' : 'Extract 5 Images & Store in LocalStorage'}</span>
              </>
            )}
          </button>

          {extractLogB && (
            <div className="text-[11px] font-mono text-teal-400 bg-teal-950/30 px-3 py-1.5 rounded-lg border border-teal-500/30">
              {extractLogB}
            </div>
          )}

          {/* 5 Extracted Images Strip */}
          {dataB && (
            <div className="flex flex-col gap-3 pt-2">
              <div className="flex items-center justify-between text-xs text-slate-400">
                <span className="font-semibold text-slate-200">5 Extracted Muzzle Images:</span>
                <div className="flex items-center gap-1 bg-slate-900 p-0.5 rounded-lg border border-slate-800 text-[10px]">
                  <button
                    onClick={() => setViewTypeB('skeleton')}
                    className={`px-2 py-0.5 rounded ${viewTypeB === 'skeleton' ? 'bg-teal-500/20 text-teal-300 font-bold border border-teal-500/40' : 'text-slate-400'}`}
                  >
                    5th Image (Lines)
                  </button>
                  <button
                    onClick={() => setViewTypeB('overlay')}
                    className={`px-2 py-0.5 rounded ${viewTypeB === 'overlay' ? 'bg-slate-800 text-teal-300 font-bold' : 'text-slate-400'}`}
                  >
                    Overlay
                  </button>
                  <button
                    onClick={() => setViewTypeB('response')}
                    className={`px-2 py-0.5 rounded ${viewTypeB === 'response' ? 'bg-slate-800 text-teal-300 font-bold' : 'text-slate-400'}`}
                  >
                    Filter Resp
                  </button>
                  <button
                    onClick={() => setViewTypeB('crop')}
                    className={`px-2 py-0.5 rounded ${viewTypeB === 'crop' ? 'bg-slate-800 text-teal-300 font-bold' : 'text-slate-400'}`}
                  >
                    Mask R-CNN Crop
                  </button>
                </div>
              </div>

              {/* 5 Images Grid */}
              <div className="grid grid-cols-5 gap-2">
                {dataB.frames.map((f, i) => {
                  const imgUrl = getFrameImageUrl(f, viewTypeB);
                  return (
                    <div
                      key={i}
                      onClick={() => setPreviewImage({ url: imgUrl, title: `Video B - Frame #${f.frame_number} (${viewTypeB})`, params: f.parameters })}
                      className="group flex flex-col rounded-lg border border-slate-800 overflow-hidden bg-slate-950 cursor-pointer hover:border-teal-500 transition-colors"
                    >
                      <div className="aspect-square relative overflow-hidden bg-slate-950">
                        <img
                          src={imgUrl}
                          alt={`Frame ${f.frame_number}`}
                          className="w-full h-full object-cover group-hover:scale-105 transition-transform"
                        />
                        <span className="absolute top-1 left-1 text-[9px] font-mono px-1 rounded bg-slate-950/80 text-slate-300">
                          #{f.frame_number}
                        </span>
                        <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity">
                          <Maximize2 className="w-3.5 h-3.5 text-white" />
                        </div>
                      </div>
                      <div className="p-1 text-[9px] font-mono text-center text-slate-400 truncate bg-slate-900/60">
                        {f.parameters?.skeleton_pixels ?? 0} skel px
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Parameters Summary Badge List (Display Only) */}
              <div className="p-3 bg-slate-950/60 rounded-xl border border-slate-800/80 grid grid-cols-2 gap-2 text-[11px] font-mono">
                <div>
                  <span className="text-slate-400">Ridge Skeleton: </span>
                  <span className="font-bold text-teal-300">{dataB.median_parameters?.skeleton_pixels ?? 'N/A'} px</span>
                </div>
                <div>
                  <span className="text-slate-400">Groove Valleys: </span>
                  <span className="font-bold text-teal-300">{dataB.median_parameters?.groove_pixels ?? 'N/A'} px</span>
                </div>
                <div>
                  <span className="text-slate-400">Endpoints: </span>
                  <span className="font-bold text-slate-200">{dataB.median_parameters?.endpoints ?? 'N/A'}</span>
                </div>
                <div>
                  <span className="text-slate-400">Junctions: </span>
                  <span className="font-bold text-slate-200">{dataB.median_parameters?.junctions ?? 'N/A'}</span>
                </div>
                <div>
                  <span className="text-slate-400">Cycles: </span>
                  <span className="font-bold text-slate-200">{dataB.median_parameters?.cycles ?? 'N/A'}</span>
                </div>
                <div>
                  <span className="text-slate-400">Euler (χ): </span>
                  <span className="font-bold text-slate-200">{dataB.median_parameters?.euler_characteristic ?? 'N/A'}</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* 1-Click Both Videos Extract Bar */}
      <div className="flex justify-center">
        <button
          onClick={handleExtractBoth}
          disabled={loadingA || loadingB}
          className="px-6 py-2.5 rounded-xl font-semibold text-xs bg-slate-900 hover:bg-slate-800 border border-slate-700 text-slate-300 flex items-center gap-2 transition-colors disabled:opacity-50"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loadingA || loadingB ? 'animate-spin text-emerald-400' : ''}`} />
          Extract Both Videos &amp; Store in LocalStorage (1-Click)
        </button>
      </div>

      {/* ========================================================= */}
      {/* PRIMARY BIOMETRIC VERIFICATION CALL-TO-ACTION             */}
      {/* ========================================================= */}
      <div className="glass-panel p-6 rounded-2xl border border-slate-800 flex flex-col items-center gap-4 text-center bg-gradient-to-b from-slate-900/60 to-slate-950/80">
        <div className="max-w-xl">
          <h3 className="text-lg font-bold text-white flex items-center justify-center gap-2">
            <ShieldCheck className="w-5 h-5 text-emerald-400" />
            Biometric Match (SIFT + Minutiae)
          </h3>
          <p className="text-xs text-slate-400 mt-1">
            Compares 5×5 (25 pairwise) canonical frame combinations using SIFT/ORB RANSAC inlier ratio (Method A) and skeleton minutiae topology (Method B) against the calibrated EER threshold.
          </p>
        </div>

        <div className="flex items-center gap-3 flex-wrap justify-center">
          {/* Main Verification Button */}
          <button
            onClick={handleRunBiometricMatch}
            disabled={matchingBiometric}
            className={`px-8 py-4 rounded-2xl font-black text-sm tracking-wide uppercase flex items-center gap-3 transition-all shadow-xl ${
              matchingBiometric
                ? 'bg-emerald-600 text-slate-950 cursor-wait'
                : 'bg-gradient-to-r from-emerald-400 via-teal-400 to-cyan-400 hover:from-emerald-300 hover:to-cyan-300 text-slate-950 shadow-emerald-500/30 scale-100 hover:scale-[1.02]'
            }`}
          >
            {matchingBiometric ? (
              <>
                <RefreshCw className="w-5 h-5 animate-spin" />
                <span>Running 5×5 Biometric Verification...</span>
              </>
            ) : (
              <>
                <Crosshair className="w-5 h-5" />
                <span>Run Biometric Verification (SIFT + Minutiae)</span>
                <ArrowRight className="w-5 h-5" />
              </>
            )}
          </button>

          {/* Secondary / Optional: Cosine Similarity button */}
          <button
            onClick={handleCalculateCosineSimilarity}
            disabled={calculatingSim || !dataA || !dataB}
            className="px-4 py-3 rounded-xl font-semibold text-xs bg-slate-900 hover:bg-slate-800 border border-slate-700 text-slate-300 flex items-center gap-2 transition-colors disabled:opacity-40"
            title="Compute Cosine Similarity on extracted 10-parameter vectors"
          >
            <GitCompare className="w-4 h-4 text-teal-400" />
            <span>Cosine Vector Match</span>
          </button>
        </div>

        {matchError && (
          <div className="p-3 bg-rose-950/40 border border-rose-500/40 text-rose-300 rounded-xl text-xs font-mono max-w-lg">
            {matchError}
          </div>
        )}
      </div>

      {/* ========================================================= */}
      {/* BIOMETRIC MATCH RESULTS DISPLAY                           */}
      {/* ========================================================= */}
      {biometricMatchResult && (
        <div className="glass-panel p-6 rounded-2xl border border-emerald-500/40 bg-gradient-to-b from-slate-900/90 to-slate-950/95 flex flex-col gap-6 animate-in fade-in slide-in-from-bottom-4 duration-300">
          {/* Top Score & Verdict Banner */}
          <div className="flex flex-wrap items-center justify-between gap-6 pb-6 border-b border-slate-800">
            <div>
              <div className="flex items-center gap-2">
                <span className="text-xs font-mono uppercase tracking-wider text-slate-400">
                  Biometric Composite Verification Score
                </span>
                {biometricMatchResult.calibrated === false && (
                  <span className="text-[10px] px-2 py-0.5 rounded bg-amber-500/20 text-amber-300 font-bold border border-amber-500/30 font-mono">
                    UNCALIBRATED THRESHOLD
                  </span>
                )}
              </div>
              <div className="flex items-baseline gap-3 mt-1.5 flex-wrap">
                <span className="text-5xl font-black tracking-tight text-white font-mono">
                  {biometricMatchResult.composite_score !== undefined
                    ? (biometricMatchResult.composite_score * 100).toFixed(1)
                    : (biometricMatchResult.score !== undefined ? (biometricMatchResult.score * 100).toFixed(1) : 'N/A')}
                  %
                </span>
                <span className="text-xs text-slate-400 font-mono">
                  (Best Pair: {biometricMatchResult.best_score !== undefined ? (biometricMatchResult.best_score * 100).toFixed(1) : 'N/A'}% &bull; Median Pair: {biometricMatchResult.median_score !== undefined ? (biometricMatchResult.median_score * 100).toFixed(1) : 'N/A'}%)
                </span>
              </div>
            </div>

            {/* Verdict Badge */}
            <div className="flex flex-col items-end gap-1.5">
              <div className={`px-5 py-3 rounded-2xl border font-black text-sm flex items-center gap-2.5 shadow-lg ${
                biometricMatchResult.calibrated === false
                  ? 'bg-amber-500/20 border-amber-500/50 text-amber-300 shadow-amber-500/20'
                  : biometricMatchResult.is_match
                  ? 'bg-emerald-500/20 border-emerald-500/60 text-emerald-300 shadow-emerald-500/20'
                  : 'bg-rose-500/20 border-rose-500/60 text-rose-300 shadow-rose-500/20'
              }`}>
                {biometricMatchResult.calibrated === false ? (
                  <>
                    <AlertCircle className="w-5 h-5 text-amber-400" />
                    <span>UNCALIBRATED &bull; {biometricMatchResult.is_match ? 'PROVISIONAL MATCH' : 'PROVISIONAL NO MATCH'}</span>
                  </>
                ) : biometricMatchResult.is_match ? (
                  <>
                    <CheckCircle2 className="w-5 h-5 text-emerald-400" />
                    <span>MATCH (SAME CATTLE)</span>
                  </>
                ) : (
                  <>
                    <XCircle className="w-5 h-5 text-rose-400" />
                    <span>NO MATCH (DISTINCT CATTLE)</span>
                  </>
                )}
              </div>
              <span className="text-[11px] font-mono text-slate-400">
                Decision Threshold: {biometricMatchResult.threshold !== undefined && biometricMatchResult.threshold !== null
                  ? `${(biometricMatchResult.threshold * 100).toFixed(1)}%`
                  : (calibratedThreshold !== null ? `${(calibratedThreshold * 100).toFixed(1)}%` : 'None (Uncalibrated)')}
              </span>
            </div>
          </div>

          {/* Methods Breakdown: Method A and Method B Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Method A Card */}
            <div className="p-4 rounded-xl border border-slate-800 bg-slate-950/70 flex flex-col gap-3">
              <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                <span className="text-xs font-bold text-slate-200 flex items-center gap-1.5">
                  <Crosshair className="w-4 h-4 text-emerald-400" />
                  Method A: SIFT / ORB + RANSAC
                </span>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-500/15 text-emerald-300 border border-emerald-500/30 font-bold">
                  Score: {biometricMatchResult.method_a?.score !== undefined ? (biometricMatchResult.method_a.score * 100).toFixed(1) : (biometricMatchResult.inlier_ratio !== undefined ? (biometricMatchResult.inlier_ratio * 100).toFixed(1) : 'N/A')}%
                </span>
              </div>
              <div className="grid grid-cols-2 gap-3 text-xs font-mono">
                <div>
                  <span className="text-slate-400 block text-[11px]">RANSAC Inliers</span>
                  <span className="text-base font-bold text-emerald-300">
                    {biometricMatchResult.inlier_count ?? biometricMatchResult.method_a?.inlier_count ?? 0}
                  </span>
                  <span className="text-[10px] text-slate-400 ml-1">inliers</span>
                </div>
                <div>
                  <span className="text-slate-400 block text-[11px]">Inlier Ratio</span>
                  <span className="text-base font-bold text-emerald-300">
                    {biometricMatchResult.inlier_ratio !== undefined
                      ? (biometricMatchResult.inlier_ratio * 100).toFixed(1)
                      : (biometricMatchResult.method_a?.inlier_ratio !== undefined ? (biometricMatchResult.method_a.inlier_ratio * 100).toFixed(1) : 'N/A')}
                    %
                  </span>
                </div>
              </div>
            </div>

            {/* Method B Card */}
            <div className="p-4 rounded-xl border border-slate-800 bg-slate-950/70 flex flex-col gap-3">
              <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                <span className="text-xs font-bold text-slate-200 flex items-center gap-1.5">
                  <Network className="w-4 h-4 text-teal-400" />
                  Method B: Skeleton Minutiae Topology
                </span>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-teal-500/15 text-teal-300 border border-teal-500/30 font-bold">
                  Score: {biometricMatchResult.method_b?.score !== undefined ? (biometricMatchResult.method_b.score * 100).toFixed(1) : (biometricMatchResult.matched_minutiae_fraction !== undefined ? (biometricMatchResult.matched_minutiae_fraction * 100).toFixed(1) : 'N/A')}%
                </span>
              </div>
              <div className="grid grid-cols-2 gap-3 text-xs font-mono">
                <div>
                  <span className="text-slate-400 block text-[11px]">Matched Minutiae</span>
                  <span className="text-base font-bold text-teal-300">
                    {biometricMatchResult.method_b?.matched_count ?? biometricMatchResult.matched_minutiae_count ?? 0}
                  </span>
                  <span className="text-[10px] text-slate-400 ml-1">pairs</span>
                </div>
                <div>
                  <span className="text-slate-400 block text-[11px]">Matched Fraction</span>
                  <span className="text-base font-bold text-teal-300">
                    {biometricMatchResult.matched_minutiae_fraction !== undefined
                      ? (biometricMatchResult.matched_minutiae_fraction * 100).toFixed(1)
                      : (biometricMatchResult.method_b?.matched_fraction !== undefined ? (biometricMatchResult.method_b.matched_fraction * 100).toFixed(1) : 'N/A')}
                    %
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Keypoint Match Visualization Image (Side-by-Side Match Viewer) */}
          {biometricMatchResult.visual_url && (
            <div className="flex flex-col gap-2 pt-2">
              <div className="flex items-center justify-between">
                <h4 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
                  <Eye className="w-4 h-4 text-emerald-400" />
                  Keypoint Match Alignment Visualization (Best Frame Pair)
                </h4>
                <button
                  onClick={() => setVisualModalOpen(true)}
                  className="flex items-center gap-1 px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-mono transition-colors"
                >
                  <Maximize2 className="w-3.5 h-3.5" />
                  Full Size
                </button>
              </div>

              <div
                onClick={() => setVisualModalOpen(true)}
                className="group relative rounded-xl border border-slate-800 overflow-hidden bg-slate-950 cursor-pointer hover:border-emerald-500/80 transition-all max-h-96 flex items-center justify-center"
              >
                <img
                  src={biometricMatchResult.visual_url}
                  alt="Biometric Keypoint Match Visualization"
                  className="w-full h-full object-contain max-h-96 group-hover:scale-[1.01] transition-transform"
                />
                <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity">
                  <div className="px-4 py-2 rounded-xl bg-slate-900/90 border border-slate-700 text-white font-semibold text-xs flex items-center gap-2">
                    <Maximize2 className="w-4 h-4 text-emerald-400" />
                    Click to Open Full-Size Alignment View
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* 10-Parameter Direct Comparison Table (Display Only) */}
          {dataA && dataB && (
            <div className="flex flex-col gap-2 pt-2 border-t border-slate-800">
              <h4 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
                <Layers className="w-4 h-4 text-emerald-400" />
                10-Parameter Muzzle Groove Characteristics (Display Reference)
              </h4>

              <div className="overflow-x-auto rounded-xl border border-slate-800">
                <table className="w-full text-xs font-mono text-left">
                  <thead className="bg-slate-900 text-slate-400 text-[11px] border-b border-slate-800">
                    <tr>
                      <th className="py-2.5 px-3">Parameter Description</th>
                      <th className="py-2.5 px-3 text-emerald-400">Video A (Median)</th>
                      <th className="py-2.5 px-3 text-teal-400">Video B (Median)</th>
                      <th className="py-2.5 px-3 text-right">Delta (&Delta;%)</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 bg-slate-950/40">
                    {Object.keys(FEATURE_LABELS).map((key) => {
                      const valA = dataA.median_parameters?.[key] ?? 0;
                      const valB = dataB.median_parameters?.[key] ?? 0;
                      const unit = FEATURE_UNITS[key] || '';
                      const delta = valA !== 0 ? Math.abs((valB - valA) / valA) * 100 : 0;
                      return (
                        <tr key={key} className="hover:bg-slate-900/40 transition-colors">
                          <td className="py-2 px-3 text-slate-300 font-sans font-medium">
                            {FEATURE_LABELS[key]}
                          </td>
                          <td className="py-2 px-3 text-emerald-300 font-bold">
                            {valA} {unit}
                          </td>
                          <td className="py-2 px-3 text-teal-300 font-bold">
                            {valB} {unit}
                          </td>
                          <td className={`py-2 px-3 text-right font-bold ${
                            delta <= 15 ? 'text-emerald-400' : delta <= 35 ? 'text-amber-400' : 'text-slate-400'
                          }`}>
                            {delta.toFixed(1)}%
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ========================================================= */}
      {/* SECONDARY COSINE RESULTS ACCORDION (IF CALCULATED)        */}
      {/* ========================================================= */}
      {similarityResult && (
        <div className="glass-panel p-5 rounded-2xl border border-slate-800 flex flex-col gap-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold text-slate-300 flex items-center gap-2">
              <GitCompare className="w-4 h-4 text-teal-400" />
              Secondary Metric: Cosine Vector Similarity ({similarityResult.cosine_similarity}%)
            </span>
            <button
              onClick={() => setShowCosineDetails(!showCosineDetails)}
              className="text-xs text-teal-400 hover:text-teal-300 font-mono"
            >
              {showCosineDetails ? 'Hide details' : 'Show details'}
            </button>
          </div>
          {showCosineDetails && (
            <div className="p-3 bg-slate-950/60 rounded-xl border border-slate-800 text-xs font-mono text-slate-400 flex flex-col gap-1">
              <div>Normalized Cosine Similarity: <strong className="text-emerald-300">{similarityResult.cosine_similarity}%</strong></div>
              <div>Raw Unnormalized Cosine: <strong className="text-teal-300">{similarityResult.raw_cosine_similarity}%</strong></div>
              <div>Verdict: <strong className="text-slate-200">{similarityResult.verdict}</strong></div>
            </div>
          )}
        </div>
      )}

      {/* ========================================================= */}
      {/* LOCALSTORAGE MODAL                                        */}
      {/* ========================================================= */}
      {showLocalStorageModal && (
        <div className="fixed inset-0 z-50 bg-slate-950/80 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-slate-900 border border-slate-700 rounded-2xl max-w-2xl w-full p-6 max-h-[85vh] flex flex-col gap-4 shadow-2xl">
            <div className="flex items-center justify-between pb-3 border-b border-slate-800">
              <div className="flex items-center gap-2">
                <Database className="w-5 h-5 text-teal-400" />
                <h3 className="font-bold text-white text-base">Browser LocalStorage Parameters</h3>
              </div>
              <button
                onClick={() => setShowLocalStorageModal(false)}
                className="text-slate-400 hover:text-white p-1 rounded-lg"
              >
                ✕
              </button>
            </div>

            <div className="overflow-y-auto flex flex-col gap-4 text-xs font-mono pr-2">
              <div>
                <span className="text-emerald-400 font-bold block mb-1">Key: 'muzzle_video_a_params'</span>
                <pre className="p-3 bg-slate-950 rounded-xl border border-slate-800 overflow-x-auto text-[11px] text-slate-300">
                  {storageData.a ? JSON.stringify(JSON.parse(storageData.a), null, 2) : 'No data in LocalStorage'}
                </pre>
              </div>

              <div>
                <span className="text-teal-400 font-bold block mb-1">Key: 'muzzle_video_b_params'</span>
                <pre className="p-3 bg-slate-950 rounded-xl border border-slate-800 overflow-x-auto text-[11px] text-slate-300">
                  {storageData.b ? JSON.stringify(JSON.parse(storageData.b), null, 2) : 'No data in LocalStorage'}
                </pre>
              </div>
            </div>

            <div className="flex justify-end pt-2 border-t border-slate-800">
              <button
                onClick={() => setShowLocalStorageModal(false)}
                className="px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-white font-semibold text-xs"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ========================================================= */}
      {/* FRAME PREVIEW MODAL                                       */}
      {/* ========================================================= */}
      {previewImage && (
        <div className="fixed inset-0 z-50 bg-slate-950/90 backdrop-blur-md flex items-center justify-center p-4">
          <div className="bg-slate-900 border border-slate-700 rounded-2xl max-w-xl w-full p-5 flex flex-col gap-4 shadow-2xl">
            <div className="flex items-center justify-between pb-2 border-b border-slate-800">
              <span className="font-bold text-white text-sm">{previewImage.title}</span>
              <button
                onClick={() => setPreviewImage(null)}
                className="text-slate-400 hover:text-white p-1 rounded-lg"
              >
                ✕
              </button>
            </div>

            <div className="aspect-square w-full rounded-xl overflow-hidden bg-slate-950 border border-slate-800 flex items-center justify-center">
              <img
                src={previewImage.url}
                alt={previewImage.title}
                className="w-full h-full object-contain"
              />
            </div>

            {previewImage.params && (
              <div className="grid grid-cols-2 gap-2 p-3 bg-slate-950 rounded-xl border border-slate-800 text-[11px] font-mono">
                <div>Skeleton: <span className="text-emerald-400 font-bold">{previewImage.params.skeleton_pixels} px</span></div>
                <div>Groove: <span className="text-teal-400 font-bold">{previewImage.params.groove_pixels} px</span></div>
                <div>Endpoints: <span className="text-slate-200 font-bold">{previewImage.params.endpoints}</span></div>
                <div>Junctions: <span className="text-slate-200 font-bold">{previewImage.params.junctions}</span></div>
                <div>Cycles: <span className="text-slate-200 font-bold">{previewImage.params.cycles}</span></div>
                <div>Euler (χ): <span className="text-slate-200 font-bold">{previewImage.params.euler_characteristic}</span></div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ========================================================= */}
      {/* FULL-SIZE KEYPOINT MATCH VISUALIZATION MODAL               */}
      {/* ========================================================= */}
      {visualModalOpen && biometricMatchResult?.visual_url && (
        <div className="fixed inset-0 z-50 bg-slate-950/95 backdrop-blur-md flex items-center justify-center p-4">
          <div className="bg-slate-900 border border-slate-700 rounded-2xl max-w-5xl w-full p-5 flex flex-col gap-4 shadow-2xl max-h-[95vh]">
            <div className="flex items-center justify-between pb-2 border-b border-slate-800">
              <div className="flex items-center gap-2">
                <Crosshair className="w-5 h-5 text-emerald-400" />
                <span className="font-bold text-white text-base">
                  SIFT / ORB Keypoint Inlier Alignment &bull; Best Frame Pair
                </span>
                <span className="text-xs font-mono text-emerald-300 px-2 py-0.5 rounded bg-emerald-500/20 border border-emerald-500/30">
                  {biometricMatchResult.inlier_count ?? biometricMatchResult.method_a?.inlier_count ?? 0} Inliers
                </span>
              </div>
              <button
                onClick={() => setVisualModalOpen(false)}
                className="text-slate-400 hover:text-white p-1 rounded-lg text-lg"
              >
                ✕
              </button>
            </div>

            <div className="w-full flex-1 min-h-[300px] overflow-auto rounded-xl bg-slate-950 border border-slate-800 flex items-center justify-center p-2">
              <img
                src={biometricMatchResult.visual_url}
                alt="Full Keypoint Match Alignment"
                className="max-w-full max-h-[75vh] object-contain rounded-lg"
              />
            </div>

            <div className="flex items-center justify-between text-xs font-mono text-slate-400 pt-2 border-t border-slate-800">
              <div>
                Composite Score: <strong className="text-white">{((biometricMatchResult.composite_score ?? 0) * 100).toFixed(1)}%</strong> | Inlier Ratio: <strong className="text-emerald-300">{((biometricMatchResult.inlier_ratio ?? 0) * 100).toFixed(1)}%</strong> | Minutiae: <strong className="text-teal-300">{((biometricMatchResult.matched_minutiae_fraction ?? 0) * 100).toFixed(1)}%</strong>
              </div>
              <button
                onClick={() => setVisualModalOpen(false)}
                className="px-4 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-white font-semibold text-xs"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
