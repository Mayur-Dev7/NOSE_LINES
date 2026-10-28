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
  Maximize2
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

  // Filter & Threshold Configuration (User Requested)
  const [filterMethod, setFilterMethod] = useState('frangi'); // 'frangi' (Frangi Hessian) | 'blackhat'
  const [thresholdMethod, setThresholdMethod] = useState('adaptive'); // 'adaptive' (Adaptive Gaussian) | 'otsu' (Standard)
  const [adaptiveBlockSize, setAdaptiveBlockSize] = useState(15);
  const [adaptiveC, setAdaptiveC] = useState(2);
  const [excludeNostrils, setExcludeNostrils] = useState(true);

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

  // Full-size image modal
  const [previewImage, setPreviewImage] = useState(null);

  // Similarity Calculation Result
  const [similarityResult, setSimilarityResult] = useState(null);
  const [calculatingSim, setCalculatingSim] = useState(false);

  // LocalStorage inspection modal
  const [showLocalStorageModal, setShowLocalStorageModal] = useState(false);
  const [storageData, setStorageData] = useState({ a: null, b: null });

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
    setSimilarityResult(null);
    setStorageData({ a: null, b: null });
  };

  // Extract 5 images & groove lines for Video A
  const handleExtractVideoA = async () => {
    setLoadingA(true);
    setSimilarityResult(null);
    setExtractLogA('Running Mask R-CNN detector (crop11dec.h5) on video frames...');
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
        }),
      });
      const data = await res.json();
      if (data.success) {
        setDataA(data);
        saveToLocalStorage('muzzle_video_a_params', data);
        setExtractLogA(`Completed! 5th-stage skeleton lines extracted (${data.median_parameters.skeleton_pixels} px, ${data.median_parameters.endpoints} endpoints). Stored in LocalStorage.`);
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
    setLoadingB(true);
    setSimilarityResult(null);
    setExtractLogB('Running Mask R-CNN detector (crop11dec.h5) on video frames...');
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
        }),
      });
      const data = await res.json();
      if (data.success) {
        setDataB(data);
        saveToLocalStorage('muzzle_video_b_params', data);
        setExtractLogB(`Completed! 5th-stage skeleton lines extracted (${data.median_parameters.skeleton_pixels} px, ${data.median_parameters.endpoints} endpoints). Stored in LocalStorage.`);
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

  // Extract both videos
  const handleExtractBoth = async () => {
    await handleExtractVideoA();
    await handleExtractVideoB();
  };

  // Calculate Cosine Similarity from LocalStorage
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

  return (
    <div className="flex flex-col gap-6 text-slate-100 max-w-7xl mx-auto w-full pb-16">
      {/* Header Bar */}
      <div className="glass-panel p-6 rounded-2xl border border-slate-800 flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
              <GitCompare className="w-6 h-6 text-emerald-400" />
              2-Video Muzzle Groove &amp; Cosine Similarity Matcher
            </h1>
            <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 font-mono">
              Mask R-CNN &bull; 5th-Image Skeleton Lines &bull; Cosine
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Detects muzzles with Mask R-CNN, upscales 2x Lanczos-4, grayscale CLAHE enhances, extracts 5th-image skeleton lines &amp; coordinates, stores parameters in LocalStorage, and calculates Cosine Similarity.
          </p>
        </div>

        {/* Global Action Buttons */}
        <div className="flex items-center gap-2">
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
            Inspect LocalStorage
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

      {/* ========================================================= */}
      {/* THRESHOLD & FILTER CONFIGURATION OPTIONS (USER REQUESTED)  */}
      {/* ========================================================= */}
      <div className="glass-panel p-4 rounded-xl border border-slate-800 flex flex-col gap-3">
        <div className="flex items-center justify-between border-b border-slate-800/80 pb-2">
          <div className="flex items-center gap-2 text-xs font-bold text-slate-200">
            <Sliders className="w-4 h-4 text-emerald-400" />
            <span>Groove Extraction &amp; Threshold Options (To Produce 5th Image Skeleton Lines):</span>
          </div>
          <span className="text-[10px] text-slate-400 font-mono">
            Pipeline: Crop &rarr; 2x Lanczos-4 &rarr; CLAHE &rarr; Filter &rarr; Threshold &rarr; 5th Image Lines
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-xs">
          {/* Option 1: Filter Method */}
          <div className="flex flex-col gap-1.5">
            <label className="text-slate-400 font-semibold flex items-center justify-between">
              <span>Filter Method:</span>
              <span className="text-[10px] font-mono text-emerald-400 font-bold">{filterMethod === 'frangi' ? 'Frangi Hessian' : 'Black-Hat'}</span>
            </label>
            <select
              value={filterMethod}
              onChange={(e) => setFilterMethod(e.target.value)}
              className="bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-slate-200 focus:outline-hidden focus:border-emerald-500 font-medium"
            >
              <option value="frangi">Frangi Hessian (Eigenvalues &amp; Black Ridges)</option>
              <option value="blackhat">Multi-Scale Black-Hat Morphology (9, 15, 21)</option>
            </select>
          </div>

          {/* Option 2: Threshold Method */}
          <div className="flex flex-col gap-1.5">
            <label className="text-slate-400 font-semibold flex items-center justify-between">
              <span>Threshold Method:</span>
              <span className="text-[10px] font-mono text-emerald-400 font-bold">{thresholdMethod === 'adaptive' ? 'Adaptive Gaussian' : 'Standard Otsu'}</span>
            </label>
            <select
              value={thresholdMethod}
              onChange={(e) => setThresholdMethod(e.target.value)}
              className="bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-slate-200 focus:outline-hidden focus:border-emerald-500 font-medium"
            >
              <option value="adaptive">Adaptive Gaussian (cv2.ADAPTIVE_THRESH_GAUSSIAN_C)</option>
              <option value="otsu">Standard / Otsu (Global Minimum Within-Class Variance)</option>
            </select>
          </div>

          {/* Option 3: Adaptive Gaussian Block Size & Constant C */}
          {thresholdMethod === 'adaptive' ? (
            <div className="flex items-center gap-3">
              <div className="flex-1 flex flex-col gap-1.5">
                <label className="text-slate-400 font-semibold flex items-center justify-between">
                  <span>Block Size:</span>
                  <span className="text-[10px] font-mono text-slate-300">{adaptiveBlockSize} px</span>
                </label>
                <select
                  value={adaptiveBlockSize}
                  onChange={(e) => setAdaptiveBlockSize(Number(e.target.value))}
                  className="bg-slate-900 border border-slate-700 rounded-lg px-2 py-1.5 text-slate-200 focus:outline-hidden focus:border-emerald-500 font-mono"
                >
                  <option value={9}>9 px (Fine)</option>
                  <option value={15}>15 px (Balanced)</option>
                  <option value={21}>21 px (Robust)</option>
                  <option value={31}>31 px (Broad)</option>
                </select>
              </div>

              <div className="flex-1 flex flex-col gap-1.5">
                <label className="text-slate-400 font-semibold flex items-center justify-between">
                  <span>Const C:</span>
                  <span className="text-[10px] font-mono text-slate-300">+{adaptiveC}</span>
                </label>
                <select
                  value={adaptiveC}
                  onChange={(e) => setAdaptiveC(Number(e.target.value))}
                  className="bg-slate-900 border border-slate-700 rounded-lg px-2 py-1.5 text-slate-200 focus:outline-hidden focus:border-emerald-500 font-mono"
                >
                  <option value={1}>1</option>
                  <option value={2}>2 (Default)</option>
                  <option value={3}>3</option>
                  <option value={5}>5 (High contrast)</option>
                </select>
              </div>
            </div>
          ) : (
            <div className="flex flex-col gap-1.5">
              <label className="text-slate-400 font-semibold">Standard Threshold:</label>
              <div className="px-3 py-1.5 rounded-lg bg-slate-900/60 border border-slate-800 text-slate-400 text-xs">
                Otsu auto-bimodal thresholding active
              </div>
            </div>
          )}

          {/* Option 4: Nostril Cavity Suppression */}
          <div className="flex flex-col gap-1.5 justify-center">
            <label className="text-slate-400 font-semibold">Nostril Cavity Suppression:</label>
            <label className="flex items-center gap-2 cursor-pointer text-slate-300 select-none py-1">
              <input
                type="checkbox"
                checked={excludeNostrils}
                onChange={(e) => setExcludeNostrils(e.target.checked)}
                className="w-4 h-4 rounded text-emerald-500 focus:ring-emerald-400 bg-slate-900 border-slate-700"
              />
              <span>Mask out dark nostril cavities</span>
            </label>
          </div>
        </div>
      </div>

      {/* Preset Quick Tests Bar */}
      <div className="glass-panel p-3.5 rounded-xl border border-slate-800 flex flex-wrap items-center justify-between gap-3 text-xs">
        <div className="flex items-center gap-2 text-slate-300 font-semibold">
          <Sparkles className="w-4 h-4 text-emerald-400" />
          <span>Quick Video Presets:</span>
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
            🔍 Cross-Video (Video 1 vs Video 2)
          </button>
          <button
            onClick={() => {
              setVideoA('/home/adis_mayur/F_CODES/muzzleVideo2.mp4');
              setVideoB('/home/adis_mayur/F_CODES/muzzleVideo3.mp4');
            }}
            className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 font-medium transition-colors"
          >
            🔍 Cross-Video (Video 2 vs Video 3)
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
                <span className="text-[11px] text-slate-400">Extracts 5 frames &bull; computes line parameters</span>
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
            disabled={loadingA}
            className={`w-full py-2.5 px-4 rounded-xl font-bold text-xs flex items-center justify-center gap-2 transition-all shadow-md ${
              loadingA
                ? 'bg-slate-800 text-slate-500 cursor-not-allowed'
                : 'bg-gradient-to-r from-emerald-500 to-teal-500 hover:from-emerald-400 hover:to-teal-400 text-slate-950 shadow-emerald-500/20'
            }`}
          >
            {loadingA ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin text-emerald-400" />
                <span>Running Mask R-CNN &amp; Extracting 5th-Stage Lines...</span>
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
                        {f.parameters.skeleton_pixels} skel px
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Parameters Summary Badge List */}
              <div className="p-3 bg-slate-950/60 rounded-xl border border-slate-800/80 grid grid-cols-2 gap-2 text-[11px] font-mono">
                <div>
                  <span className="text-slate-400">Ridge Skeleton: </span>
                  <span className="font-bold text-teal-300">{dataA.median_parameters.skeleton_pixels} px</span>
                </div>
                <div>
                  <span className="text-slate-400">Groove Valleys: </span>
                  <span className="font-bold text-teal-300">{dataA.median_parameters.groove_pixels} px</span>
                </div>
                <div>
                  <span className="text-slate-400">Endpoints: </span>
                  <span className="font-bold text-slate-200">{dataA.median_parameters.endpoints}</span>
                </div>
                <div>
                  <span className="text-slate-400">Junctions: </span>
                  <span className="font-bold text-slate-200">{dataA.median_parameters.junctions}</span>
                </div>
                <div>
                  <span className="text-slate-400">Cycles: </span>
                  <span className="font-bold text-slate-200">{dataA.median_parameters.cycles}</span>
                </div>
                <div>
                  <span className="text-slate-400">Euler (χ): </span>
                  <span className="font-bold text-slate-200">{dataA.median_parameters.euler_characteristic}</span>
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
                <span className="text-[11px] text-slate-400">Extracts 5 frames &bull; computes line parameters</span>
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
            disabled={loadingB}
            className={`w-full py-2.5 px-4 rounded-xl font-bold text-xs flex items-center justify-center gap-2 transition-all shadow-md ${
              loadingB
                ? 'bg-slate-800 text-slate-500 cursor-not-allowed'
                : 'bg-gradient-to-r from-teal-500 to-cyan-500 hover:from-teal-400 hover:to-cyan-400 text-slate-950 shadow-teal-500/20'
            }`}
          >
            {loadingB ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin text-teal-400" />
                <span>Running Mask R-CNN &amp; Extracting 5th-Stage Lines...</span>
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
                        {f.parameters.skeleton_pixels} skel px
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Parameters Summary Badge List */}
              <div className="p-3 bg-slate-950/60 rounded-xl border border-slate-800/80 grid grid-cols-2 gap-2 text-[11px] font-mono">
                <div>
                  <span className="text-slate-400">Ridge Skeleton: </span>
                  <span className="font-bold text-teal-300">{dataB.median_parameters.skeleton_pixels} px</span>
                </div>
                <div>
                  <span className="text-slate-400">Groove Valleys: </span>
                  <span className="font-bold text-teal-300">{dataB.median_parameters.groove_pixels} px</span>
                </div>
                <div>
                  <span className="text-slate-400">Endpoints: </span>
                  <span className="font-bold text-slate-200">{dataB.median_parameters.endpoints}</span>
                </div>
                <div>
                  <span className="text-slate-400">Junctions: </span>
                  <span className="font-bold text-slate-200">{dataB.median_parameters.junctions}</span>
                </div>
                <div>
                  <span className="text-slate-400">Cycles: </span>
                  <span className="font-bold text-slate-200">{dataB.median_parameters.cycles}</span>
                </div>
                <div>
                  <span className="text-slate-400">Euler (χ): </span>
                  <span className="font-bold text-slate-200">{dataB.median_parameters.euler_characteristic}</span>
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
          Extract Both Videos with Selected Threshold &amp; Store in LocalStorage (1-Click)
        </button>
      </div>

      {/* ========================================================= */}
      {/* BIG PRIMARY "CALCULATE SIMILARITY" BUTTON                 */}
      {/* ========================================================= */}
      <div className="glass-panel p-6 rounded-2xl border border-slate-800 flex flex-col items-center gap-4 text-center">
        <div className="max-w-xl">
          <h3 className="text-lg font-bold text-white flex items-center justify-center gap-2">
            <GitCompare className="w-5 h-5 text-emerald-400" />
            Compute Cosine Similarity from LocalStorage
          </h3>
          <p className="text-xs text-slate-400 mt-1">
            Reads the 10-parameter feature vectors for Video A and Video B directly from your browser's <code className="text-emerald-300 font-mono">localStorage</code> and computes Cosine Similarity.
          </p>
        </div>

        <button
          onClick={handleCalculateCosineSimilarity}
          disabled={calculatingSim || !dataA || !dataB}
          className={`px-8 py-4 rounded-2xl font-black text-sm tracking-wide uppercase flex items-center gap-3 transition-all shadow-xl ${
            !dataA || !dataB
              ? 'bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700'
              : calculatingSim
              ? 'bg-emerald-600 text-slate-950 cursor-wait'
              : 'bg-gradient-to-r from-emerald-400 via-teal-400 to-cyan-400 hover:from-emerald-300 hover:to-cyan-300 text-slate-950 shadow-emerald-500/30 scale-100 hover:scale-[1.02]'
          }`}
        >
          {calculatingSim ? (
            <>
              <RefreshCw className="w-5 h-5 animate-spin" />
              <span>Computing Cosine Similarity...</span>
            </>
          ) : (
            <>
              <Sparkles className="w-5 h-5" />
              <span>Calculate Cosine Similarity</span>
              <ArrowRight className="w-5 h-5" />
            </>
          )}
        </button>

        {(!dataA || !dataB) && (
          <span className="text-[11px] text-amber-400/90 font-mono">
            &bull; Extract 5 images for both Video A and Video B first to enable comparison
          </span>
        )}
      </div>

      {/* ========================================================= */}
      {/* SIMILARITY RESULTS DISPLAY                                */}
      {/* ========================================================= */}
      {similarityResult && (
        <div className="glass-panel p-6 rounded-2xl border border-emerald-500/40 bg-gradient-to-b from-slate-900/90 to-slate-950/90 flex flex-col gap-6 animate-in fade-in slide-in-from-bottom-4 duration-300">
          {/* Top Score Banner */}
          <div className="flex flex-wrap items-center justify-between gap-4 pb-6 border-b border-slate-800">
            <div>
              <span className="text-xs font-mono uppercase tracking-wider text-slate-400">
                1:1 Biometric Cosine Similarity Score
              </span>
              <div className="flex items-baseline gap-3 mt-1">
                <span className="text-5xl font-black tracking-tight text-white font-mono">
                  {similarityResult.cosine_similarity}%
                </span>
                <span className="text-xs text-slate-400 font-mono">
                  (Raw unnormalized: {similarityResult.raw_cosine_similarity}%)
                </span>
              </div>
            </div>

            {/* Verdict Badge */}
            <div className={`px-5 py-3 rounded-2xl border font-bold text-sm flex items-center gap-2.5 ${
              similarityResult.verdict_color === 'emerald'
                ? 'bg-emerald-500/15 border-emerald-500/50 text-emerald-300 shadow-lg shadow-emerald-500/20'
                : similarityResult.verdict_color === 'teal'
                ? 'bg-teal-500/15 border-teal-500/50 text-teal-300'
                : similarityResult.verdict_color === 'amber'
                ? 'bg-amber-500/15 border-amber-500/50 text-amber-300'
                : 'bg-rose-500/15 border-rose-500/50 text-rose-300'
            }`}>
              {similarityResult.is_match ? (
                <CheckCircle2 className="w-5 h-5" />
              ) : (
                <XCircle className="w-5 h-5" />
              )}
              <span>{similarityResult.verdict}</span>
            </div>
          </div>

          {/* 10-Parameter Direct Comparison Table */}
          <div className="flex flex-col gap-2">
            <h4 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
              <Layers className="w-4 h-4 text-emerald-400" />
              10-Parameter Groove Minutiae Comparison (From 5th-Image Lines)
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
                    const valA = similarityResult.paramsA[key] ?? 0;
                    const valB = similarityResult.paramsB[key] ?? 0;
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
      {/* FULL PREVIEW MODAL                                        */}
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
    </div>
  );
}
