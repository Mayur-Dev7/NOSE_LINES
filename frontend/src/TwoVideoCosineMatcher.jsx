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
  Eye
} from 'lucide-react';

const FEATURE_LABELS = {
  groove_pixels: 'Groove Pixels (Valleys)',
  skeleton_pixels: 'Ridge Skeleton (1-px)',
  groove_coverage_pct: 'Groove Area Coverage',
  endpoints: 'Ridge Endpoints (Degree 1)',
  junctions: 'Ridge Junctions (Degree >= 3)',
  connected_components: 'Connected Groove Clusters',
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
  const [videoB, setVideoB] = useState('/home/adis_mayur/F_CODES/muzzleVideo.mp4');

  // Loading states
  const [loadingA, setLoadingA] = useState(false);
  const [loadingB, setLoadingB] = useState(false);

  // Extracted 5-frame results
  const [dataA, setDataA] = useState(null);
  const [dataB, setDataB] = useState(null);

  // Active view toggle on the 5 cards: 'overlay' | 'crop' | 'skeleton'
  const [viewTypeA, setViewTypeA] = useState('overlay');
  const [viewTypeB, setViewTypeB] = useState('overlay');

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
      // Refresh modal state
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
    try {
      const res = await fetch('/api/video/extract-five-muzzle-parameters', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_path: videoA }),
      });
      const data = await res.json();
      if (data.success) {
        setDataA(data);
        saveToLocalStorage('muzzle_video_a_params', data);
      } else {
        alert(data.detail || 'Extraction failed for Video A');
      }
    } catch (err) {
      alert('Failed to connect to backend: ' + err.message);
    } finally {
      setLoadingA(false);
    }
  };

  // Extract 5 images & groove lines for Video B
  const handleExtractVideoB = async () => {
    setLoadingB(true);
    setSimilarityResult(null);
    try {
      const res = await fetch('/api/video/extract-five-muzzle-parameters', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_path: videoB }),
      });
      const data = await res.json();
      if (data.success) {
        setDataB(data);
        saveToLocalStorage('muzzle_video_b_params', data);
      } else {
        alert(data.detail || 'Extraction failed for Video B');
      }
    } catch (err) {
      alert('Failed to connect to backend: ' + err.message);
    } finally {
      setLoadingB(false);
    }
  };

  // Extract both videos with 1 click
  const handleExtractBoth = async () => {
    await handleExtractVideoA();
    await handleExtractVideoB();
  };

  // Calculate Cosine Similarity from LocalStorage
  const handleCalculateCosineSimilarity = async () => {
    // Read directly from LocalStorage to strictly fulfill user requirement
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

  return (
    <div className="flex flex-col gap-6 text-slate-100 max-w-7xl mx-auto w-full pb-16">
      {/* High-level Clean Title Bar */}
      <div className="glass-panel p-6 rounded-2xl border border-slate-800 flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
              <GitCompare className="w-6 h-6 text-emerald-400" />
              2-Video Muzzle Groove &amp; Cosine Similarity Matcher
            </h1>
            <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 font-mono">
              5-Image Batch &bull; LocalStorage &bull; Cosine
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Extracts 5 muzzle images per video, detects dark groove lines, stores parameters in LocalStorage, and calculates Cosine Similarity.
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
            className={`w-full py-2.5 px-4 rounded-xl text-sm font-bold flex items-center justify-center gap-2 transition-all ${
              loadingA
                ? 'bg-slate-800 text-slate-400 cursor-not-allowed'
                : 'bg-emerald-500 hover:bg-emerald-400 text-slate-950 shadow-md shadow-emerald-500/20'
            }`}
          >
            {loadingA ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin" />
                Extracting 5 Frames &amp; Groove Lines...
              </>
            ) : (
              <>
                <ScanLine className="w-4 h-4" />
                {dataA ? 'Re-Extract 5 Images (Video A)' : 'Extract 5 Images & Store in LocalStorage'}
              </>
            )}
          </button>

          {/* 5 Extracted Images Strip */}
          {dataA && (
            <div className="flex flex-col gap-3 pt-2">
              <div className="flex items-center justify-between text-xs text-slate-400">
                <span className="font-semibold text-slate-200">5 Extracted Muzzle Images:</span>
                <div className="flex items-center gap-1 bg-slate-900 p-0.5 rounded-lg border border-slate-800 text-[10px]">
                  <button
                    onClick={() => setViewTypeA('overlay')}
                    className={`px-2 py-0.5 rounded ${viewTypeA === 'overlay' ? 'bg-slate-800 text-emerald-300 font-bold' : 'text-slate-400'}`}
                  >
                    Groove Lines
                  </button>
                  <button
                    onClick={() => setViewTypeA('crop')}
                    className={`px-2 py-0.5 rounded ${viewTypeA === 'crop' ? 'bg-slate-800 text-emerald-300 font-bold' : 'text-slate-400'}`}
                  >
                    Raw Crop
                  </button>
                  <button
                    onClick={() => setViewTypeA('skeleton')}
                    className={`px-2 py-0.5 rounded ${viewTypeA === 'skeleton' ? 'bg-slate-800 text-emerald-300 font-bold' : 'text-slate-400'}`}
                  >
                    1-px Skel
                  </button>
                </div>
              </div>

              {/* 5 Images Grid */}
              <div className="grid grid-cols-5 gap-2">
                {dataA.frames.map((f, i) => {
                  const imgUrl = viewTypeA === 'crop' ? f.crop_url : viewTypeA === 'skeleton' ? f.skeleton_url : f.overlay_url;
                  return (
                    <div key={i} className="flex flex-col rounded-lg border border-slate-800 overflow-hidden bg-slate-950">
                      <div className="aspect-square relative overflow-hidden bg-slate-950">
                        <img
                          src={imgUrl}
                          alt={`Frame ${f.frame_number}`}
                          className="w-full h-full object-cover"
                        />
                        <span className="absolute top-1 left-1 text-[9px] font-mono px-1 rounded bg-slate-950/80 text-slate-300">
                          #{f.frame_number}
                        </span>
                      </div>
                      <div className="p-1 text-[9px] font-mono text-center text-slate-400 truncate">
                        {f.parameters.groove_pixels} px
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Parameters Summary Badge List */}
              <div className="p-3 bg-slate-950/60 rounded-xl border border-slate-800/80 grid grid-cols-2 gap-2 text-[11px] font-mono">
                <div>
                  <span className="text-slate-400">Groove Valleys: </span>
                  <span className="font-bold text-teal-300">{dataA.median_parameters.groove_pixels} px</span>
                </div>
                <div>
                  <span className="text-slate-400">Skeleton Length: </span>
                  <span className="font-bold text-teal-300">{dataA.median_parameters.skeleton_pixels} px</span>
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
            className={`w-full py-2.5 px-4 rounded-xl text-sm font-bold flex items-center justify-center gap-2 transition-all ${
              loadingB
                ? 'bg-slate-800 text-slate-400 cursor-not-allowed'
                : 'bg-teal-500 hover:bg-teal-400 text-slate-950 shadow-md shadow-teal-500/20'
            }`}
          >
            {loadingB ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin" />
                Extracting 5 Frames &amp; Groove Lines...
              </>
            ) : (
              <>
                <ScanLine className="w-4 h-4" />
                {dataB ? 'Re-Extract 5 Images (Video B)' : 'Extract 5 Images & Store in LocalStorage'}
              </>
            )}
          </button>

          {/* 5 Extracted Images Strip */}
          {dataB && (
            <div className="flex flex-col gap-3 pt-2">
              <div className="flex items-center justify-between text-xs text-slate-400">
                <span className="font-semibold text-slate-200">5 Extracted Muzzle Images:</span>
                <div className="flex items-center gap-1 bg-slate-900 p-0.5 rounded-lg border border-slate-800 text-[10px]">
                  <button
                    onClick={() => setViewTypeB('overlay')}
                    className={`px-2 py-0.5 rounded ${viewTypeB === 'overlay' ? 'bg-slate-800 text-teal-300 font-bold' : 'text-slate-400'}`}
                  >
                    Groove Lines
                  </button>
                  <button
                    onClick={() => setViewTypeB('crop')}
                    className={`px-2 py-0.5 rounded ${viewTypeB === 'crop' ? 'bg-slate-800 text-teal-300 font-bold' : 'text-slate-400'}`}
                  >
                    Raw Crop
                  </button>
                  <button
                    onClick={() => setViewTypeB('skeleton')}
                    className={`px-2 py-0.5 rounded ${viewTypeB === 'skeleton' ? 'bg-slate-800 text-teal-300 font-bold' : 'text-slate-400'}`}
                  >
                    1-px Skel
                  </button>
                </div>
              </div>

              {/* 5 Images Grid */}
              <div className="grid grid-cols-5 gap-2">
                {dataB.frames.map((f, i) => {
                  const imgUrl = viewTypeB === 'crop' ? f.crop_url : viewTypeB === 'skeleton' ? f.skeleton_url : f.overlay_url;
                  return (
                    <div key={i} className="flex flex-col rounded-lg border border-slate-800 overflow-hidden bg-slate-950">
                      <div className="aspect-square relative overflow-hidden bg-slate-950">
                        <img
                          src={imgUrl}
                          alt={`Frame ${f.frame_number}`}
                          className="w-full h-full object-cover"
                        />
                        <span className="absolute top-1 left-1 text-[9px] font-mono px-1 rounded bg-slate-950/80 text-slate-300">
                          #{f.frame_number}
                        </span>
                      </div>
                      <div className="p-1 text-[9px] font-mono text-center text-slate-400 truncate">
                        {f.parameters.groove_pixels} px
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Parameters Summary Badge List */}
              <div className="p-3 bg-slate-950/60 rounded-xl border border-slate-800/80 grid grid-cols-2 gap-2 text-[11px] font-mono">
                <div>
                  <span className="text-slate-400">Groove Valleys: </span>
                  <span className="font-bold text-teal-300">{dataB.median_parameters.groove_pixels} px</span>
                </div>
                <div>
                  <span className="text-slate-400">Skeleton Length: </span>
                  <span className="font-bold text-teal-300">{dataB.median_parameters.skeleton_pixels} px</span>
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

      {/* ========================================================= */}
      {/* CENTRAL ACTION: CALCULATE COSINE SIMILARITY               */}
      {/* ========================================================= */}
      <div className="flex flex-col items-center gap-3">
        <button
          onClick={handleCalculateCosineSimilarity}
          disabled={calculatingSim || !dataA || !dataB}
          className={`px-8 py-4 rounded-2xl text-base font-extrabold flex items-center gap-3 shadow-xl transition-all ${
            !dataA || !dataB
              ? 'bg-slate-800 text-slate-500 border border-slate-700 cursor-not-allowed'
              : 'bg-gradient-to-r from-emerald-400 via-teal-400 to-emerald-500 hover:from-emerald-300 hover:to-teal-300 text-slate-950 shadow-emerald-500/25 ring-2 ring-emerald-300 hover:scale-[1.02] active:scale-[0.98]'
          }`}
        >
          {calculatingSim ? (
            <>
              <RefreshCw className="w-5 h-5 animate-spin" />
              Reading LocalStorage &amp; Calculating Cosine Similarity...
            </>
          ) : (
            <>
              <GitCompare className="w-5 h-5" />
              <span>Calculate Cosine Similarity</span>
            </>
          )}
        </button>

        {(!dataA || !dataB) && (
          <span className="text-xs text-amber-400/80 font-mono flex items-center gap-1.5">
            <Info className="w-3.5 h-3.5" />
            Extract 5 images from both Video A and Video B to populate LocalStorage parameters.
          </span>
        )}
      </div>

      {/* ========================================================= */}
      {/* SIMILARITY RESULT SECTION                                 */}
      {/* ========================================================= */}
      {similarityResult && (
        <div className="glass-panel p-6 rounded-2xl border border-slate-800 flex flex-col gap-6 animate-in fade-in duration-300">
          {/* Main Verdict Card */}
          <div className="flex flex-col md:flex-row items-center justify-between gap-6 p-6 rounded-xl bg-slate-950/80 border border-slate-800">
            <div className="flex items-center gap-4">
              <div
                className={`w-16 h-16 rounded-2xl flex items-center justify-center font-bold text-2xl border ${
                  similarityResult.is_match
                    ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40 ring-4 ring-emerald-500/10'
                    : 'bg-rose-500/20 text-rose-300 border-rose-500/40 ring-4 ring-rose-500/10'
                }`}
              >
                {similarityResult.cosine_similarity}%
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <h3 className="text-xl font-extrabold text-white">
                    {similarityResult.verdict}
                  </h3>
                  <span className="text-xs font-mono px-2 py-0.5 rounded-full bg-slate-800 text-slate-300 border border-slate-700">
                    Threshold: 85.0%
                  </span>
                </div>
                <p className="text-xs text-slate-400 mt-1">
                  Cosine Similarity calculated across 10 dermatoglyphic groove parameters between{' '}
                  <span className="text-emerald-400 font-bold">{similarityResult.videoAName}</span> and{' '}
                  <span className="text-teal-400 font-bold">{similarityResult.videoBName}</span>.
                </p>
              </div>
            </div>

            {/* Score pill */}
            <div className="flex flex-col items-end gap-1 font-mono text-xs text-slate-400">
              <div className="flex items-center gap-2">
                <span>Normalized Cosine Sim:</span>
                <span className="text-sm font-bold text-teal-300">{similarityResult.cosine_similarity}%</span>
              </div>
              <div className="flex items-center gap-2">
                <span>Raw Unnormalized Cosine:</span>
                <span className="text-slate-300">{similarityResult.raw_cosine_similarity}%</span>
              </div>
            </div>
          </div>

          {/* Parameters Comparison Table */}
          <div className="flex flex-col gap-2">
            <h4 className="text-sm font-bold text-slate-200">
              10-Parameter Side-by-Side Breakdown:
            </h4>
            <div className="overflow-x-auto rounded-xl border border-slate-800">
              <table className="w-full text-left text-xs font-mono">
                <thead className="bg-slate-950 text-slate-400 uppercase text-[10px] tracking-wider border-b border-slate-800">
                  <tr>
                    <th className="py-2.5 px-4">Feature Parameter</th>
                    <th className="py-2.5 px-4 text-emerald-400">Video A (Median)</th>
                    <th className="py-2.5 px-4 text-teal-400">Video B (Median)</th>
                    <th className="py-2.5 px-4 text-right">Relative Delta (Δ)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 bg-slate-900/40">
                  {Object.keys(FEATURE_LABELS).map((k) => {
                    const valA = similarityResult.paramsA[k] || 0;
                    const valB = similarityResult.paramsB[k] || 0;
                    const maxVal = Math.max(Math.abs(valA), Math.abs(valB), 1e-4);
                    const delta = Math.abs(valA - valB);
                    const deltaPct = ((delta / maxVal) * 100).toFixed(1);
                    return (
                      <tr key={k} className="hover:bg-slate-800/30 transition-colors">
                        <td className="py-2 px-4 font-sans font-medium text-slate-300">
                          {FEATURE_LABELS[k]}
                        </td>
                        <td className="py-2 px-4 text-emerald-300 font-bold">
                          {valA} <span className="text-[10px] font-normal text-slate-500">{FEATURE_UNITS[k]}</span>
                        </td>
                        <td className="py-2 px-4 text-teal-300 font-bold">
                          {valB} <span className="text-[10px] font-normal text-slate-500">{FEATURE_UNITS[k]}</span>
                        </td>
                        <td className="py-2 px-4 text-right">
                          <span className={`px-2 py-0.5 rounded text-[11px] font-semibold ${
                            parseFloat(deltaPct) <= 15 ? 'text-emerald-400 bg-emerald-500/10' : 'text-slate-400'
                          }`}>
                            {deltaPct}%
                          </span>
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

      {/* LocalStorage Inspection Modal */}
      {showLocalStorageModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm animate-in fade-in duration-150">
          <div className="glass-panel p-6 rounded-2xl border border-slate-700 max-w-3xl w-full max-h-[85vh] flex flex-col gap-4 shadow-2xl">
            <div className="flex items-center justify-between pb-3 border-b border-slate-800">
              <div className="flex items-center gap-2">
                <Database className="w-5 h-5 text-teal-400" />
                <h3 className="text-base font-bold text-white">Browser LocalStorage Inspector</h3>
              </div>
              <button
                onClick={() => setShowLocalStorageModal(false)}
                className="text-slate-400 hover:text-white"
              >
                &times;
              </button>
            </div>

            <div className="overflow-y-auto flex-1 flex flex-col gap-4 pr-1 text-xs font-mono">
              <div>
                <span className="font-bold text-emerald-400">Key: 'muzzle_video_a_params'</span>
                <pre className="mt-1 p-3 rounded-xl bg-slate-950 border border-slate-800 text-slate-300 overflow-x-auto max-h-48">
                  {storageData.a || 'No data currently stored under this key.'}
                </pre>
              </div>

              <div>
                <span className="font-bold text-teal-400">Key: 'muzzle_video_b_params'</span>
                <pre className="mt-1 p-3 rounded-xl bg-slate-950 border border-slate-800 text-slate-300 overflow-x-auto max-h-48">
                  {storageData.b || 'No data currently stored under this key.'}
                </pre>
              </div>
            </div>

            <div className="flex justify-end pt-3 border-t border-slate-800">
              <button
                onClick={() => setShowLocalStorageModal(false)}
                className="px-4 py-2 rounded-xl text-xs font-bold bg-slate-800 hover:bg-slate-700 text-white"
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
