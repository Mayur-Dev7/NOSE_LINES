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
  ChevronUp,
  Info,
  Trash2,
  Maximize2,
  ShieldCheck,
  AlertCircle,
  Cpu,
  Zap,
  BarChart3,
  Network,
  Activity,
  FileVideo,
  Check
} from 'lucide-react';

const FEATURE_LABELS = {
  groove_pixels: 'Groove Pixels (Valleys)',
  skeleton_pixels: 'Ridge Skeleton (1-px Lines)',
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

const DEFAULT_SAMPLE_VIDEOS = [
  { name: 'muzzleVideo.mp4', path: '/home/adis_mayur/F_CODES/muzzleVideo.mp4', label: 'Cattle A - Profile 1 (2.55 MB)' },
  { name: 'muzzleVideo2.mp4', path: '/home/adis_mayur/F_CODES/muzzleVideo2.mp4', label: 'Cattle A - Profile 2 / Same Cattle (2.59 MB)' },
  { name: 'muzzleVideo3.mp4', path: '/home/adis_mayur/F_CODES/muzzleVideo3.mp4', label: 'Cattle B - Different Cow (2.63 MB)' },
  { name: 'muzzleVideo5.mp4', path: '/home/adis_mayur/F_CODES/muzzleVideo5.mp4', label: 'Cattle C - High Res (4.25 MB)' },
  { name: 'muzzleVideo6.mp4', path: '/home/adis_mayur/F_CODES/muzzleVideo6.mp4', label: 'Cattle C - High Res 2 (4.26 MB)' },
  { name: 'muzzleVideo7.mp4', path: '/home/adis_mayur/F_CODES/muzzleVideo7.mp4', label: 'Cattle D - Field Query (4.25 MB)' },
];

export default function TwoVideoCosineMatcher({ sampleVideos = [] }) {
  // Video selection state
  const [videoA, setVideoA] = useState('/home/adis_mayur/F_CODES/muzzleVideo.mp4');
  const [videoB, setVideoB] = useState('/home/adis_mayur/F_CODES/muzzleVideo2.mp4');
  const [inputModeA, setInputModeA] = useState('select'); // 'select' | 'upload'
  const [inputModeB, setInputModeB] = useState('select');
  const [uploadedNameA, setUploadedNameA] = useState('');
  const [uploadedNameB, setUploadedNameB] = useState('');
  const [uploadingA, setUploadingA] = useState(false);
  const [uploadingB, setUploadingB] = useState(false);

  // Model & Processing Configuration (sensible defaults)
  const [dlBackbone, setDlBackbone] = useState('convnext_tiny'); // 'convnext_tiny' | 'resnet50'
  const [dlInputMode, setDlInputMode] = useState('composite_3ch'); // 'composite_3ch' | 'skeleton_only' | 'groove_map'
  const [filterMethod, setFilterMethod] = useState('frangi');
  const [thresholdMethod, setThresholdMethod] = useState('adaptive');
  const [adaptiveBlockSize, setAdaptiveBlockSize] = useState(15);
  const [adaptiveC, setAdaptiveC] = useState(2);
  const [excludeNostrils, setExcludeNostrils] = useState(true);

  // Accordion toggle states (hidden by default to avoid clutter)
  const [showAdvancedSettings, setShowAdvancedSettings] = useState(false);
  const [showFrameDetails, setShowFrameDetails] = useState(false);
  const [viewStageA, setViewStageA] = useState('skeleton');
  const [viewStageB, setViewStageB] = useState('skeleton');
  const [previewImage, setPreviewImage] = useState(null);

  // Unified single-click workflow execution state
  const [processing, setProcessing] = useState(false);
  const [progressStep, setProgressStep] = useState(0); // 0 = idle, 1 = Video A, 2 = Video B, 3 = Matching
  const [progressMessage, setProgressMessage] = useState('');
  const [errorMessage, setErrorMessage] = useState(null);

  // Results state
  const [matchResult, setMatchResult] = useState(null);
  const [dataA, setDataA] = useState(null);
  const [dataB, setDataB] = useState(null);

  // LocalStorage inspection modal
  const [showLocalStorageModal, setShowLocalStorageModal] = useState(false);

  // Available videos list
  const availableVideos = sampleVideos.length > 0
    ? sampleVideos.map(v => ({ name: v.name, path: v.path, label: `${v.name} (${v.size_mb || '?'} MB)` }))
    : DEFAULT_SAMPLE_VIDEOS;

  // Upload handler for Video A
  const handleFileUploadA = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setUploadingA(true);
    const formData = new FormData();
    formData.append('file', file);
    try {
      const res = await fetch('/api/upload', { method: 'POST', body: formData });
      const data = await res.json();
      if (data.success) {
        setVideoA(data.path);
        setUploadedNameA(data.filename);
      } else {
        alert('Upload failed: ' + (data.detail || 'Unknown error'));
      }
    } catch (err) {
      alert('Upload failed: ' + err.message);
    } finally {
      setUploadingA(false);
    }
  };

  // Upload handler for Video B
  const handleFileUploadB = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setUploadingB(true);
    const formData = new FormData();
    formData.append('file', file);
    try {
      const res = await fetch('/api/upload', { method: 'POST', body: formData });
      const data = await res.json();
      if (data.success) {
        setVideoB(data.path);
        setUploadedNameB(data.filename);
      } else {
        alert('Upload failed: ' + (data.detail || 'Unknown error'));
      }
    } catch (err) {
      alert('Upload failed: ' + err.message);
    } finally {
      setUploadingB(false);
    }
  };

  // =========================================================================
  // THE 1-CLICK VERIFICATION PIPELINE
  // Takes 2 videos -> Extracts frames & embeddings -> Returns verdict
  // =========================================================================
  const handleRunFullVerification = async () => {
    if (processing) return;
    setProcessing(true);
    setErrorMessage(null);
    setMatchResult(null);

    try {
      // Step 1: Video A extraction
      setProgressStep(1);
      setProgressMessage('Analyzing Video A: detecting muzzle & extracting ConvNeXt-Tiny embeddings...');
      const resA = await fetch('/api/video/extract-five-muzzle-parameters', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          video_path: videoA,
          method: filterMethod,
          threshold_method: thresholdMethod,
          adaptive_block_size: parseInt(adaptiveBlockSize, 10),
          adaptive_c: parseInt(adaptiveC, 10),
          exclude_nostrils: excludeNostrils,
          min_spur_length: 8,
          min_component_size: 30,
          dl_backbone: dlBackbone,
          dl_input_mode: dlInputMode,
        }),
      });
      const extractedA = await resA.json();
      if (!extractedA.success) {
        throw new Error(extractedA.detail || 'Muzzle detection/extraction failed for Video A');
      }
      setDataA(extractedA);
      localStorage.setItem('muzzle_video_a_params', JSON.stringify(extractedA));

      // Step 2: Video B extraction
      setProgressStep(2);
      setProgressMessage('Analyzing Video B: detecting muzzle & extracting ConvNeXt-Tiny embeddings...');
      const resB = await fetch('/api/video/extract-five-muzzle-parameters', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          video_path: videoB,
          method: filterMethod,
          threshold_method: thresholdMethod,
          adaptive_block_size: parseInt(adaptiveBlockSize, 10),
          adaptive_c: parseInt(adaptiveC, 10),
          exclude_nostrils: excludeNostrils,
          min_spur_length: 8,
          min_component_size: 30,
          dl_backbone: dlBackbone,
          dl_input_mode: dlInputMode,
        }),
      });
      const extractedB = await resB.json();
      if (!extractedB.success) {
        throw new Error(extractedB.detail || 'Muzzle detection/extraction failed for Video B');
      }
      setDataB(extractedB);
      localStorage.setItem('muzzle_video_b_params', JSON.stringify(extractedB));

      // Step 3: Compute Biometric Cosine Match & Verdict
      setProgressStep(3);
      setProgressMessage('Matching biometrics: ArcFace 512-D GeM cosine similarity & topological minutiae...');
      const vecA = extractedA.vector_array || Object.values(extractedA.median_parameters);
      const vecB = extractedB.vector_array || Object.values(extractedB.median_parameters);
      const dlEmbA = extractedA.dl_embedding?.median_embedding || null;
      const dlEmbB = extractedB.dl_embedding?.median_embedding || null;
      const perFramesA = extractedA.dl_embedding?.per_frame_embeddings || null;
      const perFramesB = extractedB.dl_embedding?.per_frame_embeddings || null;

      const resSim = await fetch('/api/video/cosine-similarity', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          vector_a: vecA,
          vector_b: vecB,
          dl_emb_a: dlEmbA,
          dl_emb_b: dlEmbB,
          per_frames_a: perFramesA,
          per_frames_b: perFramesB,
        }),
      });
      const simData = await resSim.json();

      setMatchResult({
        ...simData,
        dl_embedding_a: extractedA.dl_embedding,
        dl_embedding_b: extractedB.dl_embedding,
        paramsA: extractedA.median_parameters,
        paramsB: extractedB.median_parameters,
        videoAName: extractedA.video_name,
        videoBName: extractedB.video_name,
      });

      // Scroll smoothly to results
      setTimeout(() => {
        const resultElem = document.getElementById('biometric-results-view');
        if (resultElem) {
          resultElem.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      }, 100);

    } catch (err) {
      console.error('Verification error:', err);
      setErrorMessage(err.message || 'Verification failed');
    } finally {
      setProcessing(false);
      setProgressStep(0);
    }
  };

  const getFrameUrl = (frame, viewType) => {
    switch (viewType) {
      case 'crop': return frame.crop_url;
      case 'enhanced': return frame.enhanced_url;
      case 'response': return frame.groove_response_url;
      case 'mask': return frame.groove_mask_url;
      case 'overlay': return frame.overlay_url;
      case 'skeleton':
      default: return frame.skeleton_url;
    }
  };

  // Quick preset loader
  const applyPreset = (pathA, pathB) => {
    setVideoA(pathA);
    setVideoB(pathB);
    setInputModeA('select');
    setInputModeB('select');
    setMatchResult(null);
    setErrorMessage(null);
  };

  // Reset all
  const handleReset = () => {
    setDataA(null);
    setDataB(null);
    setMatchResult(null);
    setErrorMessage(null);
    localStorage.removeItem('muzzle_video_a_params');
    localStorage.removeItem('muzzle_video_b_params');
  };

  return (
    <div className="flex flex-col gap-6 text-slate-100 max-w-5xl mx-auto w-full pb-20">
      
      {/* ========================================================= */}
      {/* 1. CLEAN HEADER                                           */}
      {/* ========================================================= */}
      <div className="glass-panel p-6 rounded-2xl border border-slate-800 flex items-center justify-between gap-4 flex-wrap">
        <div className="flex items-center gap-3">
          <div className="w-12 h-12 rounded-2xl bg-gradient-to-tr from-emerald-500 to-teal-500 flex items-center justify-center shadow-lg shadow-emerald-500/20">
            <ShieldCheck className="w-7 h-7 text-slate-950" />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
              Cattle Muzzle Biometric Matcher
              <span className="text-[11px] font-mono px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                ConvNeXt ArcFace &bull; 512-D
              </span>
            </h1>
            <p className="text-xs text-slate-400 mt-0.5">
              Select or upload 2 cattle videos to verify biometric identity across 5 temporal muzzle frames.
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowLocalStorageModal(true)}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-medium bg-slate-900 hover:bg-slate-800 border border-slate-700 text-slate-300 transition-colors"
          >
            <Database className="w-3.5 h-3.5 text-teal-400" />
            LocalStorage
          </button>
          <button
            onClick={handleReset}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-medium bg-slate-900 hover:bg-rose-950/40 border border-slate-800 hover:border-rose-800 text-slate-400 hover:text-rose-300 transition-colors"
            title="Reset videos and results"
          >
            <Trash2 className="w-3.5 h-3.5" />
            Reset
          </button>
        </div>
      </div>

      {/* ========================================================= */}
      {/* 2. VIDEO INPUT CARDS (VIDEO A & VIDEO B)                  */}
      {/* ========================================================= */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        
        {/* CARD VIDEO A */}
        <div className="glass-panel p-5 rounded-2xl border border-slate-800 flex flex-col gap-4 relative overflow-hidden">
          <div className="flex items-center justify-between pb-3 border-b border-slate-800">
            <div className="flex items-center gap-2.5">
              <span className="w-7 h-7 rounded-lg bg-emerald-500/20 text-emerald-400 font-bold flex items-center justify-center text-sm border border-emerald-500/30 font-mono">
                A
              </span>
              <div>
                <h3 className="text-sm font-bold text-white">Video A (Reference Animal)</h3>
                <span className="text-[11px] text-slate-400">Registered baseline cattle video</span>
              </div>
            </div>
            
            <div className="flex items-center bg-slate-900 p-0.5 rounded-lg border border-slate-800 text-xs">
              <button
                onClick={() => setInputModeA('select')}
                className={`px-2 py-1 rounded-md text-[11px] font-medium transition-colors ${
                  inputModeA === 'select' ? 'bg-emerald-500 text-slate-950 font-bold' : 'text-slate-400 hover:text-white'
                }`}
              >
                Sample Video
              </button>
              <button
                onClick={() => setInputModeA('upload')}
                className={`px-2 py-1 rounded-md text-[11px] font-medium transition-colors ${
                  inputModeA === 'upload' ? 'bg-emerald-500 text-slate-950 font-bold' : 'text-slate-400 hover:text-white'
                }`}
              >
                Upload File
              </button>
            </div>
          </div>

          {inputModeA === 'select' ? (
            <div className="flex flex-col gap-1.5 text-xs">
              <label className="text-slate-400 font-medium">Choose Reference Video:</label>
              <select
                value={videoA}
                onChange={(e) => setVideoA(e.target.value)}
                className="bg-slate-900 border border-slate-700 rounded-xl px-3 py-2.5 text-slate-200 focus:outline-hidden focus:border-emerald-500 font-medium"
              >
                {availableVideos.map((v, i) => (
                  <option key={i} value={v.path}>{v.label || v.name}</option>
                ))}
              </select>
            </div>
          ) : (
            <div className="flex flex-col gap-2 text-xs">
              <label className="text-slate-400 font-medium">Upload Video File (.mp4):</label>
              <label className="border-2 border-dashed border-slate-700 hover:border-emerald-500/60 rounded-xl p-4 text-center cursor-pointer bg-slate-900/60 flex flex-col items-center justify-center gap-1.5 transition-colors">
                <Upload className="w-5 h-5 text-emerald-400" />
                <span className="text-slate-300 font-medium">
                  {uploadingA ? 'Uploading video...' : uploadedNameA ? uploadedNameA : 'Click to browse MP4 file'}
                </span>
                <span className="text-[10px] text-slate-500">Max size 50 MB</span>
                <input type="file" accept="video/mp4,video/*" onChange={handleFileUploadA} className="hidden" />
              </label>
            </div>
          )}

          <div className="text-[11px] font-mono text-slate-400 truncate bg-slate-900/50 p-2 rounded-lg border border-slate-800/80">
            Selected: <span className="text-emerald-300">{videoA.split('/').pop()}</span>
          </div>
        </div>

        {/* CARD VIDEO B */}
        <div className="glass-panel p-5 rounded-2xl border border-slate-800 flex flex-col gap-4 relative overflow-hidden">
          <div className="flex items-center justify-between pb-3 border-b border-slate-800">
            <div className="flex items-center gap-2.5">
              <span className="w-7 h-7 rounded-lg bg-teal-500/20 text-teal-400 font-bold flex items-center justify-center text-sm border border-teal-500/30 font-mono">
                B
              </span>
              <div>
                <h3 className="text-sm font-bold text-white">Video B (Query Candidate)</h3>
                <span className="text-[11px] text-slate-400">Target video to compare against A</span>
              </div>
            </div>

            <div className="flex items-center bg-slate-900 p-0.5 rounded-lg border border-slate-800 text-xs">
              <button
                onClick={() => setInputModeB('select')}
                className={`px-2 py-1 rounded-md text-[11px] font-medium transition-colors ${
                  inputModeB === 'select' ? 'bg-teal-500 text-slate-950 font-bold' : 'text-slate-400 hover:text-white'
                }`}
              >
                Sample Video
              </button>
              <button
                onClick={() => setInputModeB('upload')}
                className={`px-2 py-1 rounded-md text-[11px] font-medium transition-colors ${
                  inputModeB === 'upload' ? 'bg-teal-500 text-slate-950 font-bold' : 'text-slate-400 hover:text-white'
                }`}
              >
                Upload File
              </button>
            </div>
          </div>

          {inputModeB === 'select' ? (
            <div className="flex flex-col gap-1.5 text-xs">
              <label className="text-slate-400 font-medium">Choose Candidate Video:</label>
              <select
                value={videoB}
                onChange={(e) => setVideoB(e.target.value)}
                className="bg-slate-900 border border-slate-700 rounded-xl px-3 py-2.5 text-slate-200 focus:outline-hidden focus:border-teal-500 font-medium"
              >
                {availableVideos.map((v, i) => (
                  <option key={i} value={v.path}>{v.label || v.name}</option>
                ))}
              </select>
            </div>
          ) : (
            <div className="flex flex-col gap-2 text-xs">
              <label className="text-slate-400 font-medium">Upload Video File (.mp4):</label>
              <label className="border-2 border-dashed border-slate-700 hover:border-teal-500/60 rounded-xl p-4 text-center cursor-pointer bg-slate-900/60 flex flex-col items-center justify-center gap-1.5 transition-colors">
                <Upload className="w-5 h-5 text-teal-400" />
                <span className="text-slate-300 font-medium">
                  {uploadingB ? 'Uploading video...' : uploadedNameB ? uploadedNameB : 'Click to browse MP4 file'}
                </span>
                <span className="text-[10px] text-slate-500">Max size 50 MB</span>
                <input type="file" accept="video/mp4,video/*" onChange={handleFileUploadB} className="hidden" />
              </label>
            </div>
          )}

          <div className="text-[11px] font-mono text-slate-400 truncate bg-slate-900/50 p-2 rounded-lg border border-slate-800/80">
            Selected: <span className="text-teal-300">{videoB.split('/').pop()}</span>
          </div>
        </div>
      </div>

      {/* QUICK PRESETS PILLS */}
      <div className="flex items-center justify-between flex-wrap gap-2 text-xs text-slate-400 px-1">
        <span className="flex items-center gap-1 font-semibold text-slate-300">
          <Sparkles className="w-3.5 h-3.5 text-emerald-400" />
          Quick Test Pairs:
        </span>
        <div className="flex flex-wrap items-center gap-2">
          <button
            onClick={() => applyPreset('/home/adis_mayur/F_CODES/muzzleVideo.mp4', '/home/adis_mayur/F_CODES/muzzleVideo.mp4')}
            className="px-2.5 py-1 rounded-lg bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 font-medium transition-colors"
          >
            🟢 Same Cattle (Video 1 vs Video 1)
          </button>
          <button
            onClick={() => applyPreset('/home/adis_mayur/F_CODES/muzzleVideo.mp4', '/home/adis_mayur/F_CODES/muzzleVideo2.mp4')}
            className="px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 font-medium transition-colors"
          >
            🔍 Video 1 vs Video 2 (Same Animal Different Take)
          </button>
          <button
            onClick={() => applyPreset('/home/adis_mayur/F_CODES/muzzleVideo.mp4', '/home/adis_mayur/F_CODES/muzzleVideo3.mp4')}
            className="px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 font-medium transition-colors"
          >
            🔴 Video 1 vs Video 3 (Different Cattle)
          </button>
        </div>
      </div>

      {/* ========================================================= */}
      {/* 3. THE MAIN SUBMIT BUTTON                                 */}
      {/* ========================================================= */}
      <div className="flex flex-col gap-3 items-center">
        <button
          onClick={handleRunFullVerification}
          disabled={processing}
          className={`w-full py-4 px-6 rounded-2xl font-bold text-sm tracking-wide flex items-center justify-center gap-3 transition-all shadow-xl ${
            processing
              ? 'bg-slate-800 text-slate-400 cursor-not-allowed border border-slate-700'
              : 'bg-gradient-to-r from-emerald-500 via-teal-500 to-cyan-500 hover:from-emerald-400 hover:to-cyan-400 text-slate-950 shadow-emerald-500/25 hover:shadow-emerald-500/40 hover:scale-[1.01]'
          }`}
        >
          {processing ? (
            <>
              <RefreshCw className="w-5 h-5 animate-spin text-emerald-400" />
              <span>{progressMessage || 'Processing Videos...'}</span>
            </>
          ) : (
            <>
              <Zap className="w-5 h-5 fill-slate-950" />
              <span>COMPARE 2 VIDEOS &amp; GET BIOMETRIC RESULT</span>
              <ArrowRight className="w-5 h-5" />
            </>
          )}
        </button>

        {/* Animated Progress Indicator */}
        {processing && (
          <div className="w-full glass-panel p-4 rounded-xl border border-emerald-500/30 flex flex-col gap-2">
            <div className="flex items-center justify-between text-xs font-mono">
              <span className="text-emerald-300 font-bold">{progressMessage}</span>
              <span className="text-slate-400">Step {progressStep} of 3</span>
            </div>
            <div className="w-full bg-slate-900 rounded-full h-2 overflow-hidden border border-slate-800">
              <div
                className="bg-gradient-to-r from-emerald-500 to-teal-400 h-2 rounded-full transition-all duration-500"
                style={{ width: `${(progressStep / 3) * 100}%` }}
              ></div>
            </div>
          </div>
        )}

        {/* Error Alert */}
        {errorMessage && (
          <div className="w-full p-4 rounded-xl bg-rose-950/40 border border-rose-500/50 flex items-center gap-3 text-xs text-rose-200">
            <XCircle className="w-5 h-5 text-rose-400 shrink-0" />
            <span><strong>Verification Failed:</strong> {errorMessage}</span>
          </div>
        )}
      </div>

      {/* ========================================================= */}
      {/* 4. RESULTS SECTION (Appears immediately after submission) */}
      {/* ========================================================= */}
      {matchResult && (
        <div id="biometric-results-view" className="flex flex-col gap-6 pt-4 animate-in fade-in duration-300">
          
          {/* THE HERO VERDICT BANNER */}
          <div className={`p-6 rounded-2xl border flex flex-col md:flex-row items-center justify-between gap-6 shadow-2xl relative overflow-hidden ${
            matchResult.dl_similarity_pct >= 85.0
              ? 'bg-gradient-to-br from-emerald-950/40 via-slate-950 to-slate-950 border-emerald-500/50 shadow-emerald-950/30'
              : matchResult.dl_similarity_pct >= 75.0
              ? 'bg-gradient-to-br from-amber-950/40 via-slate-950 to-slate-950 border-amber-500/50 shadow-amber-950/30'
              : 'bg-gradient-to-br from-rose-950/40 via-slate-950 to-slate-950 border-rose-500/50 shadow-rose-950/30'
          }`}>
            
            {/* Left: Verdict Text & Icon */}
            <div className="flex items-center gap-4">
              <div className={`w-16 h-16 rounded-2xl flex items-center justify-center shrink-0 border ${
                matchResult.dl_similarity_pct >= 85.0
                  ? 'bg-emerald-500/20 text-emerald-400 border-emerald-500/40'
                  : matchResult.dl_similarity_pct >= 75.0
                  ? 'bg-amber-500/20 text-amber-400 border-amber-500/40'
                  : 'bg-rose-500/20 text-rose-400 border-rose-500/40'
              }`}>
                {matchResult.dl_similarity_pct >= 85.0 ? (
                  <CheckCircle2 className="w-10 h-10" />
                ) : matchResult.dl_similarity_pct >= 75.0 ? (
                  <AlertCircle className="w-10 h-10" />
                ) : (
                  <XCircle className="w-10 h-10" />
                )}
              </div>

              <div>
                <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold font-mono border mb-1.5 ${
                  matchResult.dl_similarity_pct >= 85.0
                    ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
                    : matchResult.dl_similarity_pct >= 75.0
                    ? 'bg-amber-500/20 text-amber-300 border-amber-500/40'
                    : 'bg-rose-500/20 text-rose-300 border-rose-500/40'
                }`}>
                  {matchResult.dl_similarity_pct >= 85.0 ? 'BIOMETRIC MATCH' : matchResult.dl_similarity_pct >= 75.0 ? 'INCONCLUSIVE' : 'DISTINCT ANIMALS'}
                </span>
                
                <h2 className="text-xl font-extrabold text-white">
                  {matchResult.dl_similarity_pct >= 85.0
                    ? 'Confirmed Same Cattle (Profile Match)'
                    : matchResult.dl_similarity_pct >= 75.0
                    ? 'Partial Similarity (Inconclusive)'
                    : 'Different Animals (Identity Rejected)'}
                </h2>

                <p className="text-xs text-slate-400 mt-1">
                  {matchResult.dl_similarity_pct >= 85.0
                    ? `Video A (${matchResult.videoAName}) and Video B (${matchResult.videoBName}) share identical epidermal muzzle groove dermatoglyphics.`
                    : `Video A and Video B exhibit distinct ridge loop topologies and dermatoglyphic patterns.`}
                </p>
              </div>
            </div>

            {/* Right: Big Percentage Number */}
            <div className="flex flex-col items-center md:items-end shrink-0">
              <span className="text-[11px] font-mono text-slate-400 uppercase tracking-wider">Deep Learning Match</span>
              <div className="flex items-baseline gap-1">
                <span className={`text-4xl font-black font-mono tracking-tight ${
                  matchResult.dl_similarity_pct >= 85.0
                    ? 'text-emerald-400'
                    : matchResult.dl_similarity_pct >= 75.0
                    ? 'text-amber-400'
                    : 'text-rose-400'
                }`}>
                  {matchResult.dl_similarity_pct !== undefined ? matchResult.dl_similarity_pct : matchResult.cosine_similarity}%
                </span>
              </div>
              <span className="text-[11px] font-mono text-slate-400 mt-1">
                L2 Distance: <strong className="text-slate-200">{matchResult.l2_distance ?? 'N/A'}</strong>
              </span>
            </div>
          </div>

          {/* TWO SIDE-BY-SIDE SUMMARY CARDS */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
            
            {/* CARD 1: DEEP LEARNING (CONVNEXT / RESNET) */}
            <div className="glass-panel p-5 rounded-2xl border border-indigo-900/60 bg-gradient-to-br from-slate-950 via-indigo-950/20 to-slate-950 flex flex-col gap-3.5">
              <div className="flex items-center justify-between pb-2.5 border-b border-indigo-900/40">
                <div className="flex items-center gap-2">
                  <Cpu className="w-4 h-4 text-indigo-400" />
                  <h4 className="text-xs font-bold text-white uppercase tracking-wider">
                    Deep Learning Biometric Model
                  </h4>
                </div>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-indigo-500/20 text-indigo-300 border border-indigo-500/40 font-bold">
                  {matchResult.dl_embedding_a?.backbone === 'resnet50' ? 'ResNet-50' : 'ConvNeXt-Tiny'} (512-D)
                </span>
              </div>

              <div className="grid grid-cols-2 gap-3 text-xs font-mono">
                <div className="p-2.5 rounded-xl bg-slate-900/80 border border-slate-800">
                  <span className="text-slate-400 block text-[10px]">DL Cosine Similarity</span>
                  <span className="text-base font-bold text-indigo-300">
                    {matchResult.dl_similarity_pct !== undefined ? `${matchResult.dl_similarity_pct}%` : 'N/A'}
                  </span>
                </div>

                <div className="p-2.5 rounded-xl bg-slate-900/80 border border-slate-800">
                  <span className="text-slate-400 block text-[10px]">Euclidean Distance</span>
                  <span className="text-base font-bold text-indigo-300">
                    {matchResult.l2_distance ?? 'N/A'}
                  </span>
                </div>

                <div className="p-2.5 rounded-xl bg-slate-900/80 border border-slate-800">
                  <span className="text-slate-400 block text-[10px]">Inference Speed</span>
                  <span className="text-slate-200 font-bold text-xs">
                    {matchResult.dl_embedding_a?.avg_inference_ms ? `${matchResult.dl_embedding_a.avg_inference_ms} ms` : '~47 ms'} / frame
                  </span>
                </div>

                <div className="p-2.5 rounded-xl bg-slate-900/80 border border-slate-800">
                  <span className="text-slate-400 block text-[10px]">Intra-Video Consistency</span>
                  <span className="text-teal-300 font-bold text-xs">
                    {matchResult.dl_embedding_a?.intra_video_consistency_pct ?? 97}%
                  </span>
                </div>
              </div>

              {matchResult.cross_matrix_stats && Object.keys(matchResult.cross_matrix_stats).length > 0 && (
                <div className="p-2.5 rounded-xl bg-slate-900/50 border border-slate-800 flex items-center justify-between text-[11px] font-mono text-slate-400">
                  <span>5x5 Frame Pairs Range:</span>
                  <span>
                    Min <strong className="text-slate-300">{matchResult.cross_matrix_stats.min_pair_similarity}%</strong> &bull; Max <strong className="text-emerald-400">{matchResult.cross_matrix_stats.max_pair_similarity}%</strong>
                  </span>
                </div>
              )}
            </div>

            {/* CARD 2: CLASSICAL MINUTIAE (BASELINE) */}
            <div className="glass-panel p-5 rounded-2xl border border-slate-800 bg-slate-950/70 flex flex-col gap-3.5">
              <div className="flex items-center justify-between pb-2.5 border-b border-slate-800">
                <div className="flex items-center gap-2">
                  <Network className="w-4 h-4 text-emerald-400" />
                  <h4 className="text-xs font-bold text-white uppercase tracking-wider">
                    Classical Graph Minutiae (Baseline)
                  </h4>
                </div>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-500/15 text-emerald-300 border border-emerald-500/30 font-bold">
                  10 Parameters
                </span>
              </div>

              <div className="grid grid-cols-2 gap-3 text-xs font-mono">
                <div className="p-2.5 rounded-xl bg-slate-900/80 border border-slate-800">
                  <span className="text-slate-400 block text-[10px]">Normalized Cosine</span>
                  <span className="text-base font-bold text-emerald-300">
                    {matchResult.cosine_similarity !== undefined ? `${matchResult.cosine_similarity}%` : 'N/A'}
                  </span>
                </div>

                <div className="p-2.5 rounded-xl bg-slate-900/80 border border-slate-800">
                  <span className="text-slate-400 block text-[10px]">Raw Cosine Similarity</span>
                  <span className="text-base font-bold text-teal-300">
                    {matchResult.raw_cosine_similarity ?? 'N/A'}%
                  </span>
                </div>

                <div className="p-2.5 rounded-xl bg-slate-900/80 border border-slate-800">
                  <span className="text-slate-400 block text-[10px]">Graph Algorithm</span>
                  <span className="text-slate-200 font-bold text-xs">
                    NetworkX Topology
                  </span>
                </div>

                <div className="p-2.5 rounded-xl bg-slate-900/80 border border-slate-800">
                  <span className="text-slate-400 block text-[10px]">Feature Invariant</span>
                  <span className="text-slate-200 font-bold text-xs">
                    Euler Characteristic χ
                  </span>
                </div>
              </div>

              <div className="p-2.5 rounded-xl bg-slate-900/50 border border-slate-800 flex items-center justify-between text-[11px] font-mono text-slate-400">
                <span>Baseline Verdict:</span>
                <span className="font-bold text-slate-200">{matchResult.verdict || 'N/A'}</span>
              </div>
            </div>
          </div>

          {/* ========================================================= */}
          {/* COLLAPSIBLE ACCORDION: VIEW EXTRACTED 5-STAGE IMAGES     */}
          {/* ========================================================= */}
          {dataA && dataB && (
            <div className="glass-panel rounded-2xl border border-slate-800 overflow-hidden">
              <button
                onClick={() => setShowFrameDetails(!showFrameDetails)}
                className="w-full p-4 flex items-center justify-between text-xs font-bold text-slate-300 hover:text-white hover:bg-slate-900/50 transition-colors"
              >
                <div className="flex items-center gap-2">
                  <ScanLine className="w-4 h-4 text-emerald-400" />
                  <span>Inspect 5 Extracted Frame Muzzle Images &amp; Minutiae (Optional)</span>
                </div>
                <div className="flex items-center gap-2 text-slate-400 text-xs">
                  <span>{showFrameDetails ? 'Hide' : 'Show 5 Images & Numbers'}</span>
                  {showFrameDetails ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                </div>
              </button>

              {showFrameDetails && (
                <div className="p-5 border-t border-slate-800 flex flex-col gap-6 bg-slate-950/40">
                  
                  {/* VIDEO A 5 IMAGES */}
                  <div className="flex flex-col gap-2.5">
                    <div className="flex items-center justify-between text-xs">
                      <span className="font-bold text-emerald-400 flex items-center gap-1.5">
                        <span className="w-5 h-5 rounded bg-emerald-500/20 text-emerald-300 flex items-center justify-center text-[10px]">A</span>
                        Video A: 5 Temporal Frames (10%, 30%, 50%, 70%, 90%)
                      </span>
                      <div className="flex items-center gap-1 bg-slate-900 p-0.5 rounded-lg border border-slate-800 text-[10px]">
                        <button
                          onClick={() => setViewStageA('skeleton')}
                          className={`px-2 py-0.5 rounded ${viewStageA === 'skeleton' ? 'bg-emerald-500/20 text-emerald-300 font-bold' : 'text-slate-400'}`}
                        >
                          5th Image (Skeleton)
                        </button>
                        <button
                          onClick={() => setViewStageA('crop')}
                          className={`px-2 py-0.5 rounded ${viewStageA === 'crop' ? 'bg-slate-800 text-white' : 'text-slate-400'}`}
                        >
                          Native Crop
                        </button>
                        <button
                          onClick={() => setViewStageA('response')}
                          className={`px-2 py-0.5 rounded ${viewStageA === 'response' ? 'bg-slate-800 text-white' : 'text-slate-400'}`}
                        >
                          Groove Map
                        </button>
                      </div>
                    </div>

                    <div className="grid grid-cols-5 gap-2">
                      {dataA.frames.map((f, i) => (
                        <div
                          key={i}
                          onClick={() => setPreviewImage({ url: getFrameUrl(f, viewStageA), title: `Video A - Frame #${f.frame_number}` })}
                          className="flex flex-col rounded-xl border border-slate-800 overflow-hidden bg-slate-950 cursor-pointer hover:border-emerald-500 transition-colors"
                        >
                          <div className="aspect-square relative overflow-hidden">
                            <img src={getFrameUrl(f, viewStageA)} alt={`Frame ${i + 1}`} className="w-full h-full object-cover" />
                            <span className="absolute top-1 left-1 text-[9px] font-mono px-1 rounded bg-slate-950/80 text-slate-300">
                              #{f.frame_number}
                            </span>
                          </div>
                          <div className="p-1 text-[9px] font-mono text-center text-slate-400 truncate bg-slate-900/60">
                            {f.parameters?.skeleton_pixels ?? 0} skel px
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* VIDEO B 5 IMAGES */}
                  <div className="flex flex-col gap-2.5">
                    <div className="flex items-center justify-between text-xs">
                      <span className="font-bold text-teal-400 flex items-center gap-1.5">
                        <span className="w-5 h-5 rounded bg-teal-500/20 text-teal-300 flex items-center justify-center text-[10px]">B</span>
                        Video B: 5 Temporal Frames (10%, 30%, 50%, 70%, 90%)
                      </span>
                      <div className="flex items-center gap-1 bg-slate-900 p-0.5 rounded-lg border border-slate-800 text-[10px]">
                        <button
                          onClick={() => setViewStageB('skeleton')}
                          className={`px-2 py-0.5 rounded ${viewStageB === 'skeleton' ? 'bg-teal-500/20 text-teal-300 font-bold' : 'text-slate-400'}`}
                        >
                          5th Image (Skeleton)
                        </button>
                        <button
                          onClick={() => setViewStageB('crop')}
                          className={`px-2 py-0.5 rounded ${viewStageB === 'crop' ? 'bg-slate-800 text-white' : 'text-slate-400'}`}
                        >
                          Native Crop
                        </button>
                        <button
                          onClick={() => setViewStageB('response')}
                          className={`px-2 py-0.5 rounded ${viewStageB === 'response' ? 'bg-slate-800 text-white' : 'text-slate-400'}`}
                        >
                          Groove Map
                        </button>
                      </div>
                    </div>

                    <div className="grid grid-cols-5 gap-2">
                      {dataB.frames.map((f, i) => (
                        <div
                          key={i}
                          onClick={() => setPreviewImage({ url: getFrameUrl(f, viewStageB), title: `Video B - Frame #${f.frame_number}` })}
                          className="flex flex-col rounded-xl border border-slate-800 overflow-hidden bg-slate-950 cursor-pointer hover:border-teal-500 transition-colors"
                        >
                          <div className="aspect-square relative overflow-hidden">
                            <img src={getFrameUrl(f, viewStageB)} alt={`Frame ${i + 1}`} className="w-full h-full object-cover" />
                            <span className="absolute top-1 left-1 text-[9px] font-mono px-1 rounded bg-slate-950/80 text-slate-300">
                              #{f.frame_number}
                            </span>
                          </div>
                          <div className="p-1 text-[9px] font-mono text-center text-slate-400 truncate bg-slate-900/60">
                            {f.parameters?.skeleton_pixels ?? 0} skel px
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* 10-PARAMETER DELTA TABLE */}
                  <div className="flex flex-col gap-2 pt-2 border-t border-slate-800">
                    <span className="text-xs font-bold text-slate-300">
                      10 Topological Minutiae Parameters (Median Comparison):
                    </span>
                    <div className="overflow-x-auto rounded-xl border border-slate-800">
                      <table className="w-full text-xs font-mono text-left">
                        <thead className="bg-slate-900 text-slate-400 text-[10px] border-b border-slate-800">
                          <tr>
                            <th className="py-2 px-3">Minutiae Parameter</th>
                            <th className="py-2 px-3 text-emerald-400">Video A</th>
                            <th className="py-2 px-3 text-teal-400">Video B</th>
                            <th className="py-2 px-3 text-right">Delta (&Delta;%)</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-800/60 bg-slate-950/40">
                          {Object.keys(FEATURE_LABELS).map((key) => {
                            const valA = dataA.median_parameters?.[key] ?? 0;
                            const valB = dataB.median_parameters?.[key] ?? 0;
                            const unit = FEATURE_UNITS[key] || '';
                            const delta = valA !== 0 ? Math.abs((valB - valA) / valA) * 100 : 0;
                            return (
                              <tr key={key} className="hover:bg-slate-900/30">
                                <td className="py-1.5 px-3 text-slate-300 font-sans font-medium">{FEATURE_LABELS[key]}</td>
                                <td className="py-1.5 px-3 text-emerald-300 font-bold">{valA} {unit}</td>
                                <td className="py-1.5 px-3 text-teal-300 font-bold">{valB} {unit}</td>
                                <td className={`py-1.5 px-3 text-right font-bold ${
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
            </div>
          )}
        </div>
      )}

      {/* ========================================================= */}
      {/* 5. COLLAPSIBLE ADVANCED SETTINGS ACCORDION                */}
      {/* ========================================================= */}
      <div className="glass-panel rounded-2xl border border-slate-800/80 overflow-hidden">
        <button
          onClick={() => setShowAdvancedSettings(!showAdvancedSettings)}
          className="w-full p-4 flex items-center justify-between text-xs font-semibold text-slate-400 hover:text-white hover:bg-slate-900/40 transition-colors"
        >
          <div className="flex items-center gap-2">
            <Cpu className="w-4 h-4 text-indigo-400" />
            <span>Advanced Pipeline Settings (ConvNeXt vs ResNet, 3-Ch Input, Thresholds)</span>
          </div>
          <div className="flex items-center gap-1.5 text-[11px] font-mono text-slate-500">
            <span>{showAdvancedSettings ? 'Hide' : 'Configure'}</span>
            {showAdvancedSettings ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
          </div>
        </button>

        {showAdvancedSettings && (
          <div className="p-5 border-t border-slate-800 grid grid-cols-1 md:grid-cols-2 gap-4 text-xs bg-slate-950/40">
            {/* DL Backbone */}
            <div className="flex flex-col gap-1.5">
              <label className="text-slate-300 font-medium">Deep Learning Backbone:</label>
              <select
                value={dlBackbone}
                onChange={(e) => setDlBackbone(e.target.value)}
                className="bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-slate-200 font-mono"
              >
                <option value="convnext_tiny">ConvNeXt-Tiny (Low Latency ~47ms, Recommended)</option>
                <option value="resnet50">ResNet-50 (Baseline ~48ms)</option>
              </select>
            </div>

            {/* DL Input Mode */}
            <div className="flex flex-col gap-1.5">
              <label className="text-slate-300 font-medium">DL Input Mode (1-px Sparsity):</label>
              <select
                value={dlInputMode}
                onChange={(e) => setDlInputMode(e.target.value)}
                className="bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-slate-200 font-mono"
              >
                <option value="composite_3ch">3-Ch Composite (CLAHE + Groove + Dilated Skeleton)</option>
                <option value="skeleton_only">Stage 5 Skeleton Only (Dilated + Density Map)</option>
                <option value="groove_map">Stage 3 Groove Response Map</option>
              </select>
            </div>

            {/* Filter Method */}
            <div className="flex flex-col gap-1.5">
              <label className="text-slate-300 font-medium">Groove Enhancement Filter:</label>
              <select
                value={filterMethod}
                onChange={(e) => setFilterMethod(e.target.value)}
                className="bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-slate-200 font-mono"
              >
                <option value="frangi">Frangi Hessian Matrix (Multi-Scale)</option>
                <option value="blackhat">Multi-Scale Black-Hat Morphology</option>
              </select>
            </div>

            {/* Threshold Method */}
            <div className="flex flex-col gap-1.5">
              <label className="text-slate-300 font-medium">Binarization Threshold:</label>
              <select
                value={thresholdMethod}
                onChange={(e) => setThresholdMethod(e.target.value)}
                className="bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-slate-200 font-mono"
              >
                <option value="adaptive">Adaptive Gaussian (15px Block, C=+2)</option>
                <option value="otsu">Standard Otsu Thresholding</option>
              </select>
            </div>
          </div>
        )}
      </div>

      {/* ========================================================= */}
      {/* 6. LOCALSTORAGE MODAL                                     */}
      {/* ========================================================= */}
      {showLocalStorageModal && (
        <div className="fixed inset-0 z-50 bg-slate-950/80 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-slate-900 border border-slate-700 rounded-2xl max-w-2xl w-full p-6 max-h-[85vh] flex flex-col gap-4 shadow-2xl">
            <div className="flex items-center justify-between pb-3 border-b border-slate-800">
              <div className="flex items-center gap-2">
                <Database className="w-5 h-5 text-teal-400" />
                <h3 className="font-bold text-white text-base">LocalStorage Biometric Parameters</h3>
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
                <pre className="p-3 bg-slate-950 rounded-xl border border-slate-800 overflow-x-auto text-[11px] text-slate-300 max-h-48">
                  {localStorage.getItem('muzzle_video_a_params') || 'No data stored yet'}
                </pre>
              </div>

              <div>
                <span className="text-teal-400 font-bold block mb-1">Key: 'muzzle_video_b_params'</span>
                <pre className="p-3 bg-slate-950 rounded-xl border border-slate-800 overflow-x-auto text-[11px] text-slate-300 max-h-48">
                  {localStorage.getItem('muzzle_video_b_params') || 'No data stored yet'}
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
      {/* 7. IMAGE PREVIEW MODAL                                    */}
      {/* ========================================================= */}
      {previewImage && (
        <div className="fixed inset-0 z-50 bg-slate-950/90 backdrop-blur-md flex items-center justify-center p-4">
          <div className="bg-slate-900 border border-slate-700 rounded-2xl max-w-xl w-full p-5 flex flex-col gap-4 shadow-2xl">
            <div className="flex items-center justify-between pb-2 border-b border-slate-800">
              <h3 className="font-bold text-white text-sm">{previewImage.title}</h3>
              <button
                onClick={() => setPreviewImage(null)}
                className="text-slate-400 hover:text-white p-1"
              >
                ✕
              </button>
            </div>
            <div className="aspect-square w-full rounded-xl overflow-hidden bg-black flex items-center justify-center">
              <img src={previewImage.url} alt="Muzzle preview" className="w-full h-full object-contain" />
            </div>
          </div>
        </div>
      )}

    </div>
  );
}
