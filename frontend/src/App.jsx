import React, { useState, useEffect } from 'react';
import TwoVideoCosineMatcher from './TwoVideoCosineMatcher';
import { ThemeToggle } from './components/ui/theme-toggle';
import {
  Cpu,
  Zap,
  AlertCircle,
  GitCompare,
  ScanLine
} from 'lucide-react';

export default function App() {
  const [status, setStatus] = useState(null);
  const [loadingStatus, setLoadingStatus] = useState(true);

  useEffect(() => {
    fetchStatus();
  }, []);

  const fetchStatus = async () => {
    setLoadingStatus(true);
    try {
      const res = await fetch('/api/status');
      const data = await res.json();
      setStatus(data);
    } catch (err) {
      console.error('Error fetching system status:', err);
    } finally {
      setLoadingStatus(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#f5f5f7] dark:bg-[#0b0f19] text-slate-800 dark:text-slate-100 flex flex-col selection:bg-blue-600 selection:text-white transition-colors duration-200">
      {/* Top Header */}
      <header className="border-b border-black/[0.06] dark:border-white/[0.08] bg-white/80 dark:bg-slate-950/80 backdrop-blur-2xl sticky top-0 z-50 px-6 py-3.5 transition-all shadow-xs">
        <div className="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center space-x-3.5">
            <div className="w-10 h-10 rounded-2xl bg-gradient-to-tr from-blue-600 via-indigo-600 to-sky-500 flex items-center justify-center text-white font-bold shadow-lg shadow-blue-500/25 ring-1 ring-black/5 dark:ring-white/20">
              <ScanLine className="w-5 h-5 text-white" />
            </div>
            <div>
              <div className="flex items-center gap-2.5">
                <h1 className="text-base font-semibold tracking-tight text-slate-900 dark:text-white flex items-center gap-2">
                  Bovine Biometric Studio
                </h1>
                <span className="text-[10px] font-semibold tracking-wider uppercase px-2.5 py-0.5 rounded-full bg-blue-500/10 dark:bg-blue-500/20 text-blue-600 dark:text-blue-400 border border-blue-500/20 dark:border-blue-500/30 font-mono shadow-xs">
                  ConvNeXt ArcFace &bull; GeM 512-D
                </span>
              </div>
              <p className="text-[11px] text-slate-500 dark:text-slate-400 font-normal mt-0.5">
                5-Frame Temporal Muzzle Pooling &bull; Epidermal Dermatoglyphic Ridge Skeletonization &bull; 10-D Minutiae Topology
              </p>
            </div>
          </div>

          {/* System Hardware Status Badges & Theme Toggle */}
          <div className="flex items-center gap-2.5">
            <ThemeToggle />
            {status?.hardware?.gpu_available ? (
              <div className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-blue-500/10 dark:bg-blue-500/15 border border-blue-500/25 dark:border-blue-500/30 text-blue-700 dark:text-blue-300 text-xs font-medium backdrop-blur-md">
                <span className="w-1.5 h-1.5 rounded-full bg-blue-500 dark:bg-blue-400 animate-pulse"></span>
                <Zap className="w-3.5 h-3.5 text-blue-600 dark:text-blue-400" />
                <span>GPU: {status.hardware.gpu_name}</span>
              </div>
            ) : (
              <div className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-amber-500/10 border border-amber-500/20 text-amber-700 dark:text-amber-300 text-xs font-medium backdrop-blur-md">
                <span className="w-1.5 h-1.5 rounded-full bg-amber-500 dark:bg-amber-400"></span>
                <AlertCircle className="w-3.5 h-3.5 text-amber-600 dark:text-amber-400" />
                <span>GPU Standby</span>
              </div>
            )}
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-black/[0.04] dark:bg-white/[0.04] border border-black/[0.06] dark:border-white/[0.08] text-slate-700 dark:text-slate-300 text-xs font-medium backdrop-blur-md">
              <Cpu className="w-3.5 h-3.5 text-sky-500 dark:text-sky-400" />
              <span>{status?.hardware?.cpu_threads || 12} Threads</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Container */}
      <main className="max-w-7xl mx-auto w-full p-6 flex-1 flex flex-col gap-6">
        {/* Biometric Verification Workspace Header Sub-Bar */}
        <div className="flex flex-wrap items-center justify-between border-b border-black/[0.06] dark:border-white/[0.08] pb-3 gap-3">
          <div className="flex items-center gap-2">
            <span className="flex items-center gap-2 px-3 py-1.5 rounded-xl text-xs font-semibold bg-white dark:bg-white/10 text-slate-900 dark:text-white shadow-xs border border-black/[0.06] dark:border-white/15 backdrop-blur-md">
              <GitCompare className="w-3.5 h-3.5 text-blue-600 dark:text-blue-400" />
              <span>Dual-Video Biometric Verification</span>
              <span className="text-[9px] font-bold tracking-wider uppercase px-1.5 py-0.5 rounded-md bg-blue-500/15 dark:bg-blue-500/25 text-blue-700 dark:text-blue-300 font-mono">
                Flagship
              </span>
            </span>
          </div>

          <div className="flex items-center gap-2 text-[11px] font-mono text-slate-500 dark:text-slate-400">
            <span className="w-1.5 h-1.5 rounded-full bg-blue-500 animate-pulse"></span>
            Weights: <code className="text-slate-700 dark:text-slate-300">crop11dec.h5 ({status?.weights_size_mb || 245} MB)</code>
          </div>
        </div>

        {/* PRIMARY AND ONLY PAGE: 2-VIDEO 5-IMAGE COSINE MATCHER */}
        <TwoVideoCosineMatcher sampleVideos={status?.sample_videos || []} />
      </main>

      {/* Footer */}
      <footer className="border-t border-black/[0.06] dark:border-white/[0.06] px-6 py-4 text-center text-xs text-slate-500 dark:text-slate-400">
        Matterport Mask R-CNN Cattle Muzzle Detection &amp; statusprint.py ORB Verification &bull; Lanczos-4 Sinc Super-Resolution
      </footer>
    </div>
  );
}
