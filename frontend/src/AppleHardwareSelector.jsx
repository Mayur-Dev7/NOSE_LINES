import React from 'react';
import { Cpu, Zap, Sparkles, Check, Activity, Layers } from 'lucide-react';

const CPU_OPTIONS = [
  { cores: 2, label: '2 Cores', est: '~4.6s', badge: 'Eco' },
  { cores: 4, label: '4 Cores', est: '~2.5s', badge: 'Balanced' },
  { cores: 6, label: '6 Physical Cores', est: '~1.8s', badge: 'Recommended', isRecommended: true },
  { cores: 12, label: '12 Threads', est: '~1.5s', badge: 'Turbo', isTurbo: true },
];

export default function AppleHardwareSelector({
  device = 'gpu',
  setDevice,
  cpuCores = 6,
  setCpuCores,
  disabled = false,
  className = '',
}) {
  const isGpu = device === 'gpu';
  const selectedCpuOption = CPU_OPTIONS.find(opt => opt.cores === cpuCores) || CPU_OPTIONS[2];

  return (
    <div className={`backdrop-blur-2xl bg-slate-900/60 border border-white/[0.08] shadow-2xl shadow-black/40 rounded-3xl p-4 sm:p-5 flex flex-col gap-3.5 transition-all relative overflow-hidden ${className}`}>
      {/* Subtle background ambient light */}
      <div
        className={`absolute -top-20 -right-20 w-56 h-56 rounded-full blur-3xl pointer-events-none transition-all duration-700 ${
          isGpu ? 'bg-emerald-500/10' : 'bg-cyan-500/10'
        }`}
      />

      {/* Top Header Row */}
      <div className="flex items-center justify-between gap-3 flex-wrap">
        <div className="flex items-center gap-3">
          <div className={`w-8 h-8 rounded-xl flex items-center justify-center transition-all ${
            isGpu
              ? 'bg-emerald-500/15 text-emerald-300 border border-emerald-500/30 shadow-[0_0_15px_rgba(16,185,129,0.25)]'
              : 'bg-cyan-500/15 text-cyan-300 border border-cyan-500/30 shadow-[0_0_15px_rgba(6,182,212,0.25)]'
          }`}>
            {isGpu ? <Zap className="w-4 h-4" /> : <Cpu className="w-4 h-4" />}
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-xs font-semibold text-white tracking-tight">
                Hardware Acceleration Engine
              </h3>
              <span className={`text-[10px] font-mono font-semibold px-2 py-0.5 rounded-full border transition-all ${
                isGpu
                  ? 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30'
                  : 'bg-cyan-500/15 text-cyan-300 border-cyan-500/30'
              }`}>
                {isGpu ? 'CUDA 12.x Active' : `Ryzen C++ (${cpuCores}T)`}
              </span>
            </div>
            <p className="text-[11px] text-slate-400 mt-0.5">
              Select compute device for Mask R-CNN localization, Frangi Hessian filtering &amp; ConvNeXt-Tiny inference.
            </p>
          </div>
        </div>

        {/* Live Active Engine Latency Badge */}
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-slate-950/70 border border-white/[0.06] text-xs font-mono shadow-inner">
          <span className={`w-1.5 h-1.5 rounded-full ${isGpu ? 'bg-emerald-400 animate-pulse' : 'bg-cyan-400'}`} />
          <span className="text-slate-400 font-sans text-[11px]">Throughput:</span>
          <strong className={isGpu ? 'text-emerald-300' : 'text-cyan-300'}>
            {isGpu ? '⚡ ~420 ms / video' : `⚡ ${selectedCpuOption.est} / video`}
          </strong>
        </div>
      </div>

      {/* Primary Segmented Pill Selector (GPU vs CPU) */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5 bg-slate-950/60 p-1.5 rounded-2xl border border-white/[0.06] shadow-inner">
        
        {/* GPU Option Button */}
        <button
          type="button"
          disabled={disabled}
          onClick={() => setDevice('gpu')}
          className={`flex items-center justify-between p-3 rounded-xl transition-all cursor-pointer text-left relative overflow-hidden group ${
            isGpu
              ? 'bg-gradient-to-r from-emerald-500/15 via-teal-500/10 to-transparent border border-emerald-500/40 text-white shadow-[0_0_20px_rgba(16,185,129,0.15)] ring-1 ring-emerald-500/20'
              : 'hover:bg-white/[0.04] text-slate-400 border border-transparent'
          } ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
        >
          <div className="flex items-center gap-3">
            <span className="text-base select-none">🚀</span>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-xs font-semibold text-white">GPU: NVIDIA GeForce RTX 3050 Laptop GPU</span>
                <span className="text-[9px] font-bold uppercase tracking-wider px-1.5 py-0.2 rounded-md bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                  Ultra Fast
                </span>
              </div>
              <span className="text-[10px] text-slate-400 block mt-0.5">
                CUDA Tensor Cores &bull; Dedicated VRAM &bull; Sub-second extraction
              </span>
            </div>
          </div>
          {isGpu && (
            <div className="w-5 h-5 rounded-full bg-emerald-500/20 text-emerald-400 flex items-center justify-center border border-emerald-500/40 shrink-0">
              <Check className="w-3 h-3 stroke-[3]" />
            </div>
          )}
        </button>

        {/* CPU Option Button */}
        <button
          type="button"
          disabled={disabled}
          onClick={() => setDevice('cpu')}
          className={`flex items-center justify-between p-3 rounded-xl transition-all cursor-pointer text-left relative overflow-hidden group ${
            !isGpu
              ? 'bg-gradient-to-r from-cyan-500/15 via-blue-500/10 to-transparent border border-cyan-500/40 text-white shadow-[0_0_20px_rgba(6,182,212,0.15)] ring-1 ring-cyan-500/20'
              : 'hover:bg-white/[0.04] text-slate-400 border border-transparent'
          } ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
        >
          <div className="flex items-center gap-3">
            <span className="text-base select-none">⚡</span>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-xs font-semibold text-white">CPU: Multi-Core Ryzen</span>
                <span className="text-[9px] font-bold uppercase tracking-wider px-1.5 py-0.2 rounded-md bg-cyan-500/20 text-cyan-300 border border-cyan-500/30">
                  Multi-Core
                </span>
              </div>
              <span className="text-[10px] text-slate-400 block mt-0.5">
                PyTorch C++ AVX2 &bull; Parallel OpenCV &bull; Configurable Cores
              </span>
            </div>
          </div>
          {!isGpu && (
            <div className="w-5 h-5 rounded-full bg-cyan-500/20 text-cyan-400 flex items-center justify-center border border-cyan-500/40 shrink-0">
              <Check className="w-3 h-3 stroke-[3]" />
            </div>
          )}
        </button>
      </div>

      {/* Sub-Pill: CPU Core Selector (Appears seamlessly when CPU is active) */}
      {!isGpu && (
        <div className="flex flex-col gap-2 pt-1 animate-in fade-in duration-300">
          <div className="flex items-center justify-between text-[11px] text-slate-400 px-1">
            <span className="flex items-center gap-1.5 font-medium text-slate-300">
              <Activity className="w-3 h-3 text-cyan-400" />
              <span>Select CPU Worker Thread Allocation:</span>
            </span>
            <span className="text-[10px] font-mono text-cyan-300">
              Active: {selectedCpuOption.label} ({selectedCpuOption.est})
            </span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 bg-slate-950/70 p-1.5 rounded-2xl border border-white/[0.06] shadow-inner">
            {CPU_OPTIONS.map((opt) => {
              const isSelected = cpuCores === opt.cores;
              return (
                <button
                  key={opt.cores}
                  type="button"
                  disabled={disabled}
                  onClick={() => setCpuCores(opt.cores)}
                  className={`flex flex-col items-center justify-center py-2 px-3 rounded-xl transition-all cursor-pointer text-center relative ${
                    isSelected
                      ? 'bg-white/10 text-white border border-white/20 shadow-md backdrop-blur-md ring-1 ring-cyan-500/30'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-white/[0.04] border border-transparent'
                  } ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
                >
                  <div className="flex items-center gap-1.5">
                    <span className="text-xs font-semibold">{opt.label}</span>
                    {opt.isRecommended && (
                      <span className="text-[8px] font-bold uppercase tracking-wider px-1 py-0.2 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                        Rec
                      </span>
                    )}
                    {opt.isTurbo && (
                      <span className="text-[8px] font-bold uppercase tracking-wider px-1 py-0.2 rounded-full bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
                        Turbo
                      </span>
                    )}
                  </div>
                  <span className={`text-[10px] font-mono mt-0.5 ${isSelected ? 'text-cyan-300 font-medium' : 'text-slate-500'}`}>
                    {opt.est}
                  </span>
                </button>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
