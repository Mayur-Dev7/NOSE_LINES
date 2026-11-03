import React, { useState, useEffect, useRef } from 'react';
import { Terminal, Copy, Check, Trash2, ChevronDown, ChevronUp, Zap, Cpu, Activity } from 'lucide-react';

export default function LiveTelemetryLogs({
  isProcessing = false,
  elapsedSeconds = 0,
  device = 'gpu',
  className = '',
}) {
  const [logs, setLogs] = useState([]);
  const [autoScroll, setAutoScroll] = useState(true);
  const [copied, setCopied] = useState(false);
  const [isExpanded, setIsExpanded] = useState(true);
  const scrollRef = useRef(null);

  // Poll logs periodically during processing
  useEffect(() => {
    let intervalId = null;

    const fetchLogs = async () => {
      try {
        const res = await fetch('/api/logs/recent?limit=150');
        if (res.ok) {
          const data = await res.json();
          if (Array.isArray(data.logs)) {
            setLogs(data.logs);
          }
        }
      } catch {
        // Silently ignore network polling hiccups
      }
    };

    fetchLogs();

    if (isProcessing) {
      intervalId = setInterval(fetchLogs, 350);
    } else {
      // Fetch once more when done
      fetchLogs();
    }

    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [isProcessing]);

  // Auto-scroll to bottom on new log line
  useEffect(() => {
    if (autoScroll && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs, autoScroll]);

  const handleCopyLogs = () => {
    const text = logs
      .map(l => `[${l.timestamp}] [${l.level}] ${l.message}`)
      .join('\n');
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleClearLogs = async () => {
    try {
      await fetch('/api/logs/clear', { method: 'POST' });
      setLogs([]);
    } catch {
      setLogs([]);
    }
  };

  const getBadgeColor = (level, msg = '') => {
    const l = (level || '').toUpperCase();
    const m = (msg || '').toLowerCase();
    if (m.includes('gpu') || m.includes('cuda') || m.includes('tensor')) {
      return 'bg-blue-500/20 text-blue-700 dark:text-blue-300 border-blue-500/40';
    }
    if (l === 'ERROR') {
      return 'bg-rose-500/20 text-rose-700 dark:text-rose-300 border-rose-500/40';
    }
    if (l === 'WARNING') {
      return 'bg-amber-500/20 text-amber-700 dark:text-amber-300 border-amber-500/40';
    }
    if (m.includes('completed') || m.includes('success') || m.includes('early exit')) {
      return 'bg-sky-500/20 text-sky-700 dark:text-sky-300 border-sky-500/40';
    }
    return 'bg-indigo-500/20 text-indigo-700 dark:text-indigo-300 border-indigo-500/30';
  };

  return (
    <div className={`backdrop-blur-2xl bg-white/90 dark:bg-slate-950/85 border border-black/[0.08] dark:border-white/[0.08] shadow-2xl shadow-black/10 dark:shadow-black/60 rounded-3xl overflow-hidden transition-all relative ${className}`}>
      {/* Top macOS-style Terminal Bar */}
      <div className="flex items-center justify-between px-4 py-3 bg-slate-100 dark:bg-slate-900/80 border-b border-black/[0.06] dark:border-white/[0.06] select-none flex-wrap gap-2">
        <div className="flex items-center gap-3">
          {/* Mac window dots */}
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-rose-500/70 inline-block" />
            <span className="w-2.5 h-2.5 rounded-full bg-amber-500/70 inline-block" />
            <span className="w-2.5 h-2.5 rounded-full bg-blue-500/70 inline-block" />
          </div>

          <div className="flex items-center gap-2">
            <Terminal className="w-3.5 h-3.5 text-slate-500 dark:text-slate-400" />
            <span className="text-xs font-semibold text-slate-900 dark:text-slate-200 tracking-tight font-sans">
              Live Biometric Telemetry &amp; System Execution Logs
            </span>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-black/[0.05] dark:bg-white/[0.06] border border-black/[0.08] dark:border-white/[0.08] text-slate-600 dark:text-slate-400">
              {logs.length} events
            </span>
          </div>
        </div>

        {/* Right side live status & actions */}
        <div className="flex items-center gap-2">
          {isProcessing ? (
            <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-blue-500/15 border border-blue-500/30 text-blue-700 dark:text-blue-300 text-[11px] font-mono">
              <span className="w-2 h-2 rounded-full bg-blue-500 dark:bg-blue-400 animate-ping" />
              <span>LIVE RUNNING: {elapsedSeconds.toFixed(1)}s</span>
            </div>
          ) : (
            <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-slate-200 dark:bg-slate-800/80 border border-black/[0.06] dark:border-white/[0.06] text-slate-600 dark:text-slate-400 text-[11px] font-mono">
              <span className="w-1.5 h-1.5 rounded-full bg-slate-500" />
              <span>STANDBY</span>
            </div>
          )}

          {/* Auto-scroll toggle */}
          <button
            type="button"
            onClick={() => setAutoScroll(!autoScroll)}
            className={`text-[10px] font-mono px-2 py-1 rounded-lg border transition-all ${
              autoScroll
                ? 'bg-blue-500/20 text-blue-700 dark:text-blue-300 border-blue-500/40'
                : 'bg-black/[0.04] dark:bg-white/[0.04] text-slate-500 dark:text-slate-400 border-transparent hover:text-slate-900 dark:hover:text-slate-200'
            }`}
            title="Auto-scroll to latest log line"
          >
            Auto-Scroll: {autoScroll ? 'ON' : 'OFF'}
          </button>

          {/* Copy logs */}
          <button
            type="button"
            onClick={handleCopyLogs}
            disabled={logs.length === 0}
            className="p-1.5 rounded-lg bg-black/[0.04] dark:bg-white/[0.04] hover:bg-black/[0.08] dark:hover:bg-white/[0.08] text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 transition-all border border-black/[0.04] dark:border-white/[0.04] disabled:opacity-30 disabled:cursor-not-allowed"
            title="Copy logs to clipboard"
          >
            {copied ? <Check className="w-3.5 h-3.5 text-blue-600 dark:text-blue-400" /> : <Copy className="w-3.5 h-3.5" />}
          </button>

          {/* Clear logs */}
          <button
            type="button"
            onClick={handleClearLogs}
            disabled={logs.length === 0}
            className="p-1.5 rounded-lg bg-black/[0.04] dark:bg-white/[0.04] hover:bg-black/[0.08] dark:hover:bg-white/[0.08] text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 transition-all border border-black/[0.04] dark:border-white/[0.04] disabled:opacity-30 disabled:cursor-not-allowed"
            title="Clear logs"
          >
            <Trash2 className="w-3.5 h-3.5" />
          </button>

          {/* Collapse toggle */}
          <button
            type="button"
            onClick={() => setIsExpanded(!isExpanded)}
            className="p-1.5 rounded-lg bg-black/[0.04] dark:bg-white/[0.04] hover:bg-black/[0.08] dark:hover:bg-white/[0.08] text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 transition-all border border-black/[0.04] dark:border-white/[0.04]"
            title={isExpanded ? 'Collapse log viewer' : 'Expand log viewer'}
          >
            {isExpanded ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
          </button>
        </div>
      </div>

      {/* Terminal Log Console Body */}
      {isExpanded && (
        <div
          ref={scrollRef}
          className="p-4 font-mono text-xs overflow-y-auto max-h-64 min-h-[140px] flex flex-col gap-1.5 bg-slate-950/90 selection:bg-blue-600 selection:text-white"
        >
          {logs.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-8 text-slate-400 text-center gap-1.5 select-none">
              <Activity className="w-6 h-6 stroke-[1.5] text-slate-500 dark:text-slate-400 animate-pulse" />
              <span>No active telemetry logs captured yet.</span>
              <span className="text-[10px] text-slate-500 dark:text-slate-400">
                Click &ldquo;Compare 2 Videos&rdquo; or start an extraction to stream live hardware telemetry.
              </span>
            </div>
          ) : (
            logs.map((log, idx) => (
              <div key={idx} className="flex items-start gap-2.5 leading-relaxed font-mono hover:bg-white/[0.02] py-0.5 px-1 rounded transition-colors">
                <span className="text-[10px] text-slate-400 select-none shrink-0 pt-0.5">
                  {log.timestamp || '00:00:00'}
                </span>
                <span className={`text-[9px] font-bold uppercase tracking-wider px-1.5 py-0.2 rounded border shrink-0 ${getBadgeColor(log.level, log.message)}`}>
                  {log.level || 'INFO'}
                </span>
                <span className="text-slate-200 break-all text-[11px] font-mono">
                  {log.message}
                </span>
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
}
