import { useState, useEffect, useRef, useCallback } from 'react';
import { useSolver } from '@/context/SolverContext';
import { GPULaneIndicators } from '@/components/SolverStatus';
import { Play, Square, Upload, FileText } from 'lucide-react';
import { cn } from '@/lib/utils';
import { toast } from 'sonner';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts';
import { motion, AnimatePresence } from 'framer-motion';
import { itemVariants } from '@/lib/motion';

type TraceEvent = {
  stage: string;
  narration: string;
  [key: string]: any;
};

type WSMessage = {
  type: 'status' | 'trace' | 'data' | 'error';
  content?: string;
  stage?: string;
  narration?: string;
  data?: { iteration: number; primal: number; dual: number };
  state?: 'idle' | 'solving' | 'converged' | 'error';
  message?: string;
  [key: string]: any;
};

const SAMPLES = ['afiro.mps', 'flugpl.mps', 'egout.mps', 'blend.mps'];

export function Solve() {
  const { state, setState, backendMode } = useSolver();
  const [data, setData] = useState<any[]>([]);
  const [traces, setTraces] = useState<TraceEvent[]>([]);
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [isConnecting, setIsConnecting] = useState(false);
  
  const [metadata, setMetadata] = useState<{vars: number, constrs: number, nonzeros: number, obj?: string} | null>(null);
  const [inspectLoading, setInspectLoading] = useState(false);
  const [inspectError, setInspectError] = useState<string | null>(null);
  const [finalObjective, setFinalObjective] = useState<string | null>(null);
  
  const ws = useRef<WebSocket | null>(null);
  const traceEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    traceEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [traces]);

  const [uploadedMpsContent, setUploadedMpsContent] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleUploadChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (evt) => {
      const content = evt.target?.result as string;
      setUploadedMpsContent(content);
      setSelectedFile(file.name);
      setState('idle');
      setData([]);
      setTraces([]);
      setFinalObjective(null);
      loadMetadata(file.name, content);
    };
    reader.readAsText(file);
  };

  const loadMetadata = async (file: string, mpsContent: string | null) => {
    setInspectLoading(true);
    setInspectError(null);
    setMetadata(null);
    setFinalObjective(null);
    try {
      const port = window.__FIREFLY_API_PORT__ || 8000;
      const res = await fetch(`http://localhost:${port}/inspect`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(mpsContent ? { mps_content: mpsContent } : { filename: file })
      });
      if (!res.ok) throw new Error('Solver core unavailable or parsing failed.');
      const d = await res.json();
      const nonzerosMatch = d.problem_summary.match(/(\d+) nonzeros/);
      setMetadata({
        vars: d.vars,
        constrs: d.constrs,
        nonzeros: nonzerosMatch ? parseInt(nonzerosMatch[1]) : 0
      });
    } catch (e: any) {
      setInspectError(e.message || 'Inspection failed.');
    } finally {
      setInspectLoading(false);
    }
  };

  useEffect(() => {
    return () => {
      if (ws.current) ws.current.close();
    };
  }, []);

  const handleConnect = useCallback(() => {
    if (!selectedFile) {
      toast.error('Please select or upload a model first.');
      return;
    }
    
    if (state === 'solving') {
      ws.current?.close();
      setState('idle');
      return;
    }

    setIsConnecting(true);
    setState('idle');
    setData([]);
    setFinalObjective(null);
    setTraces([
      { stage: 'system', narration: 'Initializing CUDA context...' },
      { stage: 'system', narration: `Loading ${selectedFile}...` }
    ]);

    try {
      const port = window.__FIREFLY_API_PORT__ || 8000;
      ws.current = new WebSocket(`ws://localhost:${port}/ws/solve-stream`);
      
      ws.current.onopen = () => {
        setIsConnecting(false);
        setState('solving');
        toast.success('Connected to solver engine.');
        if (uploadedMpsContent) {
          ws.current?.send(JSON.stringify({ action: 'start', mps_content: uploadedMpsContent, gpu: backendMode === 'GPU' }));
        } else {
          ws.current?.send(JSON.stringify({ action: 'start', filename: selectedFile, gpu: backendMode === 'GPU' }));
        }
      };

      ws.current.onmessage = (event) => {
        try {
          const msg: WSMessage = JSON.parse(event.data);
          if (msg.type === 'status' && msg.state) {
            setState(msg.state);
            if (msg.state === 'converged') toast.success('Solve converged optimally.');
            if (msg.state === 'error') toast.error('Solver encountered an error.');
          }
          if (msg.type === 'trace' && msg.narration) {
            setTraces(prev => [...prev, { stage: msg.stage || 'unknown', narration: msg.narration!, ...msg }]);
            const match = msg.narration!.match(/objective\s+([-\d.eE]+)/i);
            if (match) setFinalObjective(match[1]);
          }
          if (msg.type === 'data' && msg.data) {
            setData(prev => [...prev, msg.data]);
          }
          if (msg.type === 'error') {
            setState('error');
            toast.error(`Solver error: ${msg.message || 'Unknown'}`);
            setTraces(prev => [...prev, { stage: 'error', narration: `ERROR: ${msg.message}` }]);
          }
        } catch (e) {
          // Silent catch in production
        }
      };

      ws.current.onerror = () => {
        setIsConnecting(false);
        setState('error');
        toast.error('WebSocket connection failed. Ensure backend is running.');
      };

      ws.current.onclose = () => {
        setState(prev => prev === 'solving' ? 'idle' : prev);
      };
    } catch (e) {
      toast.error('Failed to initialize WebSocket.');
      setIsConnecting(false);
    }
  }, [selectedFile, state, backendMode, uploadedMpsContent, setState]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        if (selectedFile && state !== 'solving' && !isConnecting) {
          e.preventDefault();
          handleConnect();
        }
      }
      if (e.key === 'Escape') {
        if (document.activeElement instanceof HTMLElement) {
          document.activeElement.blur();
        }
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleConnect, selectedFile, state, isConnecting]);



  return (
    <div className="flex flex-col h-full gap-4">
      {/* Top Controls */}
      <div className="flex gap-4 items-stretch h-12">
        <div className="flex-1 flex items-center border border-border bg-panel px-4 gap-4">
          <div className="flex items-center gap-2">
            <span className="text-xs uppercase text-text-muted font-mono">Input:</span>
            <select 
              className="bg-background border border-border text-sm px-2 py-1 outline-none focus:border-signal font-mono h-8"
              value={selectedFile || ''}
              onChange={(e) => {
                const val = e.target.value;
                setSelectedFile(val);
                setUploadedMpsContent(null);
                setState('idle');
                setData([]);
                setTraces([]);
                setFinalObjective(null);
                loadMetadata(val, null);
              }}
              disabled={state === 'solving'}
            >
              <option value="" disabled>Select sample...</option>
              {SAMPLES.map(s => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>
          <div className="w-px h-4 bg-border" />
          <input 
            type="file" 
            accept=".mps" 
            className="hidden" 
            ref={fileInputRef} 
            onChange={handleUploadChange} 
          />
          <button 
            disabled={state === 'solving'}
            onClick={() => fileInputRef.current?.click()}
            className="flex items-center gap-2 text-xs uppercase font-mono text-text-muted hover:text-text-primary transition-colors h-8 px-2 border border-transparent hover:border-border"
          >
            <Upload className="w-3 h-3" />
            Upload .mps
          </button>
        </div>

        <div className="flex items-center border border-border bg-panel px-4 gap-4">
          <GPULaneIndicators state={state} lanes={16} />
          
          <button
            onClick={handleConnect}
            disabled={isConnecting}
            className={cn(
              "flex items-center gap-2 px-4 h-8 border border-border text-xs uppercase font-mono tracking-wider transition-colors hover:bg-border/50",
              state === 'solving' && "text-signal border-signal hover:bg-signal/10"
            )}
          >
            {state === 'solving' ? (
              <><Square className="w-3 h-3" />Abort</>
            ) : isConnecting ? (
              <span className="animate-pulse">Connecting...</span>
            ) : (
              <><Play className="w-3 h-3" />{state === 'converged' ? 'Restart' : 'Solve'}</>
            )}
          </button>
        </div>
      </div>

      {/* Main Workspace */}
      <div className="flex-1 grid grid-cols-1 lg:grid-cols-3 gap-4 min-h-0">
        
        {/* Left: Model Stats (Skeleton / Empty state supported) */}
        <div className="border border-border bg-panel flex flex-col">
          <div className="p-3 border-b border-border text-xs uppercase font-mono text-text-muted">Model Context</div>
          <div className="p-4 flex-1 flex flex-col justify-center min-h-48">
            {!selectedFile ? (
              <div className="flex flex-col items-center justify-center text-text-muted gap-2 opacity-50">
                <FileText className="w-8 h-8" />
                <span className="text-xs uppercase font-mono text-center">No Model Selected<br/>Upload or pick a sample</span>
              </div>
            ) : inspectLoading ? (
              <div className="space-y-4 w-full">
                <div className="h-4 bg-border/30 w-3/4 animate-pulse" />
                <div className="h-4 bg-border/30 w-1/2 animate-pulse" />
                <div className="h-4 bg-border/30 w-2/3 animate-pulse" />
              </div>
            ) : inspectError ? (
              <div className="flex flex-col items-center justify-center text-red-500/80 gap-2 text-center">
                <span className="text-xs font-mono">{inspectError}</span>
              </div>
            ) : metadata ? (
              <motion.div 
                initial="initial"
                animate="animate"
                variants={itemVariants}
                className="space-y-3 font-mono text-sm w-full"
              >
                <div className="flex justify-between"><span className="text-text-muted">Rows</span><span>{metadata.constrs.toLocaleString()}</span></div>
                <div className="flex justify-between"><span className="text-text-muted">Cols</span><span>{metadata.vars.toLocaleString()}</span></div>
                <div className="flex justify-between"><span className="text-text-muted">NonZeros</span><span>{metadata.nonzeros.toLocaleString()}</span></div>
                <div className="w-full h-px bg-border my-4" />
                <div className="flex justify-between"><span className="text-text-muted">Objective</span><span className={finalObjective ? 'text-signal' : ''}>{finalObjective || '---'}</span></div>
              </motion.div>
            ) : null}
          </div>
        </div>

        {/* Right: Charts and Trace */}
        <div className="lg:col-span-2 flex flex-col gap-4 min-h-0">
          
          {/* Chart */}
          <div className="flex-[2] border border-border bg-panel flex flex-col min-h-0">
            <div className="p-3 border-b border-border text-xs uppercase font-mono text-text-muted flex justify-between">
              <span>Convergence Trajectory</span>
              <div className="flex gap-3">
                <span className="flex items-center gap-1"><div className="w-2 h-2 bg-signal rounded-sm" /> Primal</span>
                <span className="flex items-center gap-1"><div className="w-2 h-2 bg-chart-trace rounded-sm" /> Dual</span>
              </div>
            </div>
            <div className="flex-1 p-4 min-h-48 relative">
              {state === 'error' ? (
                <div className="absolute inset-0 flex items-center justify-center text-red-500/80 text-xs uppercase font-mono">
                  Solver Core Offline
                </div>
              ) : data.length === 0 ? (
                <div className="absolute inset-0 flex items-center justify-center text-text-muted text-xs uppercase font-mono">
                  {state === 'idle' ? 'Awaiting solve initialization...' : 'Awaiting telemetry...'}
                </div>
              ) : (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={data}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                    <XAxis dataKey="iteration" stroke="var(--text-muted)" fontSize={10} fontFamily="var(--font-mono)" tickLine={false} />
                    <YAxis stroke="var(--text-muted)" fontSize={10} fontFamily="var(--font-mono)" tickLine={false} axisLine={false} />
                    <Tooltip
                      contentStyle={{ backgroundColor: 'var(--panel)', borderColor: 'var(--border)', borderRadius: '2px', fontFamily: 'var(--font-mono)', fontSize: '12px' }}
                      itemStyle={{ color: 'var(--text-primary)' }}
                    />
                    <Line type="monotone" dataKey="primal" stroke="var(--signal)" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                    <Line type="monotone" dataKey="dual" stroke="var(--chart-trace)" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                  </LineChart>
                </ResponsiveContainer>
              )}
            </div>
          </div>

          {/* Trace Narration */}
          <div className="flex-1 border border-border bg-panel flex flex-col min-h-40">
            <div className="p-3 border-b border-border text-xs uppercase font-mono text-text-muted">Execution Trace</div>
            <div className="flex-1 p-4 overflow-y-auto font-mono text-xs text-text-muted space-y-1">
              {traces.length === 0 ? (
                <span className="opacity-50">System idle.</span>
              ) : (
                <AnimatePresence initial={false}>
                  {traces.map((t, i) => (
                    <motion.div 
                      key={i}
                      initial="initial"
                      animate="animate"
                      variants={itemVariants}
                      className={cn(
                        "flex flex-col gap-1 py-1 border-b border-border/30 last:border-b-0",
                        t.stage === 'error' && "border-red-500/30"
                      )}
                    >
                      <div className={cn("whitespace-pre-wrap flex gap-3", t.narration.includes('Optimal') ? "text-signal" : t.narration.includes('ERR') ? "text-red-500" : "text-text-primary")}>
                        <span className="w-16 shrink-0 opacity-50 uppercase tracking-wider">[{t.stage}]</span>
                        <span>{t.narration}</span>
                      </div>
                      
                      <div className="flex flex-wrap gap-x-6 gap-y-1 text-[10px] text-text-muted font-mono ml-[4.75rem]">
                        {t.stage === 'parse' && t.vars !== undefined && (
                          <>
                            <span>VARS: <span className="text-text-primary">{t.vars}</span></span>
                            <span>CONSTRS: <span className="text-text-primary">{t.constrs}</span></span>
                            {t.is_milp !== undefined && <span>MILP: <span className="text-text-primary">{t.is_milp ? 'YES' : 'NO'}</span></span>}
                            {t.sense && <span>SENSE: <span className="text-text-primary">{t.sense.toUpperCase()}</span></span>}
                          </>
                        )}
                        
                        {t.stage === 'presolve' && t.original_rows !== undefined && (
                          <>
                            <span>REMOVED_ROWS: <span className="text-text-primary">{t.rows_removed}</span></span>
                            <span>FIXED_VARS: <span className="text-text-primary">{t.variables_fixed}</span></span>
                            <span>BOUNDS_TIGHTENED: <span className="text-text-primary">{t.bounds_tightened}</span></span>
                            <span>ORIG_DIM: <span className="text-text-primary">{t.original_rows}x{t.original_cols}</span></span>
                            <span>NEW_DIM: <span className="text-text-primary">{t.reduced_rows}x{t.reduced_cols}</span></span>
                          </>
                        )}
                        
                        {t.stage === 'output' && t.status && (
                          <>
                            <span>STATUS: <span className={t.status === 'OPTIMAL' ? 'text-signal' : 'text-text-primary'}>{t.status}</span></span>
                            {t.objective !== undefined && <span>OBJECTIVE: <span className="text-text-primary">{t.objective}</span></span>}
                            {t.iterations !== undefined && <span>ITERATIONS: <span className="text-text-primary">{t.iterations}</span></span>}
                            {t.time_ms !== undefined && <span>TIME: <span className="text-text-primary">{Number(t.time_ms).toFixed(2)}ms</span></span>}
                          </>
                        )}
                      </div>
                    </motion.div>
                  ))}
                </AnimatePresence>
              )}
              {state === 'solving' && <div className="animate-pulse">&gt; _</div>}
              <div ref={traceEndRef} />
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}
