import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

import { useState, useEffect } from 'react';
import { Loader2 } from 'lucide-react';
import { toast } from 'sonner';
import { motion } from 'framer-motion';
import { itemVariants } from '@/lib/motion';

interface BenchmarkRow {
  name: string;
  type: string;
  gpuObj: number | string;
  cpuObj: number | string;
  gpuTime: number | null;
  cpuTime: number | null;
}

export function Benchmarks() {
  const [data, setData] = useState<BenchmarkRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const runBenchmarks = async () => {
    setLoading(true);
    setError(null);
    setData([]);
    const port = window.__FIREFLY_API_PORT__ || 8000;
    
    try {
      const resGpu = await fetch(`http://localhost:${port}/benchmark?gpu=true`);
      const gpuData = await resGpu.json();
      
      const resCpu = await fetch(`http://localhost:${port}/benchmark?gpu=false`);
      const cpuData = await resCpu.json();
      
      const merged: BenchmarkRow[] = [];
      const cpuMap = new Map(cpuData.benchmark_results.map((r: any) => [r.problem, r]));
      
      for (const gr of gpuData.benchmark_results) {
        if (gr.status === 'ERROR' && gr.problem === 'benchmark_suite') continue;
        const cr: any = cpuMap.get(gr.problem) || {};
        merged.push({
          name: gr.problem,
          type: 'Netlib/MIPLIB',
          gpuObj: gr.objective !== null && gr.objective !== undefined ? Number(gr.objective.toFixed(4)) : '---',
          cpuObj: cr.objective !== null && cr.objective !== undefined ? Number(cr.objective.toFixed(4)) : '---',
          gpuTime: gr.wall_time_ms ? Number((gr.wall_time_ms / 1000).toFixed(3)) : 0,
          cpuTime: cr.wall_time_ms ? Number((cr.wall_time_ms / 1000).toFixed(3)) : 0,
        });
      }
      
      setData(merged);
      toast.success('Benchmark suite completed.');
    } catch (e) {
      toast.error('Failed to run benchmarks.');
      setError('Solver backend unavailable. Ensure the core process is running.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    runBenchmarks();
  }, []);

  return (
    <div className="flex flex-col h-full gap-4 max-w-6xl mx-auto">
      <div className="border border-border bg-panel p-4 flex items-center justify-between shrink-0">
        <div className="flex flex-col gap-2">
          <h2 className="text-sm font-semibold tracking-wider uppercase text-text-primary">Performance Audit</h2>
          <p className="text-xs font-mono text-text-muted">
            Validation results comparing GPU-accelerated PDLP/Simplex against CPU fallback.
            Objective parity enforced within 1e-3 tolerance.
          </p>
        </div>
        <button 
          onClick={runBenchmarks} 
          disabled={loading}
          className="flex items-center gap-2 px-4 h-8 border border-border text-xs uppercase font-mono tracking-wider transition-colors hover:bg-border/50 disabled:opacity-50"
        >
          {loading ? <Loader2 className="w-3 h-3 animate-spin" /> : null}
          {loading ? 'Running...' : 'Rerun Suite'}
        </button>
      </div>

      <div className="flex-1 grid grid-cols-1 lg:grid-cols-2 gap-4 min-h-0">
        
        {/* Chart */}
        <div className="border border-border bg-panel flex flex-col">
          <div className="p-3 border-b border-border text-xs uppercase font-mono text-text-muted">
            Execution Time (s)
          </div>
          <div className="flex-1 p-4 relative">
            {loading && data.length === 0 ? (
               <div className="absolute inset-0 flex items-center justify-center bg-panel/50 text-text-muted font-mono text-xs uppercase animate-pulse">Running Benchmarks...</div>
            ) : error ? (
               <div className="absolute inset-0 flex items-center justify-center bg-panel/50 text-red-500/80 font-mono text-xs uppercase text-center px-4">{error}</div>
            ) : data.length === 0 ? (
               <div className="absolute inset-0 flex items-center justify-center bg-panel/50 text-text-muted font-mono text-xs uppercase text-center px-4">No benchmark data available.<br/>Click 'Rerun Suite' to begin execution.</div>
            ) : (
              <motion.div initial="initial" animate="animate" variants={itemVariants} className="w-full h-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={data} margin={{ top: 20, right: 30, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                  <XAxis dataKey="name" stroke="var(--text-muted)" fontSize={10} fontFamily="var(--font-mono)" tickLine={false} />
                  <YAxis stroke="var(--text-muted)" fontSize={10} fontFamily="var(--font-mono)" tickLine={false} axisLine={false} />
                  <Tooltip
                    cursor={{ fill: 'var(--border)', opacity: 0.2 }}
                    contentStyle={{ backgroundColor: 'var(--panel)', borderColor: 'var(--border)', borderRadius: '2px', fontFamily: 'var(--font-mono)', fontSize: '12px' }}
                    itemStyle={{ color: 'var(--text-primary)' }}
                  />
                  <Legend iconType="square" wrapperStyle={{ fontSize: '12px', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }} />
                  <Bar dataKey="gpuTime" name="GPU (s)" fill="var(--signal)" radius={[2, 2, 0, 0]} />
                  <Bar dataKey="cpuTime" name="CPU (s)" fill="var(--chart-trace)" radius={[2, 2, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
              </motion.div>
            )}
          </div>
        </div>

        {/* Parity Table */}
        <div className="border border-border bg-panel flex flex-col overflow-hidden">
          <div className="p-3 border-b border-border text-xs uppercase font-mono text-text-muted">
            Objective Parity Check
          </div>
          <div className="flex-1 overflow-auto relative">
            <table className="w-full text-left border-collapse text-xs font-mono">
              <thead className="bg-background text-text-muted sticky top-0 z-10">
                <tr>
                  <th className="p-3 border-b border-border font-normal uppercase whitespace-nowrap">Instance</th>
                  <th className="p-3 border-b border-border font-normal uppercase whitespace-nowrap text-right">GPU Obj</th>
                  <th className="p-3 border-b border-border font-normal uppercase whitespace-nowrap text-right">CPU Obj</th>
                  <th className="p-3 border-b border-border font-normal uppercase whitespace-nowrap text-center">Status</th>
                </tr>
              </thead>
              <tbody>
                {data.map((row, i) => {
                  const passed = Math.abs(Number(row.gpuObj) - Number(row.cpuObj)) < 1e-3 || row.gpuObj === row.cpuObj;
                  return (
                    <tr key={i} className="hover:bg-background/50 border-b border-border/50 last:border-0 transition-colors text-text-primary">
                      <td className="p-3 whitespace-nowrap">{row.name}</td>
                      <td className="p-3 text-right">{row.gpuObj}</td>
                      <td className="p-3 text-right">{row.cpuObj}</td>
                      <td className="p-3 text-center">
                        <span className={`px-2 py-0.5 border rounded-sm ${passed ? 'bg-signal/10 text-signal border-signal/20' : 'bg-red-500/10 text-red-500 border-red-500/20'}`}>
                          {passed ? 'PASS' : 'FAIL'}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            {loading && data.length === 0 && (
               <div className="absolute inset-0 flex items-center justify-center bg-panel/50 text-text-muted font-mono text-xs uppercase animate-pulse">Running Benchmarks...</div>
            )}
            {error && (
               <div className="absolute inset-0 flex items-center justify-center bg-panel/50 text-red-500/80 font-mono text-xs uppercase text-center px-4">{error}</div>
            )}
            {!loading && !error && data.length === 0 && (
               <div className="absolute inset-0 flex items-center justify-center bg-panel/50 text-text-muted font-mono text-xs uppercase text-center px-4">Awaiting Execution</div>
            )}
          </div>
        </div>

      </div>
    </div>
  );
}
