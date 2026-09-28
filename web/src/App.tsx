import { HashRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { invoke } from '@tauri-apps/api/core';

declare global {
  interface Window {
    __FIREFLY_API_PORT__?: number;
  }
}
import { Toaster } from 'sonner';
import { Layout } from '@/components/Layout';
import { Solve } from '@/pages/Solve';
import { Benchmarks } from '@/pages/Benchmarks';
import { Architecture } from '@/pages/Architecture';
import { useSolver } from '@/context/SolverContext';
import { Grid, Loader2, AlertTriangle } from 'lucide-react';
import { useState, useEffect } from 'react';

const isTauri = '__TAURI__' in window || '__TAURI_INTERNALS__' in window;

function BootLoaderScreen() {
  return (
    <div className="h-screen w-screen bg-background flex flex-col items-center justify-center text-text-primary">
      <Grid className="w-12 h-12 text-signal mb-6 animate-pulse" />
      <h1 className="text-sm font-semibold tracking-wider uppercase mb-2">Firefly Core</h1>
      <div className="flex items-center gap-3 text-text-muted font-mono text-sm">
        <Loader2 className="w-4 h-4 animate-spin" />
        Initializing solver backend...
      </div>
    </div>
  );
}

function BootErrorScreen() {
  return (
    <div className="h-screen w-screen bg-background flex flex-col items-center justify-center text-text-primary p-4">
      <AlertTriangle className="w-12 h-12 text-red-500 mb-6" />
      <h1 className="text-sm font-semibold tracking-wider uppercase mb-2">Initialization Failed</h1>
      <p className="text-text-muted font-mono text-sm text-center max-w-md border border-border bg-panel p-4">
        The embedded Firefly solver backend failed to respond within 10 seconds. 
        Please verify that the sidecar process was bundled correctly or check the system logs.
      </p>
    </div>
  );
}

function App() {
  const { setBackendMode } = useSolver();
  const [bootState, setBootState] = useState<'loading' | 'ready' | 'error'>(isTauri ? 'loading' : 'ready');

  useEffect(() => {
    let active = true;
    
    if (!isTauri) {
      window.__FIREFLY_API_PORT__ = 8000;
      fetch('http://localhost:8000/health')
        .then(r => r.json())
        .then(d => { if (active) setBackendMode(d.mode.includes('GPU') ? 'GPU' : 'CPU'); })
        .catch(() => {});
      return () => { active = false; };
    }

    // In Tauri, poll until ready
    let attempts = 0;
    const maxAttempts = 40; // 10s max (250ms interval)
    
    const interval = setInterval(async () => {
      try {
        let port = window.__FIREFLY_API_PORT__;
        if (!port) {
          port = await invoke<number>('get_api_port');
          window.__FIREFLY_API_PORT__ = port;
        }
        const res = await fetch(`http://localhost:${port}/health`);
        if (res.ok) {
          const data = await res.json();
          if (active) {
            setBackendMode(data.mode.includes('GPU') ? 'GPU' : 'CPU');
            setBootState('ready');
          }
          clearInterval(interval);
        }
      } catch (e) {
        attempts++;
        if (attempts >= maxAttempts) {
          if (active) setBootState('error');
          clearInterval(interval);
        }
      }
    }, 250);

    return () => {
      active = false;
      clearInterval(interval);
    };
  }, [setBackendMode]);

  if (bootState === 'loading') return <BootLoaderScreen />;
  if (bootState === 'error') return <BootErrorScreen />;

  return (
    <Router>
      <Toaster 
        theme="dark" 
        toastOptions={{ 
          className: 'bg-panel border-border text-text-primary font-sans rounded-sm',
          descriptionClassName: 'text-text-muted font-mono text-xs',
        }} 
      />
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Navigate to="/solve" replace />} />
          <Route path="solve" element={<Solve />} />
          <Route path="benchmarks" element={<Benchmarks />} />
          <Route path="architecture" element={<Architecture />} />
        </Route>
      </Routes>
    </Router>
  );
}

export default App;
