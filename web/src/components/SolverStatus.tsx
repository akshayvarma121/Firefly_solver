import { useEffect, useState } from 'react';
import { cn } from '@/lib/utils';

type SolverState = 'idle' | 'solving' | 'converged' | 'error';

interface SolverStatusLightProps {
  state: SolverState;
}

export function SolverStatusLight({ state }: SolverStatusLightProps) {
  return (
    <div className="flex items-center gap-2">
      <div className="text-text-muted text-xs font-mono uppercase tracking-wider">Solver</div>
      <div
        className={cn(
          "w-2 h-2 rotate-45 border border-border transition-colors duration-300",
          state === 'idle' && "bg-transparent",
          state === 'solving' && "bg-signal animate-pulse border-signal",
          state === 'converged' && "bg-signal border-signal"
        )}
      />
    </div>
  );
}

interface GPULaneIndicatorsProps {
  state: SolverState;
  lanes?: number;
}

export function GPULaneIndicators({ state, lanes = 8 }: GPULaneIndicatorsProps) {
  const [activeLanes, setActiveLanes] = useState<boolean[]>(Array(lanes).fill(false));

  useEffect(() => {
    if (state === 'idle') {
      setActiveLanes(Array(lanes).fill(false));
      return;
    }
    
    if (state === 'converged') {
      setActiveLanes(Array(lanes).fill(true));
      return;
    }

    // Solving state - async blinking
    const intervals = Array.from({ length: lanes }).map((_, i) => {
      const duration = 100 + Math.random() * 300;
      return setInterval(() => {
        setActiveLanes(prev => {
          const next = [...prev];
          next[i] = !next[i];
          return next;
        });
      }, duration);
    });

    return () => intervals.forEach(clearInterval);
  }, [state, lanes]);

  return (
    <div className="flex items-center gap-0.5 p-1 border border-border bg-background">
      {activeLanes.map((isActive, i) => (
        <div
          key={i}
          className={cn(
            "w-3 h-3 transition-colors duration-75 border border-border",
            isActive ? "bg-signal border-signal" : "bg-panel"
          )}
        />
      ))}
    </div>
  );
}
