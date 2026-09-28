import { createContext, useContext, useState, ReactNode } from 'react';

export type SolverState = 'idle' | 'solving' | 'converged' | 'error';

interface SolverContextProps {
  state: SolverState;
  setState: React.Dispatch<React.SetStateAction<SolverState>>;
  backendMode: 'GPU' | 'CPU' | null;
  setBackendMode: React.Dispatch<React.SetStateAction<'GPU' | 'CPU' | null>>;
}

const SolverContext = createContext<SolverContextProps | undefined>(undefined);

export function SolverProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<SolverState>('idle');
  const [backendMode, setBackendMode] = useState<'GPU' | 'CPU' | null>(null);
  
  return (
    <SolverContext.Provider value={{ state, setState, backendMode, setBackendMode }}>
      {children}
    </SolverContext.Provider>
  );
}

export function useSolver() {
  const ctx = useContext(SolverContext);
  if (!ctx) throw new Error("useSolver must be used within a SolverProvider");
  return ctx;
}
