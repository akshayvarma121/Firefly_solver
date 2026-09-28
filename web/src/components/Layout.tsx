import { Outlet, NavLink, useLocation } from 'react-router-dom';
import { AnimatePresence, motion } from 'framer-motion';
import { pageVariants } from '@/lib/motion';
import { SolverStatusLight } from './SolverStatus';
import { useSolver } from '@/context/SolverContext';
import { Play, BarChart2, Cpu, Grid } from 'lucide-react';
import { cn } from '@/lib/utils';
import { toast } from 'sonner';

export function Layout() {
  const { state, backendMode } = useSolver();
  const location = useLocation();

  const handleModeClick = () => {
    if (backendMode === 'CPU') {
      toast('No compatible NVIDIA GPU/driver detected — running on CPU', {
        icon: '⚠️',
      });
    }
  };

  return (
    <div className="min-h-screen flex flex-col font-sans bg-background text-text-primary overflow-hidden">
      {/* Top Bar */}
      <header className="h-12 border-b border-border bg-panel flex items-center justify-between px-4 shrink-0">
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2">
            <Grid className="w-4 h-4 text-signal" />
            <h1 className="text-sm font-semibold tracking-wider uppercase">Firefly</h1>
          </div>
          <div className="w-px h-4 bg-border" />
          <SolverStatusLight state={state} />
          {backendMode && (
            <button
              onClick={handleModeClick}
              className={cn(
                "ml-2 text-xs font-mono px-1.5 py-0.5 border rounded-sm transition-colors",
                backendMode === 'GPU' 
                  ? "bg-signal/10 border-signal/20 text-signal" 
                  : "bg-text-muted/10 border-text-muted/20 text-text-muted hover:border-text-muted/50 cursor-pointer"
              )}
              style={{ cursor: backendMode === 'GPU' ? 'default' : 'pointer' }}
            >
              {backendMode}
            </button>
          )}
        </div>
        <div className="text-xs font-mono text-text-muted flex items-center gap-2">
          <span>v0.1.0</span>
          <span className="w-1 h-1 bg-border rounded-full" />
          <span>CUDA sm_89</span>
        </div>
      </header>

      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        <aside className="w-48 border-r border-border bg-background flex flex-col shrink-0">
          <nav className="flex-1 p-2 space-y-1">
            <NavItem to="/solve" icon={<Play className="w-4 h-4" />} label="Solve" />
            <NavItem to="/benchmarks" icon={<BarChart2 className="w-4 h-4" />} label="Benchmarks" />
            <NavItem to="/architecture" icon={<Cpu className="w-4 h-4" />} label="Architecture" />
          </nav>
          <div className="p-4 border-t border-border text-xs font-mono text-text-muted uppercase tracking-wider leading-relaxed opacity-50">
            Firefly v0.1.0<br/>Built by The Fireflies
          </div>
        </aside>

        {/* Main Content Area */}
        <main className="flex-1 overflow-auto bg-background p-4 relative">
          {/* Responsive warning */}
          <div className="hidden max-[1024px]:flex absolute inset-0 z-50 bg-background/80 backdrop-blur-sm items-center justify-center p-8 text-center">
            <div className="border border-border bg-panel p-4 max-w-sm">
              <h2 className="text-sm font-semibold text-signal mb-2">Resolution Warning</h2>
              <p className="text-xs text-text-muted">Firefly Core UI is optimized for 1366x768 or higher (ideal at 1920x1080). Please maximize your window.</p>
            </div>
          </div>
          
          <AnimatePresence mode="wait">
            <motion.div
              key={location.pathname}
              initial="initial"
              animate="animate"
              exit="exit"
              variants={pageVariants}
              className="h-full w-full"
            >
              <Outlet />
            </motion.div>
          </AnimatePresence>
        </main>
      </div>
    </div>
  );
}

function NavItem({ to, icon, label }: { to: string, icon: React.ReactNode, label: string }) {
  return (
    <NavLink
      to={to}
      className={({ isActive }) => cn(
        "flex items-center gap-3 px-3 py-2 text-sm transition-colors border border-transparent",
        isActive 
          ? "bg-panel border-border text-text-primary" 
          : "text-text-muted hover:text-text-primary hover:bg-panel/50"
      )}
    >
      {icon}
      <span>{label}</span>
    </NavLink>
  );
}
