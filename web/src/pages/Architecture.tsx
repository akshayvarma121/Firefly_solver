import { Database, FileCode2, Cpu, LineChart } from 'lucide-react';
import { motion } from 'framer-motion';
import { itemVariants, staggerContainer } from '@/lib/motion';

export function Architecture() {
  const steps = [
    { num: 1, title: 'MPS Parser', desc: 'Reads .mps input. Optimized C++ string processing into sparse CSR format.', icon: <FileCode2 className="w-5 h-5" /> },
    { num: 2, title: 'Presolve', desc: 'Redundant row removal, fixed variable substitution, bounds tightening, Ruiz equilibration.', icon: <Database className="w-5 h-5" /> },
    { num: 3, title: 'CUDA Core', desc: 'PDLP algorithm executed on GPU sm_89. Concurrent block reductions and matrix-vector products.', icon: <Cpu className="w-5 h-5" /> },
    { num: 4, title: 'Simplex / MILP', desc: 'CPU fallback for anti-cycling, unbounded detection, or branch-and-bound logic.', icon: <LineChart className="w-5 h-5" /> }
  ];

  return (
    <div className="flex flex-col h-full gap-6 max-w-4xl mx-auto items-center justify-center p-4">
      <div className="text-center space-y-2 mb-4">
        <h2 className="text-sm font-semibold tracking-wider uppercase text-text-primary">Solver Pipeline</h2>
        <p className="text-xs font-mono text-text-muted max-w-xl mx-auto">
          Firefly implements an end-to-end mathematical optimization pipeline built from first principles, 
          leveraging CUDA C++20 for massively parallel compute.
        </p>
      </div>

      <div className="relative w-full">
        {/* Connecting line */}
        <div className="absolute top-1/2 left-0 w-full h-px bg-border -translate-y-1/2 hidden md:block" />

        <motion.div 
          className="grid grid-cols-1 md:grid-cols-4 gap-4 relative z-10"
          variants={staggerContainer}
          initial="initial"
          animate="animate"
        >
          {steps.map((step, idx) => (
            <motion.div key={idx} variants={itemVariants} className="bg-panel border border-border p-4 flex flex-col relative group hover:border-signal transition-colors">
              <div className="flex items-start justify-between mb-4">
                <div className="w-8 h-8 flex items-center justify-center bg-background border border-border text-signal font-mono text-xs">
                  0{step.num}
                </div>
                <div className="text-text-muted group-hover:text-text-primary transition-colors">
                  {step.icon}
                </div>
              </div>
              <h3 className="text-xs font-semibold tracking-wider uppercase text-text-primary mb-2">{step.title}</h3>
              <p className="text-xs font-mono text-text-muted leading-relaxed">
                {step.desc}
              </p>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </div>
  );
}
