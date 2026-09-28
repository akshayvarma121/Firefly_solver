export const sharedTransition = {
  duration: 0.3,
  ease: [0.16, 1, 0.3, 1]
};

export const pageVariants = {
  initial: { opacity: 0, y: 4 },
  animate: { opacity: 1, y: 0, transition: sharedTransition },
  exit: { opacity: 0, y: -4, transition: sharedTransition }
};

export const staggerContainer = {
  animate: { transition: { staggerChildren: 0.05 } }
};

export const itemVariants = {
  initial: { opacity: 0, y: 4 },
  animate: { opacity: 1, y: 0, transition: sharedTransition }
};
