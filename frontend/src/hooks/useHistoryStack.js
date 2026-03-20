import { useCallback, useState } from "react";

export function useHistoryStack(initial = null) {
  const [history, setHistory] = useState(initial ? [initial] : []);
  const [index, setIndex] = useState(initial ? 0 : -1);

  const push = useCallback((value) => {
    setHistory((prev) => {
      const next = prev.slice(0, index + 1);
      next.push(value);
      return next;
    });
    setIndex((prev) => prev + 1);
  }, [index]);

  const canUndo = index > 0;
  const canRedo = index >= 0 && index < history.length - 1;

  const undo = useCallback(() => {
    if (!canUndo) return null;
    setIndex((prev) => prev - 1);
    return history[index - 1];
  }, [canUndo, history, index]);

  const redo = useCallback(() => {
    if (!canRedo) return null;
    setIndex((prev) => prev + 1);
    return history[index + 1];
  }, [canRedo, history, index]);

  const current = index >= 0 ? history[index] : null;

  return { current, push, undo, redo, canUndo, canRedo };
}

