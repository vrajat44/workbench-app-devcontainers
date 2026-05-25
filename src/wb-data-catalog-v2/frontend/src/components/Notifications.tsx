import {
  createContext,
  useCallback,
  useContext,
  useState,
  type ReactNode,
} from "react";

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

type NotificationType = "success" | "warning" | "error";

interface Toast {
  id: number;
  message: string;
  type: NotificationType;
  /** true once the slide-in transition has started */
  visible: boolean;
}

interface NotificationContextValue {
  showNotification: (message: string, type: NotificationType) => void;
}

/* ------------------------------------------------------------------ */
/*  Styles                                                             */
/* ------------------------------------------------------------------ */

const BORDER_COLORS: Record<NotificationType, string> = {
  success: "#2e7d32",
  warning: "#f9a825",
  error: "#c62828",
};

const containerStyle: React.CSSProperties = {
  position: "fixed",
  top: 24,
  right: 24,
  zIndex: 9999,
  display: "flex",
  flexDirection: "column",
  gap: 10,
  pointerEvents: "none",
};

const toastBase: React.CSSProperties = {
  pointerEvents: "auto",
  background: "#fff",
  borderRadius: 8,
  padding: "12px 36px 12px 16px",
  fontSize: 13,
  fontFamily: "var(--wb-font, inherit)",
  color: "#1d2d35",
  boxShadow: "0 4px 14px rgba(0,0,0,0.12)",
  maxWidth: 380,
  minWidth: 260,
  position: "relative",
  transition: "transform 0.3s ease, opacity 0.3s ease",
};

const dismissStyle: React.CSSProperties = {
  position: "absolute",
  top: 8,
  right: 8,
  background: "none",
  border: "none",
  cursor: "pointer",
  fontSize: 14,
  lineHeight: 1,
  color: "#637381",
  padding: 4,
};

/* ------------------------------------------------------------------ */
/*  Context                                                            */
/* ------------------------------------------------------------------ */

const NotificationContext = createContext<NotificationContextValue | null>(null);

let nextId = 0;

/* ------------------------------------------------------------------ */
/*  Provider                                                           */
/* ------------------------------------------------------------------ */

export function NotificationProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const dismiss = useCallback((id: number) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const showNotification = useCallback(
    (message: string, type: NotificationType) => {
      const id = ++nextId;
      setToasts((prev) => [...prev, { id, message, type, visible: false }]);

      // Trigger slide-in on the next frame so the initial off-screen
      // position is painted first.
      requestAnimationFrame(() => {
        setToasts((prev) =>
          prev.map((t) => (t.id === id ? { ...t, visible: true } : t)),
        );
      });

      // Auto-dismiss after 5 seconds.
      setTimeout(() => dismiss(id), 5000);
    },
    [dismiss],
  );

  const ctx: NotificationContextValue = { showNotification };

  return (
    <NotificationContext.Provider value={ctx}>
      {children}

      {/* Toast container */}
      <div style={containerStyle}>
        {toasts.map((t) => (
          <ToastItem key={t.id} toast={t} onDismiss={dismiss} />
        ))}
      </div>
    </NotificationContext.Provider>
  );
}

/* ------------------------------------------------------------------ */
/*  Single toast                                                       */
/* ------------------------------------------------------------------ */

function ToastItem({
  toast,
  onDismiss,
}: {
  toast: Toast;
  onDismiss: (id: number) => void;
}) {
  return (
    <div
      role="alert"
      style={{
        ...toastBase,
        borderLeft: `4px solid ${BORDER_COLORS[toast.type]}`,
        transform: toast.visible ? "translateX(0)" : "translateX(calc(100% + 32px))",
        opacity: toast.visible ? 1 : 0,
      }}
    >
      {toast.message}
      <button
        type="button"
        aria-label="Dismiss"
        style={dismissStyle}
        onClick={() => onDismiss(toast.id)}
      >
        &#x2715;
      </button>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Hook                                                               */
/* ------------------------------------------------------------------ */

export function useNotification() {
  const ctx = useContext(NotificationContext);
  if (!ctx) {
    throw new Error("useNotification must be used within <NotificationProvider>");
  }
  return ctx;
}
