import { useRef, useEffect, useState, type CSSProperties } from "react";
import ReactMarkdown from "react-markdown";
import { useChat } from "../hooks/useChat";
import type { ChatMessageData } from "../types/chat";

interface ContextInfo {
  status: string;
  profiled_tables: number;
  fully_profiled_tables: number;
  catalog_context_chars: number;
}

function useContextInfo() {
  const [info, setInfo] = useState<ContextInfo | null>(null);
  useEffect(() => {
    fetch("/api/chat/context-info")
      .then((r) => r.json())
      .then(setInfo)
      .catch(() => {});
  }, []);
  return info;
}

function MessageBubble({ msg }: { msg: ChatMessageData }) {
  const isUser = msg.role === "user";
  const [copied, setCopied] = useState(false);

  const bubbleStyle: CSSProperties = {
    maxWidth: "75%",
    alignSelf: isUser ? "flex-end" : "flex-start",
    background: isUser ? "var(--wb-primary, #1a5c5e)" : "#f4f6f8",
    color: isUser ? "#fff" : "#222",
    padding: "12px 16px",
    borderRadius: isUser ? "16px 16px 4px 16px" : "16px 16px 16px 4px",
    fontSize: 14,
    lineHeight: 1.6,
    whiteSpace: "pre-wrap",
    wordBreak: "break-word",
  };

  const copySQL = () => {
    if (msg.sql) {
      navigator.clipboard.writeText(msg.sql);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    }
  };

  return (
    <div style={bubbleStyle}>
      <div className="chat-md" style={{ fontSize: 14, lineHeight: 1.6 }}>
        <ReactMarkdown
          components={{
            h1: ({ children }) => <div style={{ fontSize: 16, fontWeight: 700, margin: "8px 0 4px" }}>{children}</div>,
            h2: ({ children }) => <div style={{ fontSize: 15, fontWeight: 700, margin: "8px 0 4px" }}>{children}</div>,
            h3: ({ children }) => <div style={{ fontSize: 14, fontWeight: 700, margin: "6px 0 2px" }}>{children}</div>,
            p: ({ children }) => <div style={{ margin: "4px 0" }}>{children}</div>,
            ul: ({ children }) => <ul style={{ margin: "4px 0", paddingLeft: 20 }}>{children}</ul>,
            ol: ({ children }) => <ol style={{ margin: "4px 0", paddingLeft: 20 }}>{children}</ol>,
            li: ({ children }) => <li style={{ margin: "2px 0" }}>{children}</li>,
            code: ({ className, children }) => {
              const isBlock = className?.includes("language-");
              if (isBlock) {
                return <pre style={{ background: isUser ? "rgba(0,0,0,0.15)" : "#e8ecef", padding: 8, borderRadius: 4, overflowX: "auto", fontSize: 12, margin: "4px 0" }}><code>{children}</code></pre>;
              }
              return <code style={{ background: isUser ? "rgba(255,255,255,0.15)" : "#e8ecef", padding: "1px 4px", borderRadius: 3, fontSize: 12 }}>{children}</code>;
            },
            table: ({ children }) => <table style={{ borderCollapse: "collapse", fontSize: 12, margin: "6px 0", width: "100%" }}>{children}</table>,
            th: ({ children }) => <th style={{ border: "1px solid " + (isUser ? "rgba(255,255,255,0.3)" : "#ccc"), padding: "4px 8px", textAlign: "left", fontWeight: 600 }}>{children}</th>,
            td: ({ children }) => <td style={{ border: "1px solid " + (isUser ? "rgba(255,255,255,0.2)" : "#ddd"), padding: "4px 8px" }}>{children}</td>,
            strong: ({ children }) => <strong>{children}</strong>,
          }}
        >
          {msg.content}
        </ReactMarkdown>
      </div>
      {msg.sql && (
        <div style={{ marginTop: 8, background: isUser ? "rgba(0,0,0,0.15)" : "#e8ecef", borderRadius: 6, padding: "8px 10px", fontSize: 12, fontFamily: "monospace" }}>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
            <span style={{ fontWeight: 600, fontSize: 11, opacity: 0.7 }}>SQL</span>
            <button onClick={copySQL} style={{ background: "none", border: "none", cursor: "pointer", fontSize: 11, color: isUser ? "#ddd" : "#666" }}>
              {copied ? "Copied!" : "Copy"}
            </button>
          </div>
          <pre style={{ margin: 0, whiteSpace: "pre-wrap", wordBreak: "break-all" }}>{msg.sql}</pre>
        </div>
      )}
    </div>
  );
}

export default function ChatPage() {
  const { messages, loading, error, mode, sendMessage, clearChat, toggleMode } = useChat();
  const [input, setInput] = useState("");
  const [detailLevel, setDetailLevel] = useState<"summary" | "full">("summary");
  const [preloading, setPreloading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const ctxInfo = useContextInfo();

  const toggleDetail = async () => {
    if (detailLevel === "full") {
      setDetailLevel("summary");
      return;
    }
    setPreloading(true);
    try {
      await fetch("/api/chat/preload", { method: "POST" });
    } catch {}
    setPreloading(false);
    setDetailLevel("full");
  };

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = () => {
    if (!input.trim() || loading) return;
    sendMessage(input.trim(), null, detailLevel);
    setInput("");
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const isFull = detailLevel === "full";

  return (
    <div style={{ padding: "32px 40px", height: "100vh", display: "flex", flexDirection: "column" }}>
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
        <div>
          <h1 style={{ margin: 0, fontSize: 24, fontWeight: 700, color: "var(--wb-text)" }}>Data AMA Agent</h1>
          <p style={{ color: "var(--wb-muted)", margin: "4px 0 0", fontSize: 14 }}>
            Ask me anything about your datasets, tables, and metadata
          </p>
        </div>
        <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
          {/* Mode segmented toggle */}
          <div style={{
            display: "flex",
            border: "1px solid var(--wb-border)",
            borderRadius: 8,
            overflow: "hidden",
          }}>
            <button
              onClick={() => { if (mode === "agent") toggleMode(); }}
              style={{
                padding: "7px 16px",
                fontSize: 12,
                fontWeight: 600,
                border: "none",
                cursor: "pointer",
                background: mode === "metadata" ? "var(--wb-primary, #1a5c5e)" : "var(--wb-surface)",
                color: mode === "metadata" ? "#fff" : "var(--wb-muted)",
                borderRight: "1px solid var(--wb-border)",
              }}
              title="Ask questions about table structure, columns, and metadata"
            >
              Q&A
            </button>
            <button
              onClick={() => { if (mode === "metadata") toggleMode(); }}
              style={{
                padding: "7px 16px",
                fontSize: 12,
                fontWeight: 600,
                border: "none",
                cursor: "pointer",
                background: mode === "agent" ? "#e65100" : "var(--wb-surface)",
                color: mode === "agent" ? "#fff" : "var(--wb-muted)",
              }}
              title="Generate and execute SQL queries against BigQuery"
            >
              Agent
            </button>
          </div>
          {mode === "agent" && (
            <span style={{ fontSize: 11, color: "#e65100", fontWeight: 600 }}>
              Can execute SQL
            </span>
          )}
          <button
            onClick={clearChat}
            style={{
              background: "var(--wb-surface)",
              border: "1px solid var(--wb-border)",
              borderRadius: 6,
              padding: "6px 14px",
              fontSize: 13,
              cursor: "pointer",
            }}
          >
            Clear
          </button>
        </div>
      </div>

      {/* Context status bar */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 12,
          padding: "8px 14px",
          background: isFull ? "#e8f5e9" : "var(--wb-surface)",
          border: `1px solid ${isFull ? "#a5d6a7" : "var(--wb-border)"}`,
          borderRadius: 8,
          marginBottom: 12,
          fontSize: 13,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{
            width: 8, height: 8, borderRadius: "50%",
            background: ctxInfo?.status === "ready" ? (isFull ? "#2e7d32" : "#f9a825") : "#ccc",
            display: "inline-block",
          }} />
          <span style={{ color: "var(--wb-text)" }}>
            {ctxInfo?.status === "ready"
              ? `${ctxInfo.profiled_tables} table${ctxInfo.profiled_tables !== 1 ? "s" : ""} profiled`
              : ctxInfo?.status === "unconfigured"
                ? "No project configured"
                : "No profiles yet — profile tables first"}
            {ctxInfo?.status === "ready" && (
              <span style={{ color: "var(--wb-muted)", marginLeft: 6 }}>
                {isFull ? "Full detail loaded" : "Summary context"}
              </span>
            )}
          </span>
        </div>
        {ctxInfo?.status === "ready" && (
          <button
            onClick={toggleDetail}
            disabled={preloading}
            style={{
              background: preloading ? "#888" : isFull ? "#2e7d32" : "var(--wb-primary, #1a5c5e)",
              color: "#fff",
              border: "none",
              borderRadius: 6,
              padding: "5px 14px",
              fontSize: 12,
              fontWeight: 600,
              cursor: preloading ? "wait" : "pointer",
              whiteSpace: "nowrap",
              opacity: preloading ? 0.7 : 1,
            }}
          >
            {preloading ? "Loading profiles..." : isFull ? "Using full details" : "Load full details"}
          </button>
        )}
      </div>

      {/* Messages */}
      <div
        style={{
          flex: 1,
          overflow: "auto",
          display: "flex",
          flexDirection: "column",
          gap: 12,
          padding: "16px 0",
          minHeight: 0,
        }}
      >
        {messages.length === 0 && (
          <div style={{ color: "#999", textAlign: "center", marginTop: 60, fontSize: 14, lineHeight: 2 }}>
            {isFull
              ? "Full profile details loaded — ask about specific columns, stats, or joins"
              : "Ask about your datasets and tables — click \"Load full details\" for deeper answers"}
            <div style={{ display: "flex", gap: 24, justifyContent: "center", marginTop: 20, fontSize: 13 }}>
              <div style={{ textAlign: "center", maxWidth: 200 }}>
                <div style={{ fontWeight: 700, color: "var(--wb-primary)", marginBottom: 4 }}>Q&A Mode</div>
                <div style={{ color: "var(--wb-muted)", fontSize: 12, lineHeight: 1.5 }}>
                  What tables have diagnosis data?<br />
                  Explain the SUBJID column<br />
                  What joins exist between tables?
                </div>
              </div>
              <div style={{ textAlign: "center", maxWidth: 200 }}>
                <div style={{ fontWeight: 700, color: "#e65100", marginBottom: 4 }}>Agent Mode</div>
                <div style={{ color: "var(--wb-muted)", fontSize: 12, lineHeight: 1.5 }}>
                  Count patients with diabetes<br />
                  Show top 10 diagnosis codes<br />
                  Find subjects with PHQ-9 &gt; 10
                </div>
              </div>
            </div>
          </div>
        )}
        {messages.map((m, i) => (
          <MessageBubble key={i} msg={m} />
        ))}
        {loading && (
          <div style={{ alignSelf: "flex-start", color: "#999", fontSize: 13, padding: "4px 10px" }}>
            Thinking...
          </div>
        )}
        {error && !loading && (
          <div style={{ color: "#c00", fontSize: 12, padding: "4px 10px" }}>{error}</div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div
        style={{
          borderTop: "1px solid var(--wb-border, #dde)",
          padding: "12px 0",
          display: "flex",
          gap: 8,
          alignItems: "flex-end",
        }}
      >
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={mode === "agent" ? "Ask a question or request a SQL query..." : "Ask about your data..."}
          rows={2}
          style={{
            flex: 1,
            resize: "none",
            border: "1px solid var(--wb-border, #dde)",
            borderRadius: 8,
            padding: "10px 14px",
            fontSize: 14,
            fontFamily: "inherit",
            outline: "none",
          }}
          disabled={loading}
        />
        <button
          onClick={handleSend}
          disabled={loading || !input.trim()}
          style={{
            background: "var(--wb-primary, #1a5c5e)",
            color: "#fff",
            border: "none",
            borderRadius: 8,
            padding: "10px 20px",
            fontSize: 14,
            fontWeight: 600,
            cursor: loading || !input.trim() ? "not-allowed" : "pointer",
            opacity: loading || !input.trim() ? 0.5 : 1,
          }}
        >
          Send
        </button>
      </div>
    </div>
  );
}
