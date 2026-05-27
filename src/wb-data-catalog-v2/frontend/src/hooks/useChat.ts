import { useState, useCallback, useRef } from "react";
import type { ChatMessageData } from "../types/chat";

export type ChatMode = "metadata" | "agent";

export function useChat() {
  const [messages, setMessages] = useState<ChatMessageData[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [mode, setMode] = useState<ChatMode>("metadata");
  const [error, setError] = useState<string | null>(null);
  const [streamingStatus, setStreamingStatus] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const sendMessage = useCallback(
    async (text: string, fqTable?: string | null, detailLevel: "summary" | "full" = "summary") => {
      if (!text.trim()) return;
      setError(null);
      setStreamingStatus(null);

      const userMsg: ChatMessageData = {
        role: "user",
        content: text,
        timestamp: new Date().toISOString(),
        mode,
      };

      const placeholderMsg: ChatMessageData = {
        role: "assistant",
        content: "",
        timestamp: new Date().toISOString(),
        mode,
      };

      setMessages((prev) => [...prev, userMsg, placeholderMsg]);
      setLoading(true);

      const controller = new AbortController();
      abortRef.current = controller;

      try {
        const body: Record<string, unknown> = {
          message: text,
          mode,
          detail_level: detailLevel,
        };
        if (fqTable) body.fq_table = fqTable;
        if (sessionId) body.session_id = sessionId;

        const res = await fetch("/api/chat/stream", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
          signal: controller.signal,
        });

        if (!res.ok) {
          const errBody = await res.json().catch(() => ({ detail: res.statusText }));
          throw new Error(errBody.detail || `Chat failed: ${res.status}`);
        }

        const reader = res.body!.getReader();
        const decoder = new TextDecoder();
        let fullText = "";
        let sql: string | undefined;
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const parts = buffer.split("\n\n");
          buffer = parts.pop()!;

          for (const raw of parts) {
            const line = raw.trim();
            if (!line.startsWith("data: ")) continue;

            let data: any;
            try {
              data = JSON.parse(line.slice(6));
            } catch {
              continue;
            }

            if (data.type === "session") {
              setSessionId(data.session_id);
            } else if (data.type === "status") {
              setStreamingStatus(data.text);
            } else if (data.type === "chunk") {
              fullText += data.text;
              const snapshot = fullText;
              setMessages((prev) => {
                const updated = [...prev];
                updated[updated.length - 1] = {
                  ...updated[updated.length - 1],
                  content: snapshot,
                };
                return updated;
              });
            } else if (data.type === "done") {
              sql = data.sql || undefined;
              setStreamingStatus(null);
            } else if (data.type === "error") {
              throw new Error(data.text || "Stream error");
            }
          }
        }

        if (sql) {
          setMessages((prev) => {
            const updated = [...prev];
            updated[updated.length - 1] = {
              ...updated[updated.length - 1],
              content: fullText,
              sql,
            };
            return updated;
          });
        }
      } catch (e: any) {
        if (e.name === "AbortError") return;
        setError(e.message || "Chat request failed");
        setStreamingStatus(null);
        setMessages((prev) => {
          const updated = [...prev];
          updated[updated.length - 1] = {
            ...updated[updated.length - 1],
            content: `Error: ${e.message || "Request failed"}`,
          };
          return updated;
        });
      } finally {
        setLoading(false);
        setStreamingStatus(null);
        abortRef.current = null;
      }
    },
    [mode, sessionId],
  );

  const stopStreaming = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const clearChat = useCallback(async () => {
    abortRef.current?.abort();
    if (sessionId) {
      await fetch("/api/chat/clear", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId }),
      }).catch(() => {});
    }
    setMessages([]);
    setSessionId(null);
    setError(null);
    setStreamingStatus(null);
  }, [sessionId]);

  const toggleMode = useCallback(() => {
    setMode((m) => (m === "metadata" ? "agent" : "metadata"));
  }, []);

  return {
    messages, loading, error, mode, sessionId,
    streamingStatus, sendMessage, stopStreaming,
    clearChat, toggleMode, setMode,
  };
}
