import { useState, useEffect, useRef, useCallback } from "react";

const API_BASE = import.meta.env.VITE_API_URL || "";

function Sparkline({ data, color, height = 40 }) {
  if (!data || data.length < 2) return <div style={{ height }} />;
  const max = Math.max(...data, 1);
  const pts = data.map((v, i) => {
    const x = (i / (data.length - 1)) * 100;
    const y = height - (v / max) * (height - 4) - 2;
    return `${x},${y}`;
  }).join(" ");
  return (
    <svg width="100%" height={height} viewBox={`0 0 100 ${height}`} preserveAspectRatio="none">
      <polyline points={pts} fill="none" stroke={color} strokeWidth="1.5" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

function Gauge({ pct, color, label, value }) {
  const r = 36, circ = 2 * Math.PI * r, dash = (pct / 100) * circ;
  return (
    <div className="gauge-wrap">
      <svg width="88" height="88" viewBox="0 0 88 88">
        <circle cx="44" cy="44" r={r} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="6" />
        <circle cx="44" cy="44" r={r} fill="none" stroke={color} strokeWidth="6"
          strokeDasharray={`${dash} ${circ}`} strokeLinecap="round"
          transform="rotate(-90 44 44)" style={{ transition: "stroke-dasharray 0.4s ease" }} />
        <text x="44" y="48" textAnchor="middle" fill="white" fontSize="11" fontWeight="700">{pct}%</text>
      </svg>
      <div className="gauge-label">{label}</div>
      <div className="gauge-value">{value}</div>
    </div>
  );
}

function StatCard({ title, value, unit, spark, sparkColor, accent }) {
  return (
    <div className="stat-card" style={{ "--accent": accent }}>
      <div className="stat-title">{title}</div>
      <div className="stat-value">{value}{unit && <span className="stat-unit"> {unit}</span>}</div>
      {spark && <Sparkline data={spark} color={accent} />}
    </div>
  );
}

function ChatMessage({ role, text, streaming }) {
  return (
    <div className={`msg msg-${role}`}>
      <div className="msg-tag">{role === "user" ? "YOU" : "MODEL"}</div>
      <div className="msg-text">{text}{streaming && <span className="cursor">▌</span>}</div>
    </div>
  );
}

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [stats, setStats] = useState(null);
  const [history, setHistory] = useState({ tps: [], queue: [], batch: [], mem: [] });
  const chatEndRef = useRef(null);

  useEffect(() => {
    const poll = async () => {
      try {
        const res = await fetch(`${API_BASE}/stats`);
        if (!res.ok) return;
        const s = await res.json();
        setStats(s);
        setHistory(h => ({
          tps:   [...h.tps.slice(-40),   s.tokens_per_second],
          queue: [...h.queue.slice(-40),  s.queue_depth],
          batch: [...h.batch.slice(-40),  s.batch_size],
          mem:   [...h.mem.slice(-40),    s.gpu_memory_pct],
        }));
      } catch (_) {}
    };
    poll();
    const id = setInterval(poll, 1000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => { chatEndRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);

  const submit = useCallback(async () => {
    const prompt = input.trim();
    if (!prompt || streaming) return;
    setInput(""); setStreaming(true);
    setMessages(m => [...m, { role: "user", text: prompt }, { role: "model", text: "", streaming: true }]);
    try {
      const res = await fetch(`${API_BASE}/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, max_new_tokens: 300, temperature: 0.8, top_p: 0.9 }),
      });
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop();
        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const payload = JSON.parse(line.slice(6));
          if (payload.type === "token") {
            setMessages(m => {
              const u = [...m], last = u[u.length - 1];
              u[u.length - 1] = { ...last, text: last.text + payload.text };
              return u;
            });
          } else if (payload.type === "done") {
            setMessages(m => {
              const u = [...m];
              u[u.length - 1] = { ...u[u.length - 1], streaming: false };
              return u;
            });
          }
        }
      }
    } catch {
      setMessages(m => {
        const u = [...m];
        u[u.length - 1] = { role: "model", text: "Connection error — is the backend running?", streaming: false };
        return u;
      });
    } finally { setStreaming(false); }
  }, [input, streaming]);

  const handleKey = e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(); } };
  const fmtMem = mb => mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${Math.round(mb)} MB`;

  return (
    <div className="root">
      <header className="header">
        <div className="header-logo">
          <span className="logo-icon">⚡</span>
          <span className="logo-text">mini<b>vLLM</b></span>
        </div>
        <div className="header-badge">
          <span className={`badge-dot ${stats ? "online" : "offline"}`} />
          {stats ? "Engine Online" : "Connecting…"}
        </div>
      </header>
      <main className="main-layout">
        <section className="chat-panel">
          <div className="chat-messages">
            {messages.length === 0 && (
              <div className="chat-empty">
                <div className="empty-icon">🧠</div>
                <div className="empty-title">Custom Inference Engine</div>
                <div className="empty-sub">GPT-2 running a hand-written token loop with continuous batching.<br />Type a prompt to begin.</div>
              </div>
            )}
            {messages.map((m, i) => <ChatMessage key={i} role={m.role} text={m.text} streaming={m.streaming} />)}
            <div ref={chatEndRef} />
          </div>
          <div className="chat-input-row">
            <textarea className="chat-input" rows={2} value={input}
              onChange={e => setInput(e.target.value)} onKeyDown={handleKey}
              placeholder="Shift+Enter for newline · Enter to send" disabled={streaming} />
            <button className={`send-btn ${streaming ? "sending" : ""}`}
              onClick={submit} disabled={streaming || !input.trim()}>
              {streaming ? "⏳" : "▶"}
            </button>
          </div>
        </section>
        <aside className="dashboard">
          <div className="dash-title"><span>◈</span> Engine Internals</div>
          {stats && (
            <div className="gauge-row">
              <Gauge pct={stats.gpu_memory_pct} color="#4ade80" label="VRAM"
                value={`${fmtMem(stats.gpu_memory_used_mb)} / ${fmtMem(stats.gpu_memory_total_mb)}`} />
            </div>
          )}
          <div className="stat-grid">
            <StatCard title="Tokens / sec" value={stats?.tokens_per_second ?? "—"} spark={history.tps} sparkColor="#60a5fa" accent="#60a5fa" />
            <StatCard title="Queue depth" value={stats?.queue_depth ?? "—"} unit="req" spark={history.queue} sparkColor="#f472b6" accent="#f472b6" />
            <StatCard title="Active batch" value={stats?.batch_size ?? "—"} spark={history.batch} sparkColor="#fb923c" accent="#fb923c" />
            <StatCard title="Total tokens" value={stats?.total_tokens_generated?.toLocaleString() ?? "—"} accent="#a78bfa" />
          </div>
          <div className="dash-explainer">
            <div className="explainer-title">Continuous Batching</div>
            <div className="explainer-body">
              {Array.from({ length: 8 }).map((_, i) => {
                const active = i < (stats?.batch_size ?? 0);
                return <div key={i} className={`slot ${active ? "slot-active" : "slot-idle"}`}>{active ? "▶" : "·"}</div>;
              })}
            </div>
            <p className="explainer-desc">
              Each slot = one sequence in the batch. When a slot hits <code>{"<EOS>"}</code>, it immediately swaps in the next queued prompt — zero stalls.
            </p>
          </div>
          {stats && (
            <div className="uptime">
              ⏱ Uptime {Math.floor(stats.uptime_seconds / 60)}m {Math.round(stats.uptime_seconds % 60)}s
            </div>
          )}
        </aside>
      </main>
    </div>
  );
}