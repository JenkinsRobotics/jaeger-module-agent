/** Jaeger Agent's own UI: talk to the running loop, watch everything it does — in today's Jaeger design.
 * Owned by this module; the shared pieces (tokens, feed shaping) are structured to come from the
 * Nexus UI library once it is extracted from JaegerAI. */
import { useEffect, useRef, useState } from "react";
import { FEED_SKIP, feedBody, feedLabel, type LoopEvent } from "./events";

type Msg = { who: "user" | "agent"; text: string };

export function App() {
  const [ready, setReady] = useState(false);
  const [model, setModel] = useState("");
  const [status, setStatus] = useState("starting…");
  const [feed, setFeed] = useState<LoopEvent[]>([]);
  const [messages, setMessages] = useState<Msg[]>([]);
  const [draft, setDraft] = useState("");
  const bottom = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const health = () => fetch("/health").then(r => r.json()).then(h => { setReady(!!h.ready); setModel(h.model ?? ""); }, () => setReady(false));
    health();
    const t = window.setInterval(health, 3000);
    const es = new EventSource("/events");
    es.onmessage = m => {
      let e: LoopEvent; try { e = JSON.parse(m.data); } catch { return; }
      if (!e.event) return;
      if (e.event === "state" || e.event === "status") setStatus(e.text ?? "");
      if (e.event === "user" && e.text) setMessages(ms => [...ms, { who: "user", text: e.text! }]);
      if (e.event === "assistant" && e.text) setMessages(ms => {
        const last = ms[ms.length - 1];
        return last?.who === "agent" && !(e as { n?: number }).n ? ms : [...ms, { who: "agent", text: e.text! }];
      });
      if (e.event === "tool" && e.text === "scene") {   // a reply the model performed: its say-lines are the words
        const said = String((e.data as { value?: unknown })?.value ?? "").split("|").map(s => s.trim()).filter(s => s.startsWith("say ")).map(s => s.slice(4)).join(" ");
        if (said) setMessages(ms => [...ms, { who: "agent", text: said }]);
      }
      if (!FEED_SKIP.has(e.event)) setFeed(f => [e, ...f].slice(0, 250));
    };
    return () => { window.clearInterval(t); es.close(); };
  }, []);
  useEffect(() => { bottom.current?.scrollIntoView(); }, [messages]);

  const send = () => {
    const text = draft.trim();
    if (!text) return;
    setDraft("");
    setMessages(ms => [...ms, { who: "user", text }]);
    void fetch("/say", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text }) });
  };

  return (
    <div id="app">
      <header>
        <span className="brand">
          <svg viewBox="0 0 64 64" aria-hidden="true"><rect width="64" height="64" rx="18" fill="#1c2034" /><rect x="14" y="20" width="13" height="24" rx="6" fill="#5fe0d0" /><rect x="37" y="20" width="13" height="24" rx="6" fill="#5fe0d0" /></svg>
          Jaeger Agent
        </span>
        <span id="status"><i id="light" data-on={ready} /> {ready ? [model, status].filter(Boolean).join(" · ") : "loading the agent…"}</span>
      </header>
      <section className="panel" aria-label="Conversation">
        <h2>Conversation</h2>
        <div id="messages" role="log">
          {messages.length === 0 && <p className="note">Say or type anything. The agent hears and speaks on this machine's own microphone and speaker.</p>}
          {messages.map((m, i) => <div key={i} className="msg" data-who={m.who}>{m.text}</div>)}
          <div ref={bottom} />
        </div>
        <form onSubmit={e => { e.preventDefault(); send(); }}>
          <textarea rows={2} value={draft} aria-label="Message the agent" placeholder="Talk to the agent"
            onChange={e => setDraft(e.target.value)}
            onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }} />
          <button type="button" onClick={() => void fetch("/interrupt", { method: "POST" })}>Stop</button>
          <button type="submit" className="send" aria-label="Send">↑</button>
        </form>
      </section>
      <section className="panel" aria-label="Agent telemetry">
        <h2>Agent telemetry</h2>
        <div className="note">{status}</div>
        <div id="feed" role="log" aria-label="Everything the agent does, as it happens">
          {feed.length === 0 && <span className="note">Waiting for the loop…</span>}
          {feed.map((e, i) => <div key={feed.length - i} className="tele" data-kind={e.event}><b>{feedLabel(e)}</b> {feedBody(e)}</div>)}
        </div>
      </section>
    </div>
  );
}
