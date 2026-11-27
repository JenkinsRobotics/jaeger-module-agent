/** The loop's events, read once and shared: the same stream the hosts read, shaped for the feed and the chat. */
export type LoopEvent = { event?: string; text?: string; data?: unknown; [k: string]: unknown };

export const FEED_SKIP = new Set(["audio", "voiced", "done"]);   // sound payloads and timing ticks: noise in a reading feed

export const feedLabel = (e: LoopEvent): string =>
  ({ user: "heard", assistant: "says", tool: "tool", feeling: "feeling" } as Record<string, string>)[e.event ?? ""] ?? e.event ?? "";

export const feedBody = (e: LoopEvent): string => {
  const brief = (v: unknown): string => {
    if (v == null || v === "") return "";
    const t = typeof v === "string" ? v : JSON.stringify(v);
    return t.length > 220 ? t.slice(0, 220) + "…" : t;
  };
  if (e.event === "tool") return [e.text, brief((e.data as { value?: unknown })?.value ?? e.data)].filter(Boolean).join(" · ");
  return brief(e.text) || brief(e.data);
};
