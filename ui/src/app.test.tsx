import { describe, expect, it } from "vitest";
import { FEED_SKIP, feedBody, feedLabel } from "./events";

describe("the telemetry feed shaping", () => {
  it("labels the kinds a reader knows", () => {
    expect(feedLabel({ event: "user", text: "hi" })).toBe("heard");
    expect(feedLabel({ event: "assistant", text: "hello" })).toBe("says");
    expect(feedLabel({ event: "tool", text: "avatar emotion" })).toBe("tool");
    expect(feedLabel({ event: "state", text: "listening" })).toBe("state");
  });
  it("shows a tool's name with its value, and truncates long payloads", () => {
    expect(feedBody({ event: "tool", text: "avatar emotion", data: { value: "joy" } })).toBe("avatar emotion · joy");
    expect(feedBody({ event: "assistant", text: "x".repeat(300) }).endsWith("…")).toBe(true);
  });
  it("skips sound payloads and timing ticks", () => {
    for (const k of ["audio", "voiced", "done"]) expect(FEED_SKIP.has(k)).toBe(true);
    expect(FEED_SKIP.has("tool")).toBe(false);
  });
});
