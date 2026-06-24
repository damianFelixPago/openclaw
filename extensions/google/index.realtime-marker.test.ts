// The lazily-registered Google realtime voice provider in index.ts resolves the
// shared models.providers.google key independently of realtime-voice-provider.ts,
// so it must apply the same Vertex ADC marker filter: the non-secret marker is a
// Vertex transport signal, not a Gemini key, and must never be sent to the
// Generative Language realtime API as x-goog-api-key.
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
  createLazyGoogleRealtimeVoiceProvider,
  resolveGoogleRealtimeProviderConfig,
} from "./index.js";

const ENV_KEYS = ["GEMINI_API_KEY", "GOOGLE_API_KEY"] as const;
let envSnapshot: Partial<Record<(typeof ENV_KEYS)[number], string>>;

beforeEach(() => {
  envSnapshot = Object.fromEntries(ENV_KEYS.map((key) => [key, process.env[key]]));
  for (const key of ENV_KEYS) {
    delete process.env[key];
  }
});

afterEach(() => {
  for (const key of ENV_KEYS) {
    const value = envSnapshot[key];
    if (value === undefined) {
      delete process.env[key];
    } else {
      process.env[key] = value;
    }
  }
});

describe("index lazy Google realtime provider Vertex marker handling", () => {
  it("drops the Vertex ADC marker from the cfg model-provider key fallback", () => {
    const resolved = resolveGoogleRealtimeProviderConfig(
      { providers: { google: { model: "gemini-live-2.5-flash-preview" } } },
      { models: { providers: { google: { apiKey: "gcp-vertex-credentials" } } } },
    );
    expect(resolved.apiKey).toBeUndefined();
  });

  it("keeps a real Gemini cfg key in the fallback", () => {
    const resolved = resolveGoogleRealtimeProviderConfig(
      { providers: { google: { model: "gemini-live-2.5-flash-preview" } } },
      { models: { providers: { google: { apiKey: "gemini-key" } } } },
    );
    expect(resolved.apiKey).toBe("gemini-key");
  });

  it("does not report configured when only the Vertex ADC marker is present", () => {
    const provider = createLazyGoogleRealtimeVoiceProvider();
    expect(
      provider.isConfigured?.({
        cfg: { models: { providers: { google: { apiKey: "gcp-vertex-credentials" } } } } as never,
        providerConfig: {},
      }),
    ).toBe(false);
  });

  it("reports configured for a real Gemini cfg key", () => {
    const provider = createLazyGoogleRealtimeVoiceProvider();
    expect(
      provider.isConfigured?.({
        cfg: { models: { providers: { google: { apiKey: "gemini-key" } } } } as never,
        providerConfig: {},
      }),
    ).toBe(true);
  });
});
