// Model auth index tests cover auth index loading while listing models.
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { AuthProfileStore } from "../../agents/auth-profiles/types.js";
import { withEnvAsync } from "../../test-utils/env.js";
import { createModelListAuthIndex } from "./list.auth-index.js";

type PluginSnapshotResult = {
  source: "persisted" | "provided" | "derived";
  snapshot: {
    plugins: Array<{ enabled?: boolean; syntheticAuthRefs?: string[] }>;
  };
  diagnostics: [];
};

const pluginRegistryMocks = vi.hoisted(() => ({
  loadPluginRegistrySnapshotWithMetadata: vi.fn(
    (): PluginSnapshotResult => ({
      source: "persisted",
      snapshot: { plugins: [] },
      diagnostics: [],
    }),
  ),
}));

const envCandidateMocks = vi.hoisted(() => ({
  resolveProviderEnvAuthLookupMaps: vi.fn(),
}));

vi.mock("../../agents/model-auth-env-vars.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../agents/model-auth-env-vars.js")>();
  envCandidateMocks.resolveProviderEnvAuthLookupMaps.mockImplementation(
    actual.resolveProviderEnvAuthLookupMaps,
  );
  return {
    ...actual,
    resolveProviderEnvAuthLookupMaps: envCandidateMocks.resolveProviderEnvAuthLookupMaps,
  };
});

vi.mock("../../plugins/plugin-registry.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../plugins/plugin-registry.js")>();
  return {
    ...actual,
    loadPluginRegistrySnapshotWithMetadata:
      pluginRegistryMocks.loadPluginRegistrySnapshotWithMetadata,
  };
});

const emptyStore: AuthProfileStore = {
  version: 1,
  profiles: {},
};

function modelConfig(id: string) {
  return {
    id,
    name: id,
    reasoning: false,
    input: ["text" as const],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 8192,
    maxTokens: 4096,
  };
}

async function writeAnthropicVertexAdcPlugin(workspaceDir: string) {
  // A plugin that registers the Anthropic Vertex provider with the shared GCP ADC
  // marker as its env-flag evidence. Anthropic-on-Vertex (api: anthropic-messages)
  // uses the same gcp-vertex-credentials marker as Google Vertex.
  const pluginDir = path.join(workspaceDir, ".openclaw", "extensions", "anthropic-vertex-adc");
  await fs.mkdir(pluginDir, { recursive: true });
  await fs.writeFile(path.join(pluginDir, "index.ts"), "export default {}\n", "utf8");
  await fs.writeFile(
    path.join(pluginDir, "openclaw.plugin.json"),
    JSON.stringify({
      id: "anthropic-vertex-adc",
      configSchema: { type: "object" },
      setup: {
        providers: [
          {
            id: "anthropic-vertex",
            authEvidence: [
              {
                type: "env-flag",
                flagEnvVars: ["GOOGLE_VERTEX_USE_GCP_METADATA"],
                requiresAnyEnv: ["GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT"],
                requiresAllEnv: ["GOOGLE_CLOUD_LOCATION"],
                credentialMarker: "gcp-vertex-credentials",
                source: "gcp metadata adc",
              },
            ],
          },
        ],
      },
    }),
    "utf8",
  );
}

async function writeWorkspaceAuthEvidencePlugin(workspaceDir: string) {
  const pluginDir = path.join(workspaceDir, ".openclaw", "extensions", "workspace-cloud");
  await fs.mkdir(pluginDir, { recursive: true });
  await fs.writeFile(path.join(pluginDir, "index.ts"), "export default {}\n", "utf8");
  await fs.writeFile(
    path.join(pluginDir, "openclaw.plugin.json"),
    JSON.stringify({
      id: "workspace-cloud",
      configSchema: { type: "object" },
      setup: {
        providers: [
          {
            id: "workspace-cloud",
            authEvidence: [
              {
                type: "local-file-with-env",
                fileEnvVar: "WORKSPACE_CLOUD_CREDENTIALS",
                credentialMarker: "workspace-cloud-local-credentials",
                source: "workspace cloud credentials",
              },
            ],
          },
        ],
      },
    }),
    "utf8",
  );
}

describe("createModelListAuthIndex", () => {
  beforeEach(() => {
    envCandidateMocks.resolveProviderEnvAuthLookupMaps.mockClear();
    pluginRegistryMocks.loadPluginRegistrySnapshotWithMetadata.mockClear();
  });

  it("normalizes auth aliases from profiles", () => {
    const index = createModelListAuthIndex({
      cfg: {},
      authStore: {
        version: 1,
        profiles: {
          "byteplus:default": {
            type: "api_key",
            provider: "byteplus",
            key: "sk-test",
          },
        },
      },
      env: {},
    });

    expect(index.hasProviderAuth("byteplus")).toBe(true);
    expect(index.hasProviderAuth("byteplus-plan")).toBe(true);
  });

  it("records env-backed providers without resolving env candidates per row", () => {
    const index = createModelListAuthIndex({
      cfg: {},
      authStore: emptyStore,
      env: {
        MOONSHOT_API_KEY: "sk-test",
      },
    });

    expect(index.hasProviderAuth("moonshot")).toBe(true);
    expect(index.hasProviderAuth("openai")).toBe(false);
  });

  it("checks resolver-only env auth on demand", () => {
    envCandidateMocks.resolveProviderEnvAuthLookupMaps.mockReturnValueOnce({
      aliasMap: {},
      envCandidateMap: {},
      authEvidenceMap: {},
    });
    const index = createModelListAuthIndex({
      cfg: {},
      authStore: emptyStore,
      env: {
        GOOGLE_CLOUD_API_KEY: "gcp-test",
      },
    });

    expect(index.hasProviderAuth("google-vertex")).toBe(true);
  });

  it("treats a Vertex model under the shared google provider as available via google-vertex auth", () => {
    envCandidateMocks.resolveProviderEnvAuthLookupMaps.mockReturnValueOnce({
      aliasMap: {},
      envCandidateMap: {},
      authEvidenceMap: {},
    });
    const index = createModelListAuthIndex({
      cfg: {},
      authStore: emptyStore,
      env: {
        // Resolver-only env auth keyed under the dedicated google-vertex provider.
        GOOGLE_CLOUD_API_KEY: "gcp-test",
      },
    });

    // The shared "google" provider has no auth of its own...
    expect(index.hasProviderAuth("google")).toBe(false);
    expect(index.hasProviderAuth("google", "google-generative-ai")).toBe(false);
    // ...but a Vertex model registered under it resolves the google-vertex fallback.
    expect(index.hasProviderAuth("google", "google-vertex")).toBe(true);
  });

  it("treats a base-url-routed google-generative-ai row as Vertex, matching runtime", () => {
    envCandidateMocks.resolveProviderEnvAuthLookupMaps.mockReturnValueOnce({
      aliasMap: {},
      envCandidateMap: {},
      authEvidenceMap: {},
    });
    const index = createModelListAuthIndex({
      cfg: {},
      authStore: emptyStore,
      env: {
        // Resolver-only env auth keyed under the dedicated google-vertex provider.
        GOOGLE_CLOUD_API_KEY: "gcp-test",
      },
    });

    // A google-generative-ai row with no Vertex base URL is genuine AI Studio: the
    // google-vertex credential (GOOGLE_CLOUD_API_KEY) must not advertise it.
    expect(index.hasProviderAuth("google", "google-generative-ai")).toBe(false);

    // The same api routed through Vertex by an aiplatform base URL is treated as a
    // Vertex request and resolves the google-vertex fallback — exactly as runtime
    // now does (it consults the dedicated google-vertex provider for base-url-routed
    // requests too), so availability matches dispatch.
    expect(
      index.hasProviderAuth(
        "google",
        "google-generative-ai",
        "https://us-central1-aiplatform.googleapis.com",
      ),
    ).toBe(true);
    expect(
      index.hasProviderAuth(
        "google-vertex",
        "google-generative-ai",
        "https://us-central1-aiplatform.googleapis.com",
      ),
    ).toBe(true);
    // Multi-region Vertex hosts route the same way.
    expect(
      index.hasProviderAuth(
        "google",
        "google-generative-ai",
        "https://aiplatform.us.rep.googleapis.com",
      ),
    ).toBe(true);
  });

  it("does not let a Gemini key under google satisfy a Vertex model row", () => {
    envCandidateMocks.resolveProviderEnvAuthLookupMaps.mockReturnValueOnce({
      aliasMap: {},
      envCandidateMap: {},
      authEvidenceMap: {},
    });
    const index = createModelListAuthIndex({
      cfg: {},
      authStore: emptyStore,
      env: {
        // An AI Studio (Gemini) key authenticates the shared "google" provider.
        GEMINI_API_KEY: "ai-studio-key", // pragma: allowlist secret
      },
    });

    // Gemini models under "google" are available...
    expect(index.hasProviderAuth("google")).toBe(true);
    expect(index.hasProviderAuth("google", "google-generative-ai")).toBe(true);
    // ...but a Vertex model under "google" is not: the runtime resolver rejects the
    // Gemini key for the Vertex transport, so availability must not advertise it.
    expect(index.hasProviderAuth("google", "google-vertex")).toBe(false);
  });

  it("does not let a configured Vertex ADC marker satisfy a non-Vertex model row", () => {
    envCandidateMocks.resolveProviderEnvAuthLookupMaps.mockReturnValueOnce({
      aliasMap: {},
      envCandidateMap: {},
      authEvidenceMap: {},
    });
    const index = createModelListAuthIndex({
      cfg: {
        models: {
          providers: {
            "google-vertex": {
              api: "google-vertex",
              // The ADC marker is usable only by the google-vertex transport.
              apiKey: "gcp-vertex-credentials",
              baseUrl: "https://aiplatform.googleapis.com",
              models: [modelConfig("gemini-2.5-flash")],
            },
          },
        },
      },
      authStore: emptyStore,
      env: {},
    });

    // A Vertex model row resolves the marker via the model-scoped Vertex path...
    expect(index.hasProviderAuth("google-vertex", "google-vertex")).toBe(true);
    // ...but a model overriding api to a non-Vertex (Gemini) surface must not be
    // marked available: the marker isn't a usable Gemini credential at runtime.
    expect(index.hasProviderAuth("google-vertex", "google-generative-ai")).toBe(false);
  });

  it("treats a Vertex model under google as available via a configured google-vertex ADC marker", () => {
    envCandidateMocks.resolveProviderEnvAuthLookupMaps.mockReturnValueOnce({
      aliasMap: {},
      envCandidateMap: {},
      authEvidenceMap: {},
    });
    const index = createModelListAuthIndex({
      cfg: {
        models: {
          providers: {
            "google-vertex": {
              api: "google-vertex",
              // ADC marker configured on the dedicated google-vertex provider entry.
              apiKey: "gcp-vertex-credentials",
              baseUrl: "https://aiplatform.googleapis.com",
              models: [],
            },
          },
        },
      },
      authStore: emptyStore,
      env: {},
    });

    // A Vertex model registered under the shared "google" provider is available via
    // the configured ADC marker on the dedicated google-vertex provider entry, which
    // runtime resolveApiKeyForProvider now falls back to.
    expect(index.hasProviderAuth("google", "google-vertex")).toBe(true);
    // The marker still must not satisfy a non-Vertex (Gemini) row under google.
    expect(index.hasProviderAuth("google", "google-generative-ai")).toBe(false);
  });

  it("keeps the ADC marker available for a non-Gemini Vertex api (anthropic-messages)", async () => {
    const tempRoot = await fs.mkdtemp(path.join(os.tmpdir(), "openclaw-anthropic-vertex-adc-"));
    const workspaceDir = path.join(tempRoot, "workspace");
    await writeAnthropicVertexAdcPlugin(workspaceDir);

    try {
      await withEnvAsync(
        {
          GOOGLE_VERTEX_USE_GCP_METADATA: "true",
          GOOGLE_CLOUD_PROJECT: "vertex-project",
          GOOGLE_CLOUD_LOCATION: "global",
        },
        async () => {
          const index = createModelListAuthIndex({
            cfg: { plugins: { allow: ["anthropic-vertex-adc"] } },
            authStore: emptyStore,
            workspaceDir,
            env: {
              GOOGLE_VERTEX_USE_GCP_METADATA: "true",
              GOOGLE_CLOUD_PROJECT: "vertex-project",
              GOOGLE_CLOUD_LOCATION: "global",
            },
          });

          // An anthropic-messages row whose only credential is the GCP ADC marker is
          // available even without an aiplatform baseUrl: the Anthropic Vertex
          // transport exchanges the marker at runtime, so listing must mirror that.
          expect(index.hasProviderAuth("anthropic-vertex", "anthropic-messages")).toBe(true);
          // The same marker must NOT satisfy a Gemini/AI-Studio row under the provider:
          // that transport cannot use it.
          expect(index.hasProviderAuth("anthropic-vertex", "google-generative-ai")).toBe(false);
        },
      );
    } finally {
      await fs.rm(tempRoot, { recursive: true, force: true });
    }
  });

  it("does not rediscover resolver-only env auth when a command metadata snapshot is supplied", () => {
    envCandidateMocks.resolveProviderEnvAuthLookupMaps.mockReturnValueOnce({
      aliasMap: {},
      envCandidateMap: {},
      authEvidenceMap: {},
    });
    const metadataSnapshot = {
      index: { plugins: [] },
      plugins: [],
    };
    const index = createModelListAuthIndex({
      cfg: {},
      authStore: emptyStore,
      env: {
        GOOGLE_CLOUD_API_KEY: "gcp-test",
      },
      metadataSnapshot: metadataSnapshot as unknown as Parameters<
        typeof createModelListAuthIndex
      >[0]["metadataSnapshot"],
    });

    expect(index.hasProviderAuth("google-vertex")).toBe(false);
  });

  it("uses trusted workspace plugin auth evidence when workspace scope is supplied", async () => {
    const tempRoot = await fs.mkdtemp(path.join(os.tmpdir(), "openclaw-list-auth-index-"));
    const workspaceDir = path.join(tempRoot, "workspace");
    const bundledDir = path.join(tempRoot, "bundled");
    const stateDir = path.join(tempRoot, "state");
    const credentialsPath = path.join(tempRoot, "credentials.json");
    await fs.mkdir(bundledDir, { recursive: true });
    await fs.mkdir(stateDir, { recursive: true });
    await fs.writeFile(credentialsPath, "{}", "utf8");
    await writeWorkspaceAuthEvidencePlugin(workspaceDir);

    try {
      await withEnvAsync(
        {
          OPENCLAW_BUNDLED_PLUGINS_DIR: bundledDir,
          OPENCLAW_STATE_DIR: stateDir,
          WORKSPACE_CLOUD_CREDENTIALS: credentialsPath,
        },
        async () => {
          const cfg = { plugins: { allow: ["workspace-cloud"] } };
          const withoutWorkspace = createModelListAuthIndex({
            cfg,
            authStore: emptyStore,
            env: process.env,
          });
          const withWorkspace = createModelListAuthIndex({
            cfg,
            authStore: emptyStore,
            workspaceDir,
            env: process.env,
          });

          expect(withoutWorkspace.hasProviderAuth("workspace-cloud")).toBe(false);
          expect(withWorkspace.hasProviderAuth("workspace-cloud")).toBe(true);
        },
      );
    } finally {
      await fs.rm(tempRoot, { recursive: true, force: true });
    }
  });

  it("records configured provider API keys", () => {
    const index = createModelListAuthIndex({
      cfg: {
        models: {
          providers: {
            "custom-openai": {
              api: "openai-completions",
              apiKey: "sk-configured",
              baseUrl: "https://custom.example/v1",
              models: [modelConfig("local-model")],
            },
          },
        },
      },
      authStore: emptyStore,
      env: {},
    });

    expect(index.hasProviderAuth("custom-openai")).toBe(true);
  });

  it("treats OpenAI OAuth auth as usable for canonical OpenAI agent routes", () => {
    const index = createModelListAuthIndex({
      cfg: {},
      authStore: {
        version: 1,
        profiles: {
          "openai:default": {
            type: "oauth",
            provider: "openai",
            access: "access-token",
            refresh: "refresh-token",
            expires: Date.now() + 60_000,
          },
          "openai:token": {
            type: "token",
            provider: "openai",
            token: "token",
          },
        },
      },
      env: {},
    });

    expect(index.hasProviderAuth("openai")).toBe(true);
  });

  it("treats OpenAI token auth as usable for canonical OpenAI agent routes", () => {
    const index = createModelListAuthIndex({
      cfg: {},
      authStore: {
        version: 1,
        profiles: {
          "openai:token": {
            type: "token",
            provider: "openai",
            token: "token",
          },
        },
      },
      env: {},
    });

    expect(index.hasProviderAuth("openai")).toBe(true);
  });

  it("does not treat OpenAI OAuth auth as usable for custom OpenAI-compatible routes", () => {
    const index = createModelListAuthIndex({
      cfg: {
        models: {
          providers: {
            openai: {
              api: "openai-completions",
              baseUrl: "https://custom.example/v1",
              models: [modelConfig("custom-model")],
            },
          },
        },
      },
      authStore: {
        version: 1,
        profiles: {
          "openai:default": {
            type: "oauth",
            provider: "openai",
            access: "access-token",
            refresh: "refresh-token",
            expires: Date.now() + 60_000,
          },
        },
      },
      env: {},
    });

    expect(index.hasProviderAuth("openai")).toBe(false);
  });

  it("records configured local custom provider markers", () => {
    const index = createModelListAuthIndex({
      cfg: {
        models: {
          providers: {
            "local-openai": {
              api: "openai-completions",
              baseUrl: "http://127.0.0.1:8080/v1",
              models: [modelConfig("local-model")],
            },
          },
        },
      },
      authStore: emptyStore,
      env: {},
    });

    expect(index.hasProviderAuth("local-openai")).toBe(true);
  });

  it("uses injected synthetic auth refs without loading provider runtime", () => {
    const index = createModelListAuthIndex({
      cfg: {},
      authStore: emptyStore,
      env: {},
      syntheticAuthProviderRefs: ["codex"],
    });

    expect(index.hasProviderAuth("codex")).toBe(true);
  });

  it("uses an injected metadata snapshot index for synthetic auth refs", () => {
    const metadataSnapshot = {
      index: {
        plugins: [{ enabled: true, syntheticAuthRefs: ["codex"] }],
      },
      plugins: [],
    };
    pluginRegistryMocks.loadPluginRegistrySnapshotWithMetadata.mockImplementationOnce(
      ({ index }: { index?: typeof metadataSnapshot.index } = {}) => ({
        source: "provided",
        snapshot: index ?? { plugins: [] },
        diagnostics: [],
      }),
    );

    const index = createModelListAuthIndex({
      cfg: {},
      authStore: emptyStore,
      env: {},
      metadataSnapshot: metadataSnapshot as unknown as Parameters<
        typeof createModelListAuthIndex
      >[0]["metadataSnapshot"],
    });

    expect(index.hasProviderAuth("codex")).toBe(true);
    expect(pluginRegistryMocks.loadPluginRegistrySnapshotWithMetadata).toHaveBeenCalledWith(
      expect.objectContaining({ index: metadataSnapshot.index }),
    );
  });

  it("ignores synthetic auth refs from injected derived metadata snapshots", () => {
    const metadataSnapshot = {
      index: {
        plugins: [{ enabled: true, syntheticAuthRefs: ["codex"] }],
      },
      plugins: [],
      registryDiagnostics: [
        {
          level: "info",
          code: "persisted-registry-missing",
          message: "missing",
        },
      ],
    };

    const index = createModelListAuthIndex({
      cfg: {},
      authStore: emptyStore,
      env: {},
      metadataSnapshot: metadataSnapshot as unknown as Parameters<
        typeof createModelListAuthIndex
      >[0]["metadataSnapshot"],
    });

    expect(index.hasProviderAuth("codex")).toBe(false);
    expect(pluginRegistryMocks.loadPluginRegistrySnapshotWithMetadata).not.toHaveBeenCalledWith(
      expect.objectContaining({ index: metadataSnapshot.index }),
    );
  });

  it("keeps synthetic auth refs exact instead of applying auth-choice aliases", () => {
    const index = createModelListAuthIndex({
      cfg: {},
      authStore: emptyStore,
      env: {},
      syntheticAuthProviderRefs: ["claude-cli"],
    });

    expect(index.hasProviderAuth("claude-cli")).toBe(true);
    expect(index.hasProviderAuth("anthropic")).toBe(false);
  });

  it("ignores derived synthetic auth snapshots", () => {
    pluginRegistryMocks.loadPluginRegistrySnapshotWithMetadata.mockReturnValueOnce({
      source: "derived",
      snapshot: {
        plugins: [{ enabled: true, syntheticAuthRefs: ["codex"] }],
      },
      diagnostics: [],
    });
    const index = createModelListAuthIndex({
      cfg: {},
      authStore: emptyStore,
      env: {},
    });

    expect(index.hasProviderAuth("codex")).toBe(false);
  });

  it("ignores disabled synthetic auth snapshot entries", () => {
    pluginRegistryMocks.loadPluginRegistrySnapshotWithMetadata.mockReturnValueOnce({
      source: "persisted",
      snapshot: {
        plugins: [{ enabled: false, syntheticAuthRefs: ["codex"] }],
      },
      diagnostics: [],
    });
    const index = createModelListAuthIndex({
      cfg: {},
      authStore: emptyStore,
      env: {},
    });

    expect(index.hasProviderAuth("codex")).toBe(false);
  });
});
