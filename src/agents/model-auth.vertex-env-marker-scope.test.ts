// Proves the Vertex ADC marker resolved from env auth-evidence (for example the
// GCP metadata-server opt-in) is scoped to the selected model: it resolves for a
// Vertex model but is never accepted as auth for a non-Vertex model selected
// under the same provider.
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import type { OpenClawConfig } from "../config/types.openclaw.js";
import { withEnvAsync } from "../test-utils/env.js";
import type { AuthProfileStore } from "./auth-profiles.js";
import { resolveEnvApiKey } from "./model-auth-env.js";
import { hasAvailableAuthForProvider, resolveApiKeyForProvider } from "./model-auth.js";

const VERTEX_ADC_MARKER = "gcp-vertex-credentials";
const PROVIDER = "vertex-metadata-cloud";

async function writeVertexEnvFlagPlugin(
  workspaceDir: string,
  options?: { pluginId?: string; providerId?: string; envVar?: string },
) {
  // Trusted workspace plugin that declares both an explicit env candidate (which
  // otherwise short-circuits resolution) and the env-flag ADC evidence, mirroring
  // the bundled google-vertex setup.
  const pluginId = options?.pluginId ?? "vertex-meta";
  const providerId = options?.providerId ?? PROVIDER;
  const envVar = options?.envVar ?? "VERTEX_META_CLOUD_API_KEY";
  const pluginDir = path.join(workspaceDir, ".openclaw", "extensions", pluginId);
  await fs.mkdir(pluginDir, { recursive: true });
  await fs.writeFile(path.join(pluginDir, "index.ts"), "export default {}\n", "utf8");
  await fs.writeFile(
    path.join(pluginDir, "openclaw.plugin.json"),
    JSON.stringify({
      id: pluginId,
      configSchema: { type: "object" },
      setup: {
        providers: [
          {
            id: providerId,
            envVars: [envVar],
            authEvidence: [
              {
                type: "env-flag",
                flagEnvVars: ["GOOGLE_VERTEX_USE_GCP_METADATA"],
                requiresAnyEnv: ["GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT"],
                requiresAllEnv: ["GOOGLE_CLOUD_LOCATION"],
                credentialMarker: VERTEX_ADC_MARKER,
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

async function writeGoogleVertexMixedPlugin(workspaceDir: string) {
  // A plugin that registers BOTH the generic "google" provider (with a Gemini env
  // key candidate) and the dedicated "google-vertex" provider (with the env-flag
  // ADC evidence), so a Vertex model under "google" can be shadowed by a Gemini
  // env key unless the google-vertex auth provider is consulted first.
  const pluginDir = path.join(workspaceDir, ".openclaw", "extensions", "google-mixed");
  await fs.mkdir(pluginDir, { recursive: true });
  await fs.writeFile(path.join(pluginDir, "index.ts"), "export default {}\n", "utf8");
  await fs.writeFile(
    path.join(pluginDir, "openclaw.plugin.json"),
    JSON.stringify({
      id: "google-mixed",
      configSchema: { type: "object" },
      setup: {
        providers: [
          {
            id: "google",
            envVars: ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
          },
          {
            id: "google-vertex",
            envVars: ["GOOGLE_VERTEX_META_API_KEY"],
            authEvidence: [
              {
                type: "env-flag",
                flagEnvVars: ["GOOGLE_VERTEX_USE_GCP_METADATA"],
                requiresAnyEnv: ["GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT"],
                requiresAllEnv: ["GOOGLE_CLOUD_LOCATION"],
                credentialMarker: VERTEX_ADC_MARKER,
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

async function writeCustomVertexKeyPlugin(workspaceDir: string) {
  // A dedicated/custom Vertex provider that declares its OWN env API key (not the
  // ADC marker). Vertex express mode accepts literal API keys, so a request for a
  // google-vertex model under this provider id must keep the provider-specific key.
  const pluginDir = path.join(workspaceDir, ".openclaw", "extensions", "vertex-express");
  await fs.mkdir(pluginDir, { recursive: true });
  await fs.writeFile(path.join(pluginDir, "index.ts"), "export default {}\n", "utf8");
  await fs.writeFile(
    path.join(pluginDir, "openclaw.plugin.json"),
    JSON.stringify({
      id: "vertex-express",
      configSchema: { type: "object" },
      setup: {
        providers: [{ id: "vertex-express", envVars: ["VERTEX_EXPRESS_API_KEY"] }],
      },
    }),
    "utf8",
  );
}

async function writeCustomVertexWithGlobalAdcPlugin(workspaceDir: string) {
  // A custom/dedicated Vertex provider with its OWN env API key, registered
  // alongside the dedicated "google-vertex" provider that also carries env-flag
  // ADC evidence. When both the provider-specific key and the global google-vertex
  // ADC credential are present, a Vertex request under the custom provider must
  // keep the provider-specific key rather than be switched to the global ADC.
  const pluginDir = path.join(workspaceDir, ".openclaw", "extensions", "vertex-express-mixed");
  await fs.mkdir(pluginDir, { recursive: true });
  await fs.writeFile(path.join(pluginDir, "index.ts"), "export default {}\n", "utf8");
  await fs.writeFile(
    path.join(pluginDir, "openclaw.plugin.json"),
    JSON.stringify({
      id: "vertex-express-mixed",
      configSchema: { type: "object" },
      setup: {
        providers: [
          { id: "vertex-express", envVars: ["VERTEX_EXPRESS_API_KEY"] },
          {
            id: "google-vertex",
            envVars: ["GOOGLE_VERTEX_META_API_KEY"],
            authEvidence: [
              {
                type: "env-flag",
                flagEnvVars: ["GOOGLE_VERTEX_USE_GCP_METADATA"],
                requiresAnyEnv: ["GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT"],
                requiresAllEnv: ["GOOGLE_CLOUD_LOCATION"],
                credentialMarker: VERTEX_ADC_MARKER,
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

describe("Vertex ADC env marker model scoping", () => {
  it("resolves the marker for a Vertex model but never for a non-Vertex model", async () => {
    const tempRoot = await fs.mkdtemp(path.join(os.tmpdir(), "openclaw-vertex-env-marker-"));
    const workspaceDir = path.join(tempRoot, "workspace");
    const bundledDir = path.join(tempRoot, "bundled");
    const stateDir = path.join(tempRoot, "state");
    await fs.mkdir(bundledDir, { recursive: true });
    await fs.mkdir(stateDir, { recursive: true });
    await writeVertexEnvFlagPlugin(workspaceDir);

    const cfg: OpenClawConfig = { plugins: { allow: ["vertex-meta"] } };
    const store: AuthProfileStore = { version: 1, profiles: {} };

    try {
      await withEnvAsync(
        {
          OPENCLAW_BUNDLED_PLUGINS_DIR: bundledDir,
          OPENCLAW_STATE_DIR: stateDir,
          GOOGLE_VERTEX_USE_GCP_METADATA: "true",
          GOOGLE_CLOUD_PROJECT: "vertex-project",
          GOOGLE_CLOUD_LOCATION: "global",
        },
        async () => {
          // The evidence itself produces the marker (model-agnostic lookup).
          expect(resolveEnvApiKey(PROVIDER, process.env, { config: cfg, workspaceDir })).toEqual({
            apiKey: VERTEX_ADC_MARKER,
            source: "gcp metadata adc",
          });

          // A Vertex model accepts the marker.
          await expect(
            resolveApiKeyForProvider({
              provider: PROVIDER,
              cfg,
              workspaceDir,
              store,
              modelApi: "google-vertex",
            }),
          ).resolves.toEqual({
            apiKey: VERTEX_ADC_MARKER,
            source: "gcp metadata adc",
            mode: "api-key",
          });

          // A non-Vertex model must not receive the Vertex ADC marker; with no
          // other credential available, resolution fails instead of leaking it.
          await expect(
            resolveApiKeyForProvider({
              provider: PROVIDER,
              cfg,
              workspaceDir,
              store,
              modelApi: "google-generative-ai",
            }),
          ).rejects.toThrow();
        },
      );
    } finally {
      await fs.rm(tempRoot, { recursive: true, force: true });
    }
  });

  it("keeps the shared GCP ADC marker for an Anthropic Vertex (anthropic-messages) model", async () => {
    // The gcp-vertex-credentials marker is shared by the Anthropic Vertex plugin,
    // whose catalog models use api: "anthropic-messages". ADC auth must resolve the
    // marker for those models too — the marker scoping only rejects the Google AI
    // Studio (Gemini) surface, never every non-google-vertex model api.
    const tempRoot = await fs.mkdtemp(path.join(os.tmpdir(), "openclaw-anthropic-vertex-marker-"));
    const workspaceDir = path.join(tempRoot, "workspace");
    const bundledDir = path.join(tempRoot, "bundled");
    const stateDir = path.join(tempRoot, "state");
    await fs.mkdir(bundledDir, { recursive: true });
    await fs.mkdir(stateDir, { recursive: true });
    await writeVertexEnvFlagPlugin(workspaceDir, {
      pluginId: "anthropic-vertex-meta",
      providerId: "anthropic-vertex",
      envVar: "ANTHROPIC_VERTEX_META_API_KEY",
    });

    const cfg: OpenClawConfig = { plugins: { allow: ["anthropic-vertex-meta"] } };
    const store: AuthProfileStore = { version: 1, profiles: {} };

    try {
      await withEnvAsync(
        {
          OPENCLAW_BUNDLED_PLUGINS_DIR: bundledDir,
          OPENCLAW_STATE_DIR: stateDir,
          GOOGLE_VERTEX_USE_GCP_METADATA: "true",
          GOOGLE_CLOUD_PROJECT: "vertex-project",
          GOOGLE_CLOUD_LOCATION: "global",
        },
        async () => {
          await expect(
            resolveApiKeyForProvider({
              provider: "anthropic-vertex",
              cfg,
              workspaceDir,
              store,
              modelApi: "anthropic-messages",
            }),
          ).resolves.toEqual({
            apiKey: VERTEX_ADC_MARKER,
            source: "gcp metadata adc",
            mode: "api-key",
          });

          // The same provider still rejects the marker for a Gemini AI Studio model.
          await expect(
            resolveApiKeyForProvider({
              provider: "anthropic-vertex",
              cfg,
              workspaceDir,
              store,
              modelApi: "google-generative-ai",
            }),
          ).rejects.toThrow();
        },
      );
    } finally {
      await fs.rm(tempRoot, { recursive: true, force: true });
    }
  });

  it("keeps the Vertex ADC marker for a google-generative-ai model routed through Vertex", async () => {
    // A model can keep api: "google-generative-ai" yet be dispatched through the
    // Vertex transport because its provider is "google-vertex" or its base URL is
    // an aiplatform.googleapis.com host. Those requests still use GCP ADC, so the
    // marker must be preserved (the AI Studio rejection applies only to genuine
    // generativelanguage.googleapis.com calls).
    const tempRoot = await fs.mkdtemp(path.join(os.tmpdir(), "openclaw-vertex-routed-ggai-"));
    const workspaceDir = path.join(tempRoot, "workspace");
    const bundledDir = path.join(tempRoot, "bundled");
    const stateDir = path.join(tempRoot, "state");
    await fs.mkdir(bundledDir, { recursive: true });
    await fs.mkdir(stateDir, { recursive: true });
    await writeVertexEnvFlagPlugin(workspaceDir, {
      pluginId: "google-vertex-meta",
      providerId: "google-vertex",
      envVar: "GOOGLE_VERTEX_META_API_KEY",
    });
    // Custom providers that also yield the ADC marker, used for the base-URL cases.
    await writeVertexEnvFlagPlugin(workspaceDir, {
      pluginId: "vertex-host-meta",
      providerId: "vertex-host-routed",
      envVar: "VERTEX_HOST_ROUTED_API_KEY",
    });
    await writeVertexEnvFlagPlugin(workspaceDir, {
      pluginId: "vertex-multiregion-meta",
      providerId: "vertex-multiregion-routed",
      envVar: "VERTEX_MULTIREGION_ROUTED_API_KEY",
    });

    const cfg: OpenClawConfig = {
      plugins: { allow: ["google-vertex-meta", "vertex-host-meta", "vertex-multiregion-meta"] },
      models: {
        providers: {
          "vertex-host-routed": {
            api: "google-vertex",
            baseUrl: "https://us-central1-aiplatform.googleapis.com",
            models: [],
          },
          "vertex-multiregion-routed": {
            api: "google-vertex",
            baseUrl: "https://aiplatform.us.rep.googleapis.com",
            models: [],
          },
        },
      },
    };
    const store: AuthProfileStore = { version: 1, profiles: {} };

    try {
      await withEnvAsync(
        {
          OPENCLAW_BUNDLED_PLUGINS_DIR: bundledDir,
          OPENCLAW_STATE_DIR: stateDir,
          GOOGLE_VERTEX_USE_GCP_METADATA: "true",
          GOOGLE_CLOUD_PROJECT: "vertex-project",
          GOOGLE_CLOUD_LOCATION: "global",
        },
        async () => {
          // Routed through Vertex by provider id.
          await expect(
            resolveApiKeyForProvider({
              provider: "google-vertex",
              cfg,
              workspaceDir,
              store,
              modelApi: "google-generative-ai",
            }),
          ).resolves.toEqual({
            apiKey: VERTEX_ADC_MARKER,
            source: "gcp metadata adc",
            mode: "api-key",
          });

          // Routed through Vertex by a regional aiplatform base URL.
          await expect(
            resolveApiKeyForProvider({
              provider: "vertex-host-routed",
              cfg,
              workspaceDir,
              store,
              modelApi: "google-generative-ai",
            }),
          ).resolves.toEqual({
            apiKey: VERTEX_ADC_MARKER,
            source: "gcp metadata adc",
            mode: "api-key",
          });

          // Routed through Vertex by a multi-region (.rep.googleapis.com) base URL.
          await expect(
            resolveApiKeyForProvider({
              provider: "vertex-multiregion-routed",
              cfg,
              workspaceDir,
              store,
              modelApi: "google-generative-ai",
            }),
          ).resolves.toEqual({
            apiKey: VERTEX_ADC_MARKER,
            source: "gcp metadata adc",
            mode: "api-key",
          });
        },
      );
    } finally {
      await fs.rm(tempRoot, { recursive: true, force: true });
    }
  });

  it("falls back to the google-vertex auth provider for a Vertex model under the google provider id", async () => {
    // Deployment shape: a "google-vertex" model is registered under the generic
    // "google" provider id, and ambient credentials come from the metadata-server
    // opt-in (env-flag evidence keyed under the dedicated "google-vertex" auth
    // provider). The request provider id is "google", so resolution must fall back
    // to "google-vertex" to find the evidence rather than failing as missing auth.
    const tempRoot = await fs.mkdtemp(path.join(os.tmpdir(), "openclaw-vertex-env-fallback-"));
    const workspaceDir = path.join(tempRoot, "workspace");
    const bundledDir = path.join(tempRoot, "bundled");
    const stateDir = path.join(tempRoot, "state");
    await fs.mkdir(bundledDir, { recursive: true });
    await fs.mkdir(stateDir, { recursive: true });
    await writeGoogleVertexMixedPlugin(workspaceDir);

    const cfg: OpenClawConfig = { plugins: { allow: ["google-mixed"] } };
    const store: AuthProfileStore = { version: 1, profiles: {} };

    try {
      await withEnvAsync(
        {
          OPENCLAW_BUNDLED_PLUGINS_DIR: bundledDir,
          OPENCLAW_STATE_DIR: stateDir,
          GOOGLE_VERTEX_USE_GCP_METADATA: "true",
          GOOGLE_CLOUD_PROJECT: "vertex-project",
          GOOGLE_CLOUD_LOCATION: "global",
        },
        async () => {
          // A Vertex model under the generic "google" provider id resolves the
          // marker via the google-vertex auth-provider fallback.
          await expect(
            resolveApiKeyForProvider({
              provider: "google",
              cfg,
              workspaceDir,
              store,
              modelApi: "google-vertex",
            }),
          ).resolves.toEqual({
            apiKey: VERTEX_ADC_MARKER,
            source: "gcp metadata adc",
            mode: "api-key",
          });

          // A generic Google (Gemini) env key must not shadow the Vertex ADC
          // evidence for a Vertex model: the Gemini key is not a valid Vertex
          // credential, so the google-vertex auth provider is consulted first.
          await withEnvAsync({ GEMINI_API_KEY: "gemini-not-for-vertex" }, async () => {
            await expect(
              resolveApiKeyForProvider({
                provider: "google",
                cfg,
                workspaceDir,
                store,
                modelApi: "google-vertex",
              }),
            ).resolves.toEqual({
              apiKey: VERTEX_ADC_MARKER,
              source: "gcp metadata adc",
              mode: "api-key",
            });
          });

          // With the metadata opt-in absent there is no valid Vertex env
          // credential, so a present Gemini key must NOT be accepted for a Vertex
          // model: resolution fails instead of sending an AI Studio key to Vertex.
          await withEnvAsync(
            {
              GOOGLE_VERTEX_USE_GCP_METADATA: undefined,
              GEMINI_API_KEY: "gemini-not-for-vertex",
            },
            async () => {
              await expect(
                resolveApiKeyForProvider({
                  provider: "google",
                  cfg,
                  workspaceDir,
                  store,
                  modelApi: "google-vertex",
                }),
              ).rejects.toThrow();
            },
          );

          // A non-Vertex model under "google" must not trigger the fallback or
          // receive the Vertex ADC marker; resolution fails instead of leaking it.
          await expect(
            resolveApiKeyForProvider({
              provider: "google",
              cfg,
              workspaceDir,
              store,
              modelApi: "google-generative-ai",
            }),
          ).rejects.toThrow();

          // A model-agnostic, provider-level "google" caller (no model api) must
          // not pick up the Vertex evidence via the fallback either.
          await expect(
            resolveApiKeyForProvider({
              provider: "google",
              cfg,
              workspaceDir,
              store,
            }),
          ).rejects.toThrow();
        },
      );
    } finally {
      await fs.rm(tempRoot, { recursive: true, force: true });
    }
  });

  it("treats a base-url-routed google-generative-ai model under the google provider as Vertex", async () => {
    // Deployment shape: a model keeps api "google-generative-ai" but its base URL is
    // an aiplatform Vertex host, so the Google plugin dispatches it through the
    // Vertex transport. Auth must mirror a real Vertex request under the generic
    // "google" id: fall back to the google-vertex evidence, never accept a Gemini
    // key, and report availability consistently.
    const tempRoot = await fs.mkdtemp(path.join(os.tmpdir(), "openclaw-vertex-baseurl-google-"));
    const workspaceDir = path.join(tempRoot, "workspace");
    const bundledDir = path.join(tempRoot, "bundled");
    const stateDir = path.join(tempRoot, "state");
    await fs.mkdir(bundledDir, { recursive: true });
    await fs.mkdir(stateDir, { recursive: true });
    await writeGoogleVertexMixedPlugin(workspaceDir);

    const cfg: OpenClawConfig = { plugins: { allow: ["google-mixed"] } };
    const store: AuthProfileStore = { version: 1, profiles: {} };
    const vertexBaseUrl = "https://us-central1-aiplatform.googleapis.com";

    try {
      await withEnvAsync(
        {
          OPENCLAW_BUNDLED_PLUGINS_DIR: bundledDir,
          OPENCLAW_STATE_DIR: stateDir,
          GOOGLE_VERTEX_USE_GCP_METADATA: "true",
          GOOGLE_CLOUD_PROJECT: "vertex-project",
          GOOGLE_CLOUD_LOCATION: "global",
        },
        async () => {
          // The aiplatform base URL routes the request through Vertex: resolve the
          // marker via the google-vertex fallback.
          await expect(
            resolveApiKeyForProvider({
              provider: "google",
              cfg,
              workspaceDir,
              store,
              modelApi: "google-generative-ai",
              baseUrl: vertexBaseUrl,
            }),
          ).resolves.toEqual({
            apiKey: VERTEX_ADC_MARKER,
            source: "gcp metadata adc",
            mode: "api-key",
          });
          await expect(
            hasAvailableAuthForProvider({
              provider: "google",
              cfg,
              workspaceDir,
              store,
              modelApi: "google-generative-ai",
              baseUrl: vertexBaseUrl,
            }),
          ).resolves.toBe(true);

          // A present Gemini key must not shadow the Vertex evidence for the
          // base-url-routed request.
          await withEnvAsync({ GEMINI_API_KEY: "gemini-not-for-vertex" }, async () => {
            await expect(
              resolveApiKeyForProvider({
                provider: "google",
                cfg,
                workspaceDir,
                store,
                modelApi: "google-generative-ai",
                baseUrl: vertexBaseUrl,
              }),
            ).resolves.toEqual({
              apiKey: VERTEX_ADC_MARKER,
              source: "gcp metadata adc",
              mode: "api-key",
            });
          });

          // Without the metadata opt-in, a present Gemini key must NOT satisfy the
          // base-url-routed Vertex request.
          await withEnvAsync(
            {
              GOOGLE_VERTEX_USE_GCP_METADATA: undefined,
              GEMINI_API_KEY: "gemini-not-for-vertex",
            },
            async () => {
              await expect(
                resolveApiKeyForProvider({
                  provider: "google",
                  cfg,
                  workspaceDir,
                  store,
                  modelApi: "google-generative-ai",
                  baseUrl: vertexBaseUrl,
                }),
              ).rejects.toThrow();
            },
          );

          // The same model with no Vertex base URL is genuine AI Studio: the marker
          // is not leaked and resolution fails without a Gemini key.
          await expect(
            resolveApiKeyForProvider({
              provider: "google",
              cfg,
              workspaceDir,
              store,
              modelApi: "google-generative-ai",
            }),
          ).rejects.toThrow();
        },
      );
    } finally {
      await fs.rm(tempRoot, { recursive: true, force: true });
    }
  });

  it("drops the ADC marker for the OpenAI-compatible Vertex endpoint (openai-completions)", async () => {
    // A google-vertex provider can be configured for the OpenAI-compatible Vertex
    // endpoint (api: openai-completions, baseUrl .../endpoints/openapi). That host is
    // an aiplatform host, but the request is dispatched through the OpenAI transport,
    // which would send the ADC marker as a literal bearer key. The marker must be
    // dropped for openai-completions even though the base URL is a Vertex host, while
    // a real Vertex (google-vertex api) request under the same provider still gets it.
    const tempRoot = await fs.mkdtemp(path.join(os.tmpdir(), "openclaw-vertex-openai-compat-"));
    const workspaceDir = path.join(tempRoot, "workspace");
    const bundledDir = path.join(tempRoot, "bundled");
    const stateDir = path.join(tempRoot, "state");
    await fs.mkdir(bundledDir, { recursive: true });
    await fs.mkdir(stateDir, { recursive: true });
    await writeVertexEnvFlagPlugin(workspaceDir, {
      pluginId: "vertex-openai-compat-meta",
      providerId: "vertex-openai-compat",
      envVar: "VERTEX_OPENAI_COMPAT_API_KEY",
    });

    const cfg: OpenClawConfig = {
      plugins: { allow: ["vertex-openai-compat-meta"] },
      models: {
        providers: {
          "vertex-openai-compat": {
            api: "openai-completions",
            baseUrl:
              "https://aiplatform.googleapis.com/v1/projects/p/locations/global/endpoints/openapi",
            models: [],
          },
        },
      },
    };
    const store: AuthProfileStore = { version: 1, profiles: {} };

    try {
      await withEnvAsync(
        {
          OPENCLAW_BUNDLED_PLUGINS_DIR: bundledDir,
          OPENCLAW_STATE_DIR: stateDir,
          GOOGLE_VERTEX_USE_GCP_METADATA: "true",
          GOOGLE_CLOUD_PROJECT: "vertex-project",
          GOOGLE_CLOUD_LOCATION: "global",
        },
        async () => {
          // openai-completions on the aiplatform host: marker dropped, no other auth.
          await expect(
            resolveApiKeyForProvider({
              provider: "vertex-openai-compat",
              cfg,
              workspaceDir,
              store,
              modelApi: "openai-completions",
              baseUrl:
                "https://aiplatform.googleapis.com/v1/projects/p/locations/global/endpoints/openapi",
            }),
          ).rejects.toThrow();

          // A real Vertex (google-vertex api) request under the same provider keeps
          // the marker: it dispatches through the Vertex transport that exchanges it.
          await expect(
            resolveApiKeyForProvider({
              provider: "vertex-openai-compat",
              cfg,
              workspaceDir,
              store,
              modelApi: "google-vertex",
            }),
          ).resolves.toEqual({
            apiKey: VERTEX_ADC_MARKER,
            source: "gcp metadata adc",
            mode: "api-key",
          });
        },
      );
    } finally {
      await fs.rm(tempRoot, { recursive: true, force: true });
    }
  });

  it("keeps a provider-specific Vertex env API key for a Vertex model under a custom provider id", async () => {
    // A custom/dedicated Vertex provider declares its own env API key (not the ADC
    // marker). The marker-only rejection applies only to the generic "google"
    // provider, so this provider's legitimate key must still resolve.
    const tempRoot = await fs.mkdtemp(path.join(os.tmpdir(), "openclaw-vertex-express-"));
    const workspaceDir = path.join(tempRoot, "workspace");
    const bundledDir = path.join(tempRoot, "bundled");
    const stateDir = path.join(tempRoot, "state");
    await fs.mkdir(bundledDir, { recursive: true });
    await fs.mkdir(stateDir, { recursive: true });
    await writeCustomVertexKeyPlugin(workspaceDir);

    const cfg: OpenClawConfig = { plugins: { allow: ["vertex-express"] } };
    const store: AuthProfileStore = { version: 1, profiles: {} };

    try {
      await withEnvAsync(
        {
          OPENCLAW_BUNDLED_PLUGINS_DIR: bundledDir,
          OPENCLAW_STATE_DIR: stateDir,
          VERTEX_EXPRESS_API_KEY: "vertex-express-real-key",
        },
        async () => {
          const resolved = await resolveApiKeyForProvider({
            provider: "vertex-express",
            cfg,
            workspaceDir,
            store,
            modelApi: "google-vertex",
          });
          expect(resolved.apiKey).toBe("vertex-express-real-key");
          expect(resolved.mode).toBe("api-key");
        },
      );
    } finally {
      await fs.rm(tempRoot, { recursive: true, force: true });
    }
  });

  it("falls back to a stored google-vertex profile for a Vertex model under the google provider id", async () => {
    // A Vertex model registered under the shared "google" provider resolves auth
    // from the dedicated "google-vertex" provider, including its stored profiles —
    // not only env ADC evidence. This mirrors the model-list availability index and
    // the session model registry so a model reported as available also dispatches.
    const store: AuthProfileStore = {
      version: 1,
      profiles: {
        "google-vertex:default": {
          type: "api_key",
          provider: "google-vertex",
          key: "vertex-profile-key", // pragma: allowlist secret
        },
      },
    };

    await withEnvAsync(
      {
        GEMINI_API_KEY: undefined,
        GOOGLE_API_KEY: undefined,
        GOOGLE_CLOUD_API_KEY: undefined,
        GOOGLE_VERTEX_USE_GCP_METADATA: undefined,
      },
      async () => {
        const resolved = await resolveApiKeyForProvider({
          provider: "google",
          store,
          modelApi: "google-vertex",
        });
        expect(resolved.apiKey).toBe("vertex-profile-key");

        await expect(
          hasAvailableAuthForProvider({ provider: "google", store, modelApi: "google-vertex" }),
        ).resolves.toBe(true);

        // A non-Vertex (Gemini) model under "google" must NOT borrow the
        // google-vertex profile: it is not a valid AI Studio credential.
        await expect(
          resolveApiKeyForProvider({
            provider: "google",
            store,
            modelApi: "google-generative-ai",
          }),
        ).rejects.toThrow();
        await expect(
          hasAvailableAuthForProvider({
            provider: "google",
            store,
            modelApi: "google-generative-ai",
          }),
        ).resolves.toBe(false);
      },
    );
  });

  it("does not let a stored google (Gemini) profile preempt google-vertex auth for a Vertex model", async () => {
    // A user can have both a generic "google" (AI Studio/Gemini) profile and a
    // dedicated "google-vertex" profile. A Vertex model under "google" must
    // dispatch with the google-vertex credential, never the Gemini profile (which
    // the Vertex transport rejects), even though the google profile is ordered
    // first for the requested provider id.
    const store: AuthProfileStore = {
      version: 1,
      profiles: {
        "google:default": {
          type: "api_key",
          provider: "google",
          key: "gemini-not-for-vertex", // pragma: allowlist secret
        },
        "google-vertex:default": {
          type: "api_key",
          provider: "google-vertex",
          key: "vertex-profile-key", // pragma: allowlist secret
        },
      },
    };

    await withEnvAsync(
      {
        GEMINI_API_KEY: undefined,
        GOOGLE_API_KEY: undefined,
        GOOGLE_CLOUD_API_KEY: undefined,
        GOOGLE_VERTEX_USE_GCP_METADATA: undefined,
      },
      async () => {
        const resolved = await resolveApiKeyForProvider({
          provider: "google",
          store,
          modelApi: "google-vertex",
        });
        expect(resolved.apiKey).toBe("vertex-profile-key");

        // With only the Gemini profile present, a Vertex request must fail rather
        // than dispatch the AI Studio key to Vertex.
        const geminiOnlyStore: AuthProfileStore = {
          version: 1,
          profiles: {
            "google:default": {
              type: "api_key",
              provider: "google",
              key: "gemini-not-for-vertex", // pragma: allowlist secret
            },
          },
        };
        await expect(
          resolveApiKeyForProvider({
            provider: "google",
            store: geminiOnlyStore,
            modelApi: "google-vertex",
          }),
        ).rejects.toThrow();
        await expect(
          hasAvailableAuthForProvider({
            provider: "google",
            store: geminiOnlyStore,
            modelApi: "google-vertex",
          }),
        ).resolves.toBe(false);
      },
    );
  });

  it("prefers a custom provider's own Vertex env key over the global google-vertex ADC", async () => {
    // Both the custom provider's own key and the global google-vertex ADC marker
    // are available. A Vertex request under the custom provider must NOT be
    // switched to the global ADC credential; the provider-specific key wins.
    const tempRoot = await fs.mkdtemp(path.join(os.tmpdir(), "openclaw-vertex-express-mixed-"));
    const workspaceDir = path.join(tempRoot, "workspace");
    const bundledDir = path.join(tempRoot, "bundled");
    const stateDir = path.join(tempRoot, "state");
    await fs.mkdir(bundledDir, { recursive: true });
    await fs.mkdir(stateDir, { recursive: true });
    await writeCustomVertexWithGlobalAdcPlugin(workspaceDir);

    const cfg: OpenClawConfig = { plugins: { allow: ["vertex-express-mixed"] } };
    const store: AuthProfileStore = { version: 1, profiles: {} };

    try {
      await withEnvAsync(
        {
          OPENCLAW_BUNDLED_PLUGINS_DIR: bundledDir,
          OPENCLAW_STATE_DIR: stateDir,
          VERTEX_EXPRESS_API_KEY: "vertex-express-real-key",
          GOOGLE_VERTEX_USE_GCP_METADATA: "true",
          GOOGLE_CLOUD_PROJECT: "vertex-project",
          GOOGLE_CLOUD_LOCATION: "global",
        },
        async () => {
          const resolved = await resolveApiKeyForProvider({
            provider: "vertex-express",
            cfg,
            workspaceDir,
            store,
            modelApi: "google-vertex",
          });
          expect(resolved.apiKey).toBe("vertex-express-real-key");
          expect(resolved.mode).toBe("api-key");

          // And when the custom provider declares NO usable key, the global
          // google-vertex ADC marker is still available as a fallback.
          await withEnvAsync({ VERTEX_EXPRESS_API_KEY: undefined }, async () => {
            await expect(
              resolveApiKeyForProvider({
                provider: "vertex-express",
                cfg,
                workspaceDir,
                store,
                modelApi: "google-vertex",
              }),
            ).resolves.toEqual({
              apiKey: VERTEX_ADC_MARKER,
              source: "gcp metadata adc",
              mode: "api-key",
            });
          });
        },
      );
    } finally {
      await fs.rm(tempRoot, { recursive: true, force: true });
    }
  });

  it("ignores an explicit generic-google profile candidate for a Vertex model", async () => {
    // The auth controller passes each profile candidate as an explicit profileId.
    // For a Vertex model under the shared "google" provider, the generic "google"
    // (Gemini) candidate must be skipped in favor of the dedicated google-vertex
    // credential rather than dispatched to the Vertex transport.
    const store: AuthProfileStore = {
      version: 1,
      profiles: {
        "google:default": {
          type: "api_key",
          provider: "google",
          key: "gemini-not-for-vertex", // pragma: allowlist secret
        },
        "google-vertex:default": {
          type: "api_key",
          provider: "google-vertex",
          key: "vertex-profile-key", // pragma: allowlist secret
        },
      },
    };

    await withEnvAsync(
      {
        GEMINI_API_KEY: undefined,
        GOOGLE_API_KEY: undefined,
        GOOGLE_CLOUD_API_KEY: undefined,
        GOOGLE_VERTEX_USE_GCP_METADATA: undefined,
      },
      async () => {
        // An explicit generic "google" candidate is ignored; the google-vertex
        // credential is used instead.
        const resolved = await resolveApiKeyForProvider({
          provider: "google",
          store,
          modelApi: "google-vertex",
          profileId: "google:default",
        });
        expect(resolved.apiKey).toBe("vertex-profile-key");

        // An explicit google-vertex candidate is still honored directly.
        const resolvedVertex = await resolveApiKeyForProvider({
          provider: "google",
          store,
          modelApi: "google-vertex",
          profileId: "google-vertex:default",
        });
        expect(resolvedVertex.apiKey).toBe("vertex-profile-key");

        // With only the Gemini profile present, an explicit Gemini candidate must
        // not be dispatched to Vertex; the request fails instead.
        const geminiOnlyStore: AuthProfileStore = {
          version: 1,
          profiles: {
            "google:default": {
              type: "api_key",
              provider: "google",
              key: "gemini-not-for-vertex", // pragma: allowlist secret
            },
          },
        };
        await expect(
          resolveApiKeyForProvider({
            provider: "google",
            store: geminiOnlyStore,
            modelApi: "google-vertex",
            profileId: "google:default",
          }),
        ).rejects.toThrow();
      },
    );
  });

  it("keeps the Vertex ADC marker when a model base URL routes a google-generative-ai request through Vertex", async () => {
    // A per-model baseUrl override can route a request through the Vertex transport
    // even when the provider config carries no Vertex host. The auth path must use
    // the model-level base URL (not just the provider-level one) so an ADC-marker
    // credential survives instead of being dropped as an AI Studio request.
    const tempRoot = await fs.mkdtemp(path.join(os.tmpdir(), "openclaw-vertex-model-baseurl-"));
    const workspaceDir = path.join(tempRoot, "workspace");
    const bundledDir = path.join(tempRoot, "bundled");
    const stateDir = path.join(tempRoot, "state");
    await fs.mkdir(bundledDir, { recursive: true });
    await fs.mkdir(stateDir, { recursive: true });
    await writeVertexEnvFlagPlugin(workspaceDir, {
      pluginId: "vertex-model-baseurl-meta",
      providerId: "vertex-model-baseurl",
      envVar: "VERTEX_MODEL_BASEURL_API_KEY",
    });

    // The provider config intentionally carries NO Vertex base URL: only the
    // model-level baseUrl makes the request Vertex-routed.
    const cfg: OpenClawConfig = {
      plugins: { allow: ["vertex-model-baseurl-meta"] },
      models: {
        providers: {
          "vertex-model-baseurl": {
            api: "google-vertex",
            // A non-Vertex provider base URL: only the per-model baseUrl below routes
            // the request through Vertex.
            baseUrl: "https://generativelanguage.googleapis.com",
            models: [],
          },
        },
      },
    };
    const store: AuthProfileStore = { version: 1, profiles: {} };

    try {
      await withEnvAsync(
        {
          OPENCLAW_BUNDLED_PLUGINS_DIR: bundledDir,
          OPENCLAW_STATE_DIR: stateDir,
          GOOGLE_VERTEX_USE_GCP_METADATA: "true",
          GOOGLE_CLOUD_PROJECT: "vertex-project",
          GOOGLE_CLOUD_LOCATION: "global",
        },
        async () => {
          // The model base URL points at an aiplatform host: marker preserved.
          await expect(
            resolveApiKeyForProvider({
              provider: "vertex-model-baseurl",
              cfg,
              workspaceDir,
              store,
              modelApi: "google-generative-ai",
              baseUrl: "https://us-central1-aiplatform.googleapis.com",
            }),
          ).resolves.toEqual({
            apiKey: VERTEX_ADC_MARKER,
            source: "gcp metadata adc",
            mode: "api-key",
          });

          // No Vertex base URL anywhere: the google-generative-ai request is treated
          // as genuine AI Studio, the marker is dropped, and no auth resolves.
          await expect(
            resolveApiKeyForProvider({
              provider: "vertex-model-baseurl",
              cfg,
              workspaceDir,
              store,
              modelApi: "google-generative-ai",
            }),
          ).rejects.toThrow();
        },
      );
    } finally {
      await fs.rm(tempRoot, { recursive: true, force: true });
    }
  });
});
