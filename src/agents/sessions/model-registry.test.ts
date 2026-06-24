// Model registry tests cover models.json auth modes and plugin-owned model
// catalog shards.
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { withEnvAsync } from "../../test-utils/env.js";
import {
  PLUGIN_MODEL_CATALOG_FILE,
  PLUGIN_MODEL_CATALOG_GENERATED_BY,
} from "../plugin-model-catalog.js";
import { AuthStorage } from "./auth-storage.js";
import { ModelRegistry } from "./model-registry.js";

const tempDirs: string[] = [];

function writeModelsJson(contents: unknown): string {
  const dir = mkdtempSync(join(tmpdir(), "openclaw-model-registry-"));
  tempDirs.push(dir);
  const file = join(dir, "models.json");
  writeFileSync(file, JSON.stringify(contents, null, 2), "utf-8");
  return file;
}

function writeModelsJsonWithPluginCatalog(params: {
  root: unknown;
  pluginRelativePath: string;
  pluginCatalog: unknown;
}): string {
  return writeModelsJsonWithPluginCatalogs({
    root: params.root,
    pluginCatalogs: [
      {
        pluginRelativePath: params.pluginRelativePath,
        pluginCatalog: params.pluginCatalog,
      },
    ],
  });
}

function writeModelsJsonWithPluginCatalogs(params: {
  root: unknown;
  pluginCatalogs: Array<{
    pluginRelativePath: string;
    pluginCatalog: unknown;
  }>;
}): string {
  const dir = mkdtempSync(join(tmpdir(), "openclaw-model-registry-"));
  tempDirs.push(dir);
  const file = join(dir, "models.json");
  writeFileSync(file, JSON.stringify(params.root, null, 2), "utf-8");
  for (const pluginCatalog of params.pluginCatalogs) {
    const pluginFile = join(dir, pluginCatalog.pluginRelativePath);
    mkdirSync(dirname(pluginFile), { recursive: true });
    writeFileSync(pluginFile, JSON.stringify(pluginCatalog.pluginCatalog, null, 2), "utf-8");
  }
  return file;
}

function pluginOwnerSnapshot(providerId: string, pluginId: string, enabled = true) {
  return pluginOwnerSnapshotEntries([{ providerId, pluginId, enabled }]);
}

function pluginOwnerSnapshotEntries(
  entries: Array<{ providerId: string; pluginId: string; enabled?: boolean }>,
) {
  // The registry only trusts generated provider shards that are still owned by
  // an enabled plugin in the current metadata snapshot.
  return {
    index: {
      plugins: entries.map((entry) => ({
        pluginId: entry.pluginId,
        enabled: entry.enabled ?? true,
      })),
    },
    normalizePluginId: (id: string) => id,
    owners: {
      channels: new Map(),
      channelConfigs: new Map(),
      providers: new Map(entries.map((entry) => [entry.providerId, [entry.pluginId]])),
      modelCatalogProviders: new Map(entries.map((entry) => [entry.providerId, [entry.pluginId]])),
      cliBackends: new Map(),
      setupProviders: new Map(),
      commandAliases: new Map(),
      contracts: new Map(),
    },
  };
}

afterEach(() => {
  for (const dir of tempDirs.splice(0)) {
    rmSync(dir, { recursive: true, force: true });
  }
});

describe("ModelRegistry models.json auth", () => {
  it("accepts Bedrock AWS SDK auth without apiKey", async () => {
    // AWS SDK credential resolution is provider-owned; requiring an apiKey here
    // would make Bedrock catalogs impossible to express in models.json.
    const modelsPath = writeModelsJson({
      providers: {
        "amazon-bedrock": {
          baseUrl: "https://bedrock-runtime.us-east-1.amazonaws.com",
          api: "bedrock-converse-stream",
          auth: "aws-sdk",
          models: [
            {
              id: "anthropic.claude-sonnet-4-5-20250929-v1:0",
              name: "Claude Sonnet 4.5",
            },
          ],
        },
      },
    });

    const registry = ModelRegistry.create(AuthStorage.inMemory(), modelsPath);
    const model = registry.find("amazon-bedrock", "anthropic.claude-sonnet-4-5-20250929-v1:0");

    expect(registry.getError()).toBeUndefined();
    expect(model).toBeDefined();
    expect(registry.getAvailable()).toEqual([model]);
    await expect(registry.getApiKeyAndHeaders(model!)).resolves.toEqual({
      ok: true,
      apiKey: undefined,
      headers: undefined,
    });
    expect(registry.getProviderAuthStatus("amazon-bedrock")).toEqual({
      configured: true,
      source: "models_json_key",
      label: "aws-sdk",
    });
  });

  it("treats a Vertex model under the google provider as available via google-vertex auth", async () => {
    // A Vertex model can be registered under the shared "google" provider yet have
    // its credentials keyed under the dedicated "google-vertex" provider (e.g. GKE
    // metadata-server ADC). Both hasConfiguredAuth() (availability) and
    // getApiKeyAndHeaders() (request auth) must mirror that fallback so an
    // advertised model also dispatches with the right credential.
    const modelsPath = writeModelsJson({
      providers: {
        google: {
          baseUrl: "https://aiplatform.googleapis.com",
          api: "google-vertex",
          // No literal/provider apiKey: auth resolves via the google-vertex provider.
          auth: "oauth",
          models: [{ id: "gemini-2.5-flash", name: "Gemini 2.5 Flash" }],
        },
      },
    });

    await withEnvAsync(
      {
        GOOGLE_VERTEX_USE_GCP_METADATA: undefined,
        GOOGLE_CLOUD_PROJECT: undefined,
        GCLOUD_PROJECT: undefined,
        GOOGLE_CLOUD_LOCATION: undefined,
        GOOGLE_API_KEY: undefined,
        GEMINI_API_KEY: undefined,
      },
      async () => {
        const withVertexAuth = ModelRegistry.create(
          AuthStorage.inMemory({ "google-vertex": { type: "api_key", key: "vertex-adc" } }),
          modelsPath,
        );
        const model = withVertexAuth.find("google", "gemini-2.5-flash");
        expect(withVertexAuth.getError()).toBeUndefined();
        expect(model).toBeDefined();
        expect(withVertexAuth.getAvailable()).toEqual([model]);
        // Request auth resolves the google-vertex credential, not "no API key".
        await expect(withVertexAuth.getApiKeyAndHeaders(model!)).resolves.toEqual({
          ok: true,
          apiKey: "vertex-adc",
          headers: undefined,
        });

        const withoutVertexAuth = ModelRegistry.create(AuthStorage.inMemory(), modelsPath);
        expect(withoutVertexAuth.getAvailable()).toEqual([]);
      },
    );
  });

  it("does not let a Gemini key under the google provider satisfy or dispatch a Vertex model", async () => {
    // A Vertex model (api: "google-vertex") registered under the shared "google"
    // provider must never be authenticated or dispatched with the "google"
    // provider's AI Studio (Gemini) credential — the Vertex transport rejects it.
    // Only the dedicated "google-vertex" provider satisfies a Vertex request,
    // mirroring the model-scoped availability/listing path.
    const modelsPath = writeModelsJson({
      providers: {
        google: {
          baseUrl: "https://aiplatform.googleapis.com",
          api: "google-vertex",
          auth: "oauth",
          models: [{ id: "gemini-2.5-flash", name: "Gemini 2.5 Flash" }],
        },
      },
    });

    await withEnvAsync(
      {
        GOOGLE_VERTEX_USE_GCP_METADATA: undefined,
        GOOGLE_CLOUD_PROJECT: undefined,
        GCLOUD_PROJECT: undefined,
        GOOGLE_CLOUD_LOCATION: undefined,
        GOOGLE_API_KEY: undefined,
        GEMINI_API_KEY: undefined,
      },
      async () => {
        // Only a Gemini key under "google" — no google-vertex auth.
        const geminiOnly = ModelRegistry.create(
          AuthStorage.inMemory({ google: { type: "api_key", key: "gemini-key" } }),
          modelsPath,
        );
        const model = geminiOnly.find("google", "gemini-2.5-flash");
        expect(geminiOnly.getError()).toBeUndefined();
        expect(model).toBeDefined();
        // Not available: the Gemini key cannot satisfy a Vertex model.
        expect(geminiOnly.getAvailable()).toEqual([]);
        // Request auth must not leak the Gemini key to the Vertex transport.
        await expect(geminiOnly.getApiKeyAndHeaders(model!)).resolves.toEqual({
          ok: true,
          apiKey: undefined,
          headers: undefined,
        });

        // With both a Gemini key under "google" and dedicated google-vertex auth,
        // dispatch uses the Vertex credential, never the Gemini key.
        const both = ModelRegistry.create(
          AuthStorage.inMemory({
            google: { type: "api_key", key: "gemini-key" },
            "google-vertex": { type: "api_key", key: "vertex-adc" },
          }),
          modelsPath,
        );
        const bothModel = both.find("google", "gemini-2.5-flash");
        expect(both.getAvailable()).toEqual([bothModel]);
        await expect(both.getApiKeyAndHeaders(bothModel!)).resolves.toEqual({
          ok: true,
          apiKey: "vertex-adc",
          headers: undefined,
        });
      },
    );
  });

  it("honors a configured Vertex key on a google provider declared as google-vertex", async () => {
    // A models.json provider literally named "google" may declare api:
    // "google-vertex" and supply its own apiKey (a literal Vertex Express key,
    // GOOGLE_CLOUD_API_KEY, or the ADC marker). That configured key is a legitimate
    // Vertex credential and must satisfy availability and dispatch even without a
    // separate google-vertex auth-storage credential.
    const modelsPath = writeModelsJson({
      providers: {
        google: {
          baseUrl: "https://aiplatform.googleapis.com",
          api: "google-vertex",
          apiKey: "vertex-express-key",
          models: [{ id: "gemini-2.5-flash", name: "Gemini 2.5 Flash" }],
        },
      },
    });

    await withEnvAsync(
      {
        GOOGLE_VERTEX_USE_GCP_METADATA: undefined,
        GOOGLE_CLOUD_PROJECT: undefined,
        GCLOUD_PROJECT: undefined,
        GOOGLE_CLOUD_LOCATION: undefined,
        GOOGLE_API_KEY: undefined,
        GEMINI_API_KEY: undefined,
      },
      async () => {
        // No google-vertex auth storage: the configured provider key is the source.
        const registry = ModelRegistry.create(AuthStorage.inMemory(), modelsPath);
        const model = registry.find("google", "gemini-2.5-flash");
        expect(registry.getError()).toBeUndefined();
        expect(model).toBeDefined();
        expect(registry.getAvailable()).toEqual([model]);
        await expect(registry.getApiKeyAndHeaders(model!)).resolves.toEqual({
          ok: true,
          apiKey: "vertex-express-key",
          headers: undefined,
        });
      },
    );
  });

  it("still rejects api-key custom models without apiKey", () => {
    const modelsPath = writeModelsJson({
      providers: {
        custom: {
          baseUrl: "https://models.example/v1",
          api: "openai-responses",
          models: [{ id: "example-model" }],
        },
      },
    });

    const registry = ModelRegistry.create(AuthStorage.inMemory(), modelsPath);

    expect(registry.getError()).toContain('Provider custom: "apiKey" is required');
    expect(registry.find("custom", "example-model")).toBeUndefined();
  });

  it("loads provider models from generated plugin catalog shards", () => {
    const modelsPath = writeModelsJsonWithPluginCatalog({
      root: { providers: {} },
      pluginRelativePath: join("plugins", "zai", PLUGIN_MODEL_CATALOG_FILE),
      pluginCatalog: {
        generatedBy: PLUGIN_MODEL_CATALOG_GENERATED_BY,
        providers: {
          zai: {
            baseUrl: "https://api.z.ai/api/paas/v4",
            api: "openai-completions",
            apiKey: "ZAI_API_KEY",
            models: [{ id: "glm-5.1", name: "GLM 5.1" }],
          },
        },
      },
    });

    const registry = ModelRegistry.create(
      AuthStorage.inMemory({ zai: { type: "api_key", key: "sk-test" } }),
      modelsPath,
      { pluginMetadataSnapshot: pluginOwnerSnapshot("zai", "zai") },
    );

    expect(registry.getError()).toBeUndefined();
    expect(registry.find("zai", "glm-5.1")?.name).toBe("GLM 5.1");
  });

  it("isolates invalid generated plugin catalog shards from valid models", () => {
    const modelsPath = writeModelsJsonWithPluginCatalogs({
      root: {
        providers: {
          custom: {
            baseUrl: "https://models.example/v1",
            api: "openai-responses",
            apiKey: "CUSTOM_API_KEY",
            models: [{ id: "root-model", name: "Root Model" }],
          },
        },
      },
      pluginCatalogs: [
        {
          pluginRelativePath: join("plugins", "google", PLUGIN_MODEL_CATALOG_FILE),
          pluginCatalog: {
            generatedBy: PLUGIN_MODEL_CATALOG_GENERATED_BY,
            providers: {
              "google-vertex": {
                baseUrl: "https://us-central1-aiplatform.googleapis.com/v1",
                apiKey: "GOOGLE_API_KEY",
                models: [{ id: "gemini-3.1-pro-preview", name: "Gemini 3.1 Pro" }],
              },
            },
          },
        },
        {
          pluginRelativePath: join("plugins", "zai", PLUGIN_MODEL_CATALOG_FILE),
          pluginCatalog: {
            generatedBy: PLUGIN_MODEL_CATALOG_GENERATED_BY,
            providers: {
              zai: {
                baseUrl: "https://api.z.ai/api/paas/v4",
                api: "openai-completions",
                apiKey: "ZAI_API_KEY",
                models: [{ id: "glm-5.1", name: "GLM 5.1" }],
              },
            },
          },
        },
      ],
    });

    const registry = ModelRegistry.create(AuthStorage.inMemory(), modelsPath, {
      pluginMetadataSnapshot: pluginOwnerSnapshotEntries([
        { providerId: "google-vertex", pluginId: "google" },
        { providerId: "zai", pluginId: "zai" },
      ]),
    });

    expect(registry.getError()).toContain(
      'Provider google-vertex, model gemini-3.1-pro-preview: no "api" specified',
    );
    expect(registry.find("custom", "root-model")?.name).toBe("Root Model");
    expect(registry.find("zai", "glm-5.1")?.name).toBe("GLM 5.1");
    expect(registry.find("google-vertex", "gemini-3.1-pro-preview")).toBeUndefined();
  });

  it("preserves model params from generated plugin catalog shards", () => {
    const modelsPath = writeModelsJsonWithPluginCatalog({
      root: { providers: {} },
      pluginRelativePath: join("plugins", "amazon-bedrock", PLUGIN_MODEL_CATALOG_FILE),
      pluginCatalog: {
        generatedBy: PLUGIN_MODEL_CATALOG_GENERATED_BY,
        providers: {
          "amazon-bedrock": {
            baseUrl: "https://bedrock-runtime.us-east-1.amazonaws.com",
            api: "bedrock-converse-stream",
            auth: "aws-sdk",
            models: [
              {
                id: "company-fable",
                name: "Company Fable",
                params: { canonicalModelId: "claude-fable-5" },
              },
            ],
          },
        },
      },
    });

    const registry = ModelRegistry.create(AuthStorage.inMemory(), modelsPath, {
      pluginMetadataSnapshot: pluginOwnerSnapshot("amazon-bedrock", "amazon-bedrock"),
    });

    expect(registry.getError()).toBeUndefined();
    expect(registry.find("amazon-bedrock", "company-fable")?.params).toEqual({
      canonicalModelId: "claude-fable-5",
    });
  });

  it("ignores non-generated plugin catalog files", () => {
    // Plugin catalog shards are codegen artifacts; hand-written lookalikes must
    // not extend the provider registry.
    const modelsPath = writeModelsJsonWithPluginCatalog({
      root: { providers: {} },
      pluginRelativePath: join("plugins", "zai", PLUGIN_MODEL_CATALOG_FILE),
      pluginCatalog: {
        providers: {
          zai: {
            baseUrl: "https://api.z.ai/api/paas/v4",
            api: "openai-completions",
            apiKey: "ZAI_API_KEY",
            models: [{ id: "glm-5.1", name: "GLM 5.1" }],
          },
        },
      },
    });

    const registry = ModelRegistry.create(
      AuthStorage.inMemory({ zai: { type: "api_key", key: "sk-test" } }),
      modelsPath,
    );

    expect(registry.getError()).toBeUndefined();
    expect(registry.find("zai", "glm-5.1")).toBeUndefined();
  });

  it("ignores generated plugin catalog providers without current ownership", () => {
    const modelsPath = writeModelsJsonWithPluginCatalog({
      root: { providers: {} },
      pluginRelativePath: join("plugins", "zai", PLUGIN_MODEL_CATALOG_FILE),
      pluginCatalog: {
        generatedBy: PLUGIN_MODEL_CATALOG_GENERATED_BY,
        providers: {
          zai: {
            baseUrl: "https://api.z.ai/api/paas/v4",
            api: "openai-completions",
            apiKey: "ZAI_API_KEY",
            models: [{ id: "glm-5.1", name: "GLM 5.1" }],
          },
        },
      },
    });

    const registry = ModelRegistry.create(
      AuthStorage.inMemory({ zai: { type: "api_key", key: "sk-test" } }),
      modelsPath,
      { pluginMetadataSnapshot: pluginOwnerSnapshot("other", "other") },
    );

    expect(registry.getError()).toBeUndefined();
    expect(registry.find("zai", "glm-5.1")).toBeUndefined();
  });

  it("ignores generated plugin catalog providers owned by disabled plugins", () => {
    const modelsPath = writeModelsJsonWithPluginCatalog({
      root: { providers: {} },
      pluginRelativePath: join("plugins", "zai", PLUGIN_MODEL_CATALOG_FILE),
      pluginCatalog: {
        generatedBy: PLUGIN_MODEL_CATALOG_GENERATED_BY,
        providers: {
          zai: {
            baseUrl: "https://api.z.ai/api/paas/v4",
            api: "openai-completions",
            apiKey: "ZAI_API_KEY",
            models: [{ id: "glm-5.1", name: "GLM 5.1" }],
          },
        },
      },
    });

    const registry = ModelRegistry.create(
      AuthStorage.inMemory({ zai: { type: "api_key", key: "sk-test" } }),
      modelsPath,
      { pluginMetadataSnapshot: pluginOwnerSnapshot("zai", "zai", false) },
    );

    expect(registry.getError()).toBeUndefined();
    expect(registry.find("zai", "glm-5.1")).toBeUndefined();
  });
});
