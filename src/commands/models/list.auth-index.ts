/** Auth availability index for `openclaw models list` rows. */
import { normalizeProviderIdForAuth } from "@openclaw/model-catalog-core/provider-id";
import type { AuthProfileStore } from "../../agents/auth-profiles/types.js";
import type { AuthProfileCredential } from "../../agents/auth-profiles/types.js";
import {
  listProviderEnvAuthLookupKeys,
  resolveProviderEnvAuthLookupMaps,
} from "../../agents/model-auth-env-vars.js";
import { type EnvApiKeyResult, resolveEnvApiKey } from "../../agents/model-auth-env.js";
import { GCP_VERTEX_CREDENTIALS_MARKER } from "../../agents/model-auth-markers.js";
import { resolveAwsSdkEnvVarName } from "../../agents/model-auth-runtime-shared.js";
import {
  hasSyntheticLocalProviderAuthConfig,
  hasUsableCustomProviderApiKey,
  isGoogleVertexBaseUrl,
  resolveUsableCustomProviderApiKey,
} from "../../agents/model-auth.js";
import {
  OPENAI_CODEX_PROVIDER_ID,
  OPENAI_PROVIDER_ID,
  openAIProviderUsesCodexRuntimeByDefault,
} from "../../agents/openai-routing.js";
import { resolveAgentModelPrimaryValue } from "../../config/model-input.js";
import type { OpenClawConfig } from "../../config/types.openclaw.js";
import type { PluginMetadataSnapshot } from "../../plugins/plugin-metadata-snapshot.types.js";
import { loadPluginRegistrySnapshotWithMetadata } from "../../plugins/plugin-registry.js";

export type ModelListAuthIndex = {
  hasProviderAuth(provider: string, modelApi?: string, baseUrl?: string): boolean;
  allowsProviderAuthAvailabilityFallback(provider: string): boolean;
};

/** Inputs used to build the auth index without re-reading process-wide state. */
export type CreateModelListAuthIndexParams = {
  cfg: OpenClawConfig;
  authStore: AuthProfileStore;
  workspaceDir?: string;
  env?: NodeJS.ProcessEnv;
  syntheticAuthProviderRefs?: readonly string[];
  metadataSnapshot?: PluginMetadataSnapshot;
};

function normalizeAuthProvider(
  provider: string,
  aliasMap: Readonly<Record<string, string>>,
): string {
  const normalized = normalizeProviderIdForAuth(provider);
  return aliasMap[normalized] ?? normalized;
}

function normalizeStoredAuthProvider(
  provider: string,
  aliasMap: Readonly<Record<string, string>>,
): string {
  const normalized = normalizeProviderIdForAuth(provider);
  if (normalized === OPENAI_CODEX_PROVIDER_ID) {
    return normalized;
  }
  return aliasMap[normalized] ?? normalized;
}

function listValidatedSyntheticAuthProviderRefs(params: {
  cfg: OpenClawConfig;
  workspaceDir?: string;
  env: NodeJS.ProcessEnv;
  metadataSnapshot?: PluginMetadataSnapshot;
}): readonly string[] {
  if (params.metadataSnapshot && (params.metadataSnapshot.registryDiagnostics?.length ?? 0) > 0) {
    return [];
  }
  const result = loadPluginRegistrySnapshotWithMetadata({
    config: params.cfg,
    workspaceDir: params.workspaceDir,
    env: params.env,
    index: params.metadataSnapshot?.index,
  });
  if (result.source !== "persisted" && result.source !== "provided") {
    return [];
  }
  return result.snapshot.plugins
    .filter((plugin) => plugin.enabled)
    .flatMap((plugin) => plugin.syntheticAuthRefs ?? []);
}

/** Builds a provider-auth lookup from profiles, env, config, and synthetic plugin refs. */
export function createModelListAuthIndex(
  params: CreateModelListAuthIndexParams,
): ModelListAuthIndex {
  const env = params.env ?? process.env;
  const lookupParams = {
    config: params.cfg,
    workspaceDir: params.workspaceDir,
    env,
    metadataSnapshot: params.metadataSnapshot,
  };
  const { aliasMap, envCandidateMap, authEvidenceMap } =
    resolveProviderEnvAuthLookupMaps(lookupParams);
  const skipSetupProviderFallback = params.metadataSnapshot !== undefined;
  const authenticatedProviders = new Set<string>();
  const syntheticAuthProviders = new Set<string>();
  const envProviderAuthResultCache = new Map<string, EnvApiKeyResult | null>();
  const credentialAuthsProvider = (credential: AuthProfileCredential): boolean => {
    const normalizedProvider = normalizeStoredAuthProvider(credential.provider, aliasMap);
    if (normalizedProvider !== OPENAI_PROVIDER_ID) {
      return true;
    }
    if (credential.type === "api_key") {
      return true;
    }
    if (credential.type !== "oauth" && credential.type !== "token") {
      return false;
    }
    // OpenAI OAuth/token profiles only authenticate provider rows when config
    // routes OpenAI through Codex runtime semantics.
    return openAIProviderUsesCodexRuntimeByDefault({
      provider: normalizedProvider,
      config: params.cfg,
    });
  };
  const addProvider = (provider: string | undefined) => {
    if (!provider?.trim()) {
      return;
    }
    authenticatedProviders.add(normalizeStoredAuthProvider(provider, aliasMap));
  };
  const addSyntheticProvider = (provider: string | undefined) => {
    const normalized = provider?.trim() ? normalizeProviderIdForAuth(provider) : "";
    if (!normalized) {
      return;
    }
    syntheticAuthProviders.add(normalized);
  };

  for (const credential of Object.values(params.authStore.profiles ?? {})) {
    if (credentialAuthsProvider(credential)) {
      addProvider(credential.provider);
    }
  }

  for (const provider of listProviderEnvAuthLookupKeys({ envCandidateMap, authEvidenceMap })) {
    const envResult = resolveEnvApiKey(provider, env, {
      aliasMap,
      candidateMap: envCandidateMap,
      authEvidenceMap,
      skipSetupProviderFallback,
      config: params.cfg,
      workspaceDir: params.workspaceDir,
    });
    // A Vertex ADC marker authenticates only the google-vertex transport, so keep
    // it out of the coarse provider-auth set; otherwise it would mark a non-Vertex
    // (e.g. Gemini) model row available. The Vertex path checks the marker
    // separately via hasGoogleVertexModelAuth.
    if (envResult && envResult.apiKey !== GCP_VERTEX_CREDENTIALS_MARKER) {
      addProvider(provider);
    }
  }

  if (resolveAwsSdkEnvVarName(env)) {
    addProvider("amazon-bedrock");
  }

  for (const provider of Object.keys(params.cfg.models?.providers ?? {})) {
    // A config-provided Vertex ADC marker is usable only by the google-vertex
    // transport; exclude it from the coarse provider-auth set so it cannot satisfy
    // a non-Vertex model row (the Vertex path scopes the marker separately).
    const configKey = resolveUsableCustomProviderApiKey({ cfg: params.cfg, provider, env });
    const hasNonMarkerConfigKey = Boolean(
      configKey && configKey.apiKey !== GCP_VERTEX_CREDENTIALS_MARKER,
    );
    if (
      hasNonMarkerConfigKey ||
      hasSyntheticLocalProviderAuthConfig({ cfg: params.cfg, provider })
    ) {
      addProvider(provider);
    }
  }
  const primaryModelProvider = resolveAgentModelPrimaryValue(
    params.cfg.agents?.defaults?.model,
  )?.split("/", 1)[0];
  if (primaryModelProvider === "codex") {
    // A Codex primary model is a synthetic provider auth signal even when no
    // normal provider key exists in the profile store.
    addSyntheticProvider("codex");
  }

  for (const provider of params.syntheticAuthProviderRefs ??
    listValidatedSyntheticAuthProviderRefs({
      cfg: params.cfg,
      workspaceDir: params.workspaceDir,
      env,
      metadataSnapshot: params.metadataSnapshot,
    })) {
    addSyntheticProvider(provider);
  }

  const resolveEnvProviderAuth = (provider: string): EnvApiKeyResult | null => {
    const normalized = normalizeAuthProvider(provider, aliasMap);
    const cached = envProviderAuthResultCache.get(normalized);
    if (cached !== undefined) {
      return cached;
    }
    const hasPrecomputedCandidates = Object.hasOwn(envCandidateMap, normalized);
    const hasPrecomputedEvidence = Object.hasOwn(authEvidenceMap, normalized);
    const result = resolveEnvApiKey(provider, env, {
      aliasMap,
      candidateMap:
        skipSetupProviderFallback || hasPrecomputedCandidates ? envCandidateMap : undefined,
      authEvidenceMap:
        skipSetupProviderFallback || hasPrecomputedEvidence ? authEvidenceMap : undefined,
      skipSetupProviderFallback,
      config: params.cfg,
      workspaceDir: params.workspaceDir,
    });
    envProviderAuthResultCache.set(normalized, result);
    // Only a non-marker env credential joins the coarse provider-auth set; the
    // Vertex ADC marker is reserved for the model-scoped Vertex path so it cannot
    // mark a non-Vertex model row available.
    if (result && result.apiKey !== GCP_VERTEX_CREDENTIALS_MARKER) {
      authenticatedProviders.add(normalized);
    }
    return result;
  };
  // Unscoped env auth (includes the Vertex ADC marker): used by the Vertex path.
  const hasEnvProviderAuth = (provider: string): boolean =>
    Boolean(resolveEnvProviderAuth(provider));
  // Env auth excluding the Vertex ADC marker: used for non-Vertex model rows so a
  // Vertex-only credential never authenticates a Gemini (or other) provider row.
  const hasNonVertexMarkerEnvProviderAuth = (provider: string): boolean => {
    const result = resolveEnvProviderAuth(provider);
    return Boolean(result && result.apiKey !== GCP_VERTEX_CREDENTIALS_MARKER);
  };

  const hasOpenAICodexRuntimeAuth = (provider: string): boolean => {
    const normalizedProvider = normalizeAuthProvider(provider, aliasMap);
    return (
      openAIProviderUsesCodexRuntimeByDefault({
        provider: normalizedProvider,
        config: params.cfg,
      }) &&
      (authenticatedProviders.has(OPENAI_PROVIDER_ID) ||
        authenticatedProviders.has(OPENAI_CODEX_PROVIDER_ID))
    );
  };

  const genericGoogleProvider = normalizeAuthProvider("google", aliasMap);
  const vertexAuthProvider = normalizeAuthProvider("google-vertex", aliasMap);

  // A Vertex model can be registered under a non-Vertex provider id (e.g. the shared
  // "google" provider) yet resolve auth via the dedicated "google-vertex" provider.
  // Mirror the runtime model-scoped resolution so availability matches dispatch: a
  // Vertex request is satisfied only by Vertex-valid credentials, never by the shared
  // "google" provider's AI Studio (Gemini) keys, which the runtime resolver rejects
  // for the Vertex transport.
  const hasGoogleVertexModelAuth = (provider: string, normalizedProvider: string): boolean => {
    // Config ADC marker / Vertex Express key scoped to a Vertex request, on the
    // requested provider or on the dedicated "google-vertex" provider entry. The
    // latter is excluded from the coarse authenticatedProviders set (its only
    // credential is the non-secret ADC marker), but runtime resolveApiKeyForProvider
    // falls back to the configured "google-vertex" auth, so availability must too.
    if (
      hasUsableCustomProviderApiKey(params.cfg, provider, env, "google-vertex") ||
      hasUsableCustomProviderApiKey(params.cfg, "google-vertex", env, "google-vertex")
    ) {
      return true;
    }
    // The dedicated "google-vertex" auth provider (ADC marker / google-vertex env evidence).
    if (authenticatedProviders.has(vertexAuthProvider) || hasEnvProviderAuth("google-vertex")) {
      return true;
    }
    // A custom/dedicated (non-generic-"google") provider id carries its own Vertex
    // credential, so its direct auth counts. The shared "google" provider's own auth
    // is intentionally excluded above.
    if (normalizedProvider !== genericGoogleProvider && normalizedProvider !== vertexAuthProvider) {
      return (
        authenticatedProviders.has(normalizedProvider) ||
        syntheticAuthProviders.has(normalizeProviderIdForAuth(provider)) ||
        hasEnvProviderAuth(provider)
      );
    }
    return false;
  };

  return {
    hasProviderAuth(provider: string, modelApi?: string, baseUrl?: string): boolean {
      const normalizedProvider = normalizeAuthProvider(provider, aliasMap);
      // Gemini-on-Vertex routing: a per-model api "google-vertex", or a
      // google-generative-ai model routed through the Vertex stream by an aiplatform
      // base URL. Both resolve the cross-provider google-vertex fallback at runtime
      // (the dedicated provider's env/config/ADC), even when registered under the
      // shared "google" id, and exclude that provider's AI Studio (Gemini) keys. The
      // base-URL shortcut is gated on the Gemini api so an OpenAI-compatible Vertex
      // endpoint (openai-completions on an aiplatform host) is not misrouted here.
      if (
        modelApi === "google-vertex" ||
        (modelApi === "google-generative-ai" && isGoogleVertexBaseUrl(baseUrl))
      ) {
        return hasGoogleVertexModelAuth(provider, normalizedProvider);
      }
      // The shared GCP ADC marker authorizes Vertex transports only, so it must not
      // satisfy a Gemini/AI-Studio or other non-Vertex row. It IS, however, a valid
      // credential for a non-Gemini Vertex api such as Anthropic Vertex
      // (anthropic-messages), whose runtime transport exchanges it — so keep the
      // marker for those rows and strip it only for rows that cannot use it.
      const rowAcceptsVertexAdcMarker = modelApi === "anthropic-messages";
      const hasDirectAuth =
        authenticatedProviders.has(normalizedProvider) ||
        syntheticAuthProviders.has(normalizeProviderIdForAuth(provider)) ||
        (rowAcceptsVertexAdcMarker
          ? hasEnvProviderAuth(provider)
          : hasNonVertexMarkerEnvProviderAuth(provider));
      if (hasDirectAuth) {
        return true;
      }
      return hasOpenAICodexRuntimeAuth(normalizedProvider);
    },
    allowsProviderAuthAvailabilityFallback(provider: string): boolean {
      return hasOpenAICodexRuntimeAuth(provider);
    },
  };
}
