/**
 * Resolves model-provider credentials from config, env, auth profiles, and
 * provider synthetic auth hooks. This module is the shared auth boundary for
 * runtime dispatch, setup checks, and model metadata reporting.
 */
import path from "node:path";
import {
  normalizeLowercaseStringOrEmpty,
  normalizeOptionalLowercaseString,
} from "@openclaw/normalization-core/string-coerce";
import { normalizeUniqueStringEntries } from "@openclaw/normalization-core/string-normalization";
import { formatCliCommand } from "../cli/command-format.js";
import {
  getRuntimeConfigSnapshot,
  getRuntimeConfigSourceSnapshot,
  selectApplicableRuntimeConfig,
} from "../config/config.js";
import type { ModelProviderAuthMode, ModelProviderConfig } from "../config/types.js";
import type { OpenClawConfig } from "../config/types.openclaw.js";
import { coerceSecretRef } from "../config/types.secrets.js";
import { formatErrorMessage } from "../infra/errors.js";
import { getShellEnvAppliedKeys } from "../infra/shell-env.js";
import type { Model } from "../llm/types.js";
import { createSubsystemLogger } from "../logging/subsystem.js";
import {
  buildProviderMissingAuthMessageWithPlugin,
  resolveProviderSyntheticAuthWithPlugin,
  shouldDeferProviderSyntheticProfileAuthWithPlugin,
} from "../plugins/provider-runtime.js";
import { resolveOwningPluginIdsForProviderRef } from "../plugins/providers.js";
import { resolveRuntimeSyntheticAuthProviderRefState } from "../plugins/synthetic-auth.runtime.js";
import { resolveDefaultSecretProviderAlias } from "../secrets/ref-contract.js";
import { normalizeOptionalSecretInput } from "../utils/normalize-secret-input.js";
import { resolveDefaultAgentDir } from "./agent-scope-config.js";
import {
  type AuthProfileCredential,
  type AuthProfileStore,
  externalCliDiscoveryForProviderAuth,
  ensureAuthProfileStore,
  isConfiguredAwsSdkAuthProfileForProvider,
  isStoredCredentialCompatibleWithAuthProvider,
  listProfilesForProvider,
  resolveApiKeyForProfile,
  resolveAuthProfileOrder,
  resolveAuthStorePathForDisplay,
} from "./auth-profiles.js";
import * as cliCredentials from "./cli-credentials.js";
import { resolveProviderEnvAuthLookupMaps } from "./model-auth-env-vars.js";
import {
  resolveEnvApiKey,
  type EnvApiKeyLookupOptions,
  type EnvApiKeyResult,
} from "./model-auth-env.js";
import {
  CUSTOM_LOCAL_AUTH_MARKER,
  GCP_VERTEX_CREDENTIALS_MARKER,
  isKnownEnvApiKeyMarker,
  isNonSecretApiKeyMarker,
  NON_ENV_SECRETREF_MARKER,
} from "./model-auth-markers.js";
import { ProviderAuthError, type ResolvedProviderAuth } from "./model-auth-runtime-shared.js";
import { normalizeProviderId } from "./model-selection.js";

export {
  ensureAuthProfileStore,
  ensureAuthProfileStoreWithoutExternalProfiles,
  resolveAuthProfileOrder,
} from "./auth-profiles.js";
export {
  formatMissingAuthError,
  isMissingProviderAuthError,
  isProviderAuthError,
  MissingProviderAuthError,
  ProviderAuthError,
  requireApiKey,
  resolveAwsSdkEnvVarName,
} from "./model-auth-runtime-shared.js";
export type { ResolvedProviderAuth } from "./model-auth-runtime-shared.js";
export type ProviderCredentialPrecedence = "profile-first" | "env-first";

/** Precomputed provider-auth lookup tables reused during one runtime turn. */
export type RuntimeProviderAuthLookup = {
  envApiKey: Pick<
    EnvApiKeyLookupOptions,
    "aliasMap" | "candidateMap" | "authEvidenceMap" | "skipSetupProviderFallback"
  >;
  setupProviderFallbackRefs?: readonly string[];
  syntheticAuthProviderRefs?: readonly string[];
  syntheticAuthProviderRefsComplete?: boolean;
};

const log = createSubsystemLogger("model-auth");
const OPENAI_PROVIDER_ID = "openai";
const OPENAI_CODEX_RESPONSES_API = "openai-chatgpt-responses";

function directOpenAIPlatformModelRequiresApiKey(params: {
  provider: string;
  modelApi?: string;
}): boolean {
  return (
    normalizeProviderId(params.provider) === OPENAI_PROVIDER_ID &&
    params.modelApi !== undefined &&
    normalizeLowercaseStringOrEmpty(params.modelApi) !== OPENAI_CODEX_RESPONSES_API
  );
}

function isAuthModeAllowedForModel(params: {
  provider: string;
  modelApi?: string;
  mode: ResolvedProviderAuth["mode"];
}): boolean {
  return !directOpenAIPlatformModelRequiresApiKey(params) || params.mode === "api-key";
}

function assertAuthModeAllowedForModel(params: {
  provider: string;
  modelApi?: string;
  profileId: string;
  mode: ResolvedProviderAuth["mode"];
}): void {
  if (isAuthModeAllowedForModel(params)) {
    return;
  }
  throw new Error(
    `Auth profile "${params.profileId}" uses ${params.mode} auth, but ${params.provider}/${params.modelApi} requires an OpenAI API key profile.`,
  );
}

function resolveConfigAwareEnvApiKey(
  cfg: OpenClawConfig | undefined,
  provider: string,
  workspaceDir?: string,
): EnvApiKeyResult | null {
  return resolveEnvApiKey(provider, process.env, { config: cfg, workspaceDir });
}

function resolveProviderConfig(
  cfg: OpenClawConfig | undefined,
  provider: string,
): ModelProviderConfig | undefined {
  const providers = cfg?.models?.providers ?? {};
  const direct = providers[provider] as ModelProviderConfig | undefined;
  if (direct) {
    return direct;
  }
  const normalized = normalizeProviderId(provider);
  if (normalized === provider) {
    const matched = Object.entries(providers).find(
      ([key]) => normalizeProviderId(key) === normalized,
    );
    return matched?.[1];
  }
  return (
    (providers[normalized] as ModelProviderConfig | undefined) ??
    Object.entries(providers).find(([key]) => normalizeProviderId(key) === normalized)?.[1]
  );
}

/** Builds stable env/synthetic auth lookup data for repeated provider checks. */
export function createRuntimeProviderAuthLookup(params: {
  cfg?: OpenClawConfig;
  workspaceDir?: string;
  env?: NodeJS.ProcessEnv;
  includePluginSyntheticAuth?: boolean;
}): RuntimeProviderAuthLookup {
  const env = params.env ?? process.env;
  const lookupParams = {
    config: params.cfg,
    workspaceDir: params.workspaceDir,
    env,
  };
  const syntheticAuthProviderRefs =
    params.includePluginSyntheticAuth === false
      ? undefined
      : resolveRuntimeSyntheticAuthProviderRefState(lookupParams);
  const authLookupMaps = resolveProviderEnvAuthLookupMaps(lookupParams);
  return {
    envApiKey: {
      aliasMap: authLookupMaps.aliasMap,
      candidateMap: authLookupMaps.envCandidateMap,
      authEvidenceMap: authLookupMaps.authEvidenceMap,
      skipSetupProviderFallback: true,
    },
    setupProviderFallbackRefs: authLookupMaps.setupProviderFallbackRefs,
    syntheticAuthProviderRefs: syntheticAuthProviderRefs?.complete
      ? syntheticAuthProviderRefs.refs
      : undefined,
    syntheticAuthProviderRefsComplete: syntheticAuthProviderRefs?.complete,
  };
}

function runtimeLookupAllowsSetupProviderFallback(params: {
  provider: string;
  runtimeLookup?: RuntimeProviderAuthLookup;
}): boolean {
  const refs = params.runtimeLookup?.setupProviderFallbackRefs;
  if (!refs?.length) {
    return false;
  }
  const normalizedProvider = normalizeProviderId(params.provider);
  const aliasTarget = params.runtimeLookup?.envApiKey.aliasMap?.[normalizedProvider];
  return refs.includes(normalizedProvider) || (aliasTarget ? refs.includes(aliasTarget) : false);
}

function resolveRuntimeEnvApiKeyLookupOptions(params: {
  provider: string;
  runtimeLookup?: RuntimeProviderAuthLookup;
}):
  | Pick<
      EnvApiKeyLookupOptions,
      "aliasMap" | "candidateMap" | "authEvidenceMap" | "skipSetupProviderFallback"
    >
  | undefined {
  const envApiKey = params.runtimeLookup?.envApiKey;
  if (!envApiKey) {
    return undefined;
  }
  const skipSetupProviderFallback =
    envApiKey.skipSetupProviderFallback === true
      ? !runtimeLookupAllowsSetupProviderFallback(params)
      : envApiKey.skipSetupProviderFallback;
  return {
    ...envApiKey,
    ...(skipSetupProviderFallback !== undefined ? { skipSetupProviderFallback } : {}),
  };
}

/** Reads a literal or env-secret marker for a custom provider entry. */
export function getCustomProviderApiKey(
  cfg: OpenClawConfig | undefined,
  provider: string,
): string | undefined {
  const entry = resolveProviderConfig(cfg, provider);
  const literal = normalizeOptionalSecretInput(entry?.apiKey);
  if (literal) {
    return literal;
  }
  const ref = coerceSecretRef(entry?.apiKey);
  if (!ref) {
    return undefined;
  }
  if (ref.source === "env") {
    const envId = ref.id.trim();
    return envId || NON_ENV_SECRETREF_MARKER;
  }
  return NON_ENV_SECRETREF_MARKER;
}

type ResolvedCustomProviderApiKey = {
  apiKey: string;
  source: string;
};

function canResolveEnvSecretRefInReadOnlyPath(params: {
  cfg: OpenClawConfig | undefined;
  provider: string;
  id: string;
}): boolean {
  const providerConfig = params.cfg?.secrets?.providers?.[params.provider];
  if (!providerConfig) {
    return params.provider === resolveDefaultSecretProviderAlias(params.cfg ?? {}, "env");
  }
  if (providerConfig.source !== "env") {
    return false;
  }
  const allowlist = providerConfig.allowlist;
  return !allowlist || allowlist.includes(params.id);
}

/**
 * True when resolving auth for a Google Vertex AI request. The non-secret Vertex
 * ADC marker authorizes the Vertex transport only, so it must never reach a
 * non-Vertex Google client (the Gemini API sends it as `x-goog-api-key`).
 *
 * Maestro and other deployments register Vertex under the "google" provider id
 * while marking the Vertex API per-model (api: "google-vertex"), so the same
 * provider id can serve both Vertex and non-Vertex requests. The match is
 * therefore:
 *   - when the request model api is known, it is authoritative: only
 *     "google-vertex" qualifies (a per-model override outranks the provider
 *     default, and a non-Vertex model never qualifies);
 *   - when no model api is known (provider-level SDK callers such as image/
 *     video/music generation resolve `provider: "google"` without a model api
 *     and then send a Gemini request), the marker only applies when the provider
 *     is explicitly declared Vertex at the provider level (provider-level api
 *     "google-vertex") AND it is a dedicated Vertex provider id, not the generic
 *     "google" id those Gemini SDKs share. A bare per-model "google-vertex" entry
 *     (or a provider-level Vertex declaration on the shared "google" id) is
 *     intentionally not enough for model-agnostic callers, so the marker cannot
 *     leak into Gemini-API clients.
 */
function configTargetsGoogleVertex(
  providerConfig: ModelProviderConfig | undefined,
  provider: string,
  modelApi?: string,
  baseUrl?: string,
): boolean {
  if (!providerConfig) {
    return false;
  }
  if (modelApi !== undefined) {
    // A per-model api "google-vertex", or a google-generative-ai model routed
    // through Vertex by an aiplatform base URL, both target the Vertex transport.
    // The base-URL shortcut is gated on the Gemini api so an OpenAI-compatible
    // Vertex endpoint (openai-completions on an aiplatform host) does not falsely
    // claim the marker.
    return isVertexAuthModelRequest(modelApi, baseUrl);
  }
  return providerConfig.api === "google-vertex" && normalizeProviderId(provider) !== "google";
}

/**
 * Model API surfaces whose transport exchanges the `gcp-vertex-credentials` ADC
 * marker for a real bearer token: Google Vertex (`google-vertex`) and Anthropic
 * Vertex (`anthropic-messages`, declared by the anthropic-vertex plugin). Every
 * other api — AI Studio Gemini (`google-generative-ai`), and crucially the
 * OpenAI-compatible Vertex endpoint (`openai-completions`, which lives on an
 * aiplatform host but keeps the OpenAI transport) — would send the marker as a
 * literal key, so the marker must be dropped for them.
 */
const VERTEX_ADC_TRANSPORT_MODEL_APIS: ReadonlySet<string> = new Set([
  "google-vertex",
  "anthropic-messages",
]);

/**
 * True when the selected model dispatches through a GCP Vertex transport that
 * exchanges the ADC marker. A Gemini api (`google-generative-ai`) qualifies only
 * when routed through the Vertex stream — by an aiplatform base URL or the
 * dedicated "google-vertex" provider id — never on the host/provider being merely
 * Vertex-shaped while the api keeps a non-Vertex transport (e.g. openai-completions).
 */
function modelApiUsesVertexAdcTransport(
  provider: string,
  modelApi: string | undefined,
  baseUrl: string | undefined,
): boolean {
  if (modelApi === undefined) {
    return false;
  }
  if (VERTEX_ADC_TRANSPORT_MODEL_APIS.has(modelApi)) {
    return true;
  }
  return (
    modelApi === "google-generative-ai" &&
    (isGoogleVertexBaseUrl(baseUrl) || normalizeProviderId(provider) === "google-vertex")
  );
}

// Mirrors the google extension's Vertex host detection (provider-policy.ts):
// the global host, regional `<region>-aiplatform.googleapis.com` hosts, and the
// multi-region `.rep.googleapis.com` hosts the Vertex transport also routes.
const GOOGLE_VERTEX_HOST = "aiplatform.googleapis.com";
const GOOGLE_VERTEX_REGION_HOST_SUFFIX = "-aiplatform.googleapis.com";
const GOOGLE_VERTEX_MULTI_REGION_HOSTS: ReadonlySet<string> = new Set([
  "aiplatform.eu.rep.googleapis.com",
  "aiplatform.us.rep.googleapis.com",
]);

/** True when a base URL points at a GCP Vertex AI host (global, regional, or multi-region). */
export function isGoogleVertexBaseUrl(baseUrl: string | undefined): boolean {
  const trimmed = baseUrl?.trim();
  if (!trimmed) {
    return false;
  }
  let host: string;
  try {
    host = new URL(trimmed).hostname.toLowerCase();
  } catch {
    return false;
  }
  return (
    host === GOOGLE_VERTEX_HOST ||
    host.endsWith(GOOGLE_VERTEX_REGION_HOST_SUFFIX) ||
    GOOGLE_VERTEX_MULTI_REGION_HOSTS.has(host)
  );
}

/**
 * True when the selected model resolves its credential from the dedicated
 * "google-vertex" auth provider: either a per-model api `google-vertex`, or a
 * `google-generative-ai` model routed through the Vertex stream by an
 * `aiplatform.googleapis.com` base URL. The base-URL shortcut is gated on the
 * Gemini api because the host alone is not sufficient — an OpenAI-compatible Vertex
 * endpoint (`api: openai-completions`, `.../endpoints/openapi`) lives on the same
 * host but keeps the OpenAI transport and must not borrow Vertex (Gemini) auth.
 * For these requests AI Studio keys/profiles are excluded under the shared "google"
 * id, the ADC marker is honored, and the cross-provider "google-vertex" fallback is
 * consulted. Anthropic Vertex (`anthropic-messages`) is intentionally excluded: it
 * resolves from the anthropic-vertex provider, not "google-vertex".
 */
function isVertexAuthModelRequest(
  modelApi: string | undefined,
  baseUrl: string | undefined,
): boolean {
  return (
    modelApi === "google-vertex" ||
    (modelApi === "google-generative-ai" && isGoogleVertexBaseUrl(baseUrl))
  );
}

/**
 * Drops the non-secret Vertex ADC marker from an env-resolved credential when the
 * request is a genuine Google AI Studio (Gemini) call. The marker (for example
 * from the metadata-server opt-in auth evidence) is the shared GCP ADC sentinel
 * that authorizes GCP Vertex transports only — both Google Vertex
 * (`google-vertex`) and Anthropic Vertex (`anthropic-messages`) — so it must never
 * be accepted as auth for an AI Studio (Gemini) request, which authenticates with
 * a plain `x-goog-api-key`.
 */
function scopeVertexAdcEnvMarker(
  result: EnvApiKeyResult | null,
  dropMarker: boolean,
): EnvApiKeyResult | null {
  if (dropMarker && result?.apiKey === GCP_VERTEX_CREDENTIALS_MARKER) {
    return null;
  }
  return result;
}

/**
 * Resolves an env-based API key for a provider, scoped to the selected model.
 *
 * Vertex is commonly registered under a non-Vertex provider id with the Vertex
 * API marked per-model (for example a "google-vertex" model under the "google"
 * provider). The metadata-server ADC opt-in evidence
 * (`GOOGLE_VERTEX_USE_GCP_METADATA`) is keyed under the dedicated "google-vertex"
 * auth provider, and only that provider holds credentials valid for the Vertex
 * transport (the ADC marker, or a google-vertex env key). Generic Google env keys
 * (`GEMINI_API_KEY`/`GOOGLE_API_KEY`) under the "google" provider are Gemini API
 * keys that must never be sent as a Vertex credential.
 *
 * For a Vertex request (`modelApi === "google-vertex"`) issued under a different
 * provider id, resolution depends on whether that id is the generic "google"
 * provider or a custom/dedicated one:
 * - Under the generic "google" provider the "google-vertex" auth provider is
 *   consulted first so a present Gemini env key cannot shadow the ADC evidence,
 *   and on fallback only "google"'s own non-secret ADC marker is accepted (its
 *   `GEMINI_API_KEY`/`GOOGLE_API_KEY` are Gemini credentials invalid for Vertex),
 *   so the transport is reported unavailable until real Vertex auth is configured
 *   rather than being handed an AI Studio key.
 * - Under a custom/dedicated provider id its own declared env credential (a
 *   provider-specific Vertex API key or ADC marker) is preferred, falling back to
 *   the built-in "google-vertex" auth provider only when the requested provider
 *   declares none; the global credential never overrides a provider-specific one.
 * Non-Vertex and model-agnostic callers never trigger this lookup or receive the
 * marker.
 */
function resolveModelScopedEnvApiKey(params: {
  provider: string;
  modelApi: string | undefined;
  /** Request base URL, used to detect Vertex routing for AI Studio model apis. */
  baseUrl?: string;
  resolve: (provider: string) => EnvApiKeyResult | null;
}): EnvApiKeyResult | null {
  const normalizedProvider = normalizeProviderId(params.provider);
  // The shared GCP ADC marker is kept only for an api+route that dispatches through
  // a Vertex transport able to exchange it (Google Vertex, Anthropic Vertex, or a
  // Gemini api routed through the Vertex stream). It is dropped for any other known
  // api — a genuine AI Studio (Gemini) request, and crucially the OpenAI-compatible
  // Vertex endpoint (openai-completions on an aiplatform host), whose transport
  // would otherwise send the marker as a literal bearer key. Model-agnostic callers
  // (no api) never receive the marker here regardless.
  const dropAdcMarker =
    params.modelApi !== undefined &&
    !modelApiUsesVertexAdcTransport(params.provider, params.modelApi, params.baseUrl);
  if (
    isVertexAuthModelRequest(params.modelApi, params.baseUrl) &&
    normalizedProvider !== "google-vertex"
  ) {
    // The generic "google" provider's env keys (GEMINI_API_KEY/GOOGLE_API_KEY) are
    // Gemini credentials that must never be sent as a Vertex credential, so the
    // built-in "google-vertex" auth provider is consulted first (a present Gemini
    // key cannot shadow the ADC evidence) and only "google"'s own ADC marker is
    // accepted on fallback.
    if (normalizedProvider === "google") {
      const vertexEnv = scopeVertexAdcEnvMarker(params.resolve("google-vertex"), dropAdcMarker);
      if (vertexEnv) {
        return vertexEnv;
      }
      const providerEnv = scopeVertexAdcEnvMarker(params.resolve("google"), dropAdcMarker);
      return providerEnv?.apiKey === GCP_VERTEX_CREDENTIALS_MARKER ? providerEnv : null;
    }
    // A custom/dedicated provider id carries its own Vertex credential (a
    // provider-specific Vertex API key or ADC marker), so it is preferred over the
    // built-in "google-vertex" global credential; the global auth provider is only
    // a fallback when the requested provider declares no usable env credential.
    const providerEnv = scopeVertexAdcEnvMarker(params.resolve(params.provider), dropAdcMarker);
    if (providerEnv) {
      return providerEnv;
    }
    return scopeVertexAdcEnvMarker(params.resolve("google-vertex"), dropAdcMarker);
  }
  return scopeVertexAdcEnvMarker(params.resolve(params.provider), dropAdcMarker);
}

/** Resolves custom provider API keys that are usable without mutating secret stores. */
export function resolveUsableCustomProviderApiKey(params: {
  cfg: OpenClawConfig | undefined;
  provider: string;
  /** Selected model api, when known, to scope provider-wide auth markers. */
  modelApi?: string;
  /** Selected model base URL, used to detect Vertex routing for AI Studio apis. */
  baseUrl?: string;
  env?: NodeJS.ProcessEnv;
}): ResolvedCustomProviderApiKey | null {
  const customProviderConfig = resolveProviderConfig(params.cfg, params.provider);
  // A per-model override outranks the provider config base URL when classifying
  // the request as Vertex-routed.
  const effectiveBaseUrl = params.baseUrl ?? customProviderConfig?.baseUrl;
  // Under the generic "google" provider, the AI Studio credentials
  // GEMINI_API_KEY / GOOGLE_API_KEY are Gemini keys that are invalid for the
  // Vertex transport. For a Vertex request they must not be returned (or they
  // would be sent to Vertex as x-goog-api-key); the non-secret ADC marker is
  // honored separately below. Any other configured key under "google" -- a
  // literal value, a custom env secret-ref, or the GOOGLE_CLOUD_API_KEY Vertex
  // key marker -- is a legitimate Vertex Express credential and is preserved.
  // Dedicated Vertex provider ids are never affected.
  const isVertexRequestUnderGenericGoogle =
    isVertexAuthModelRequest(params.modelApi, effectiveBaseUrl) &&
    normalizeProviderId(params.provider) === "google";
  const isGeminiAiStudioEnvName = (name: string): boolean =>
    name === "GEMINI_API_KEY" || name === "GOOGLE_API_KEY";
  const apiKeyRef = coerceSecretRef(customProviderConfig?.apiKey);
  if (apiKeyRef) {
    if (apiKeyRef.source !== "env") {
      return null;
    }
    const envVarName = apiKeyRef.id.trim();
    if (!envVarName) {
      return null;
    }
    if (isVertexRequestUnderGenericGoogle && isGeminiAiStudioEnvName(envVarName)) {
      return null;
    }
    if (
      !canResolveEnvSecretRefInReadOnlyPath({
        cfg: params.cfg,
        provider: apiKeyRef.provider,
        id: envVarName,
      })
    ) {
      return null;
    }
    const envValue = normalizeOptionalSecretInput((params.env ?? process.env)[envVarName]);
    if (!envValue) {
      return null;
    }
    const applied = new Set(getShellEnvAppliedKeys());
    return {
      apiKey: envValue,
      source: resolveEnvSourceLabel({
        applied,
        envVars: [envVarName],
        label: `${envVarName} (models.json secretref)`,
      }),
    };
  }

  const customKey = getCustomProviderApiKey(params.cfg, params.provider);
  if (!customKey) {
    return null;
  }
  if (!isNonSecretApiKeyMarker(customKey)) {
    // A literal key is preserved: under "google" it may be a Vertex Express key,
    // which the Vertex transport accepts; only the named AI Studio env markers
    // below are excluded for Vertex requests.
    return { apiKey: customKey, source: "models.json" };
  }
  if (isKnownEnvApiKeyMarker(customKey)) {
    if (isVertexRequestUnderGenericGoogle && isGeminiAiStudioEnvName(customKey)) {
      return null;
    }
    const envValue = normalizeOptionalSecretInput((params.env ?? process.env)[customKey]);
    if (!envValue) {
      return null;
    }
    const applied = new Set(getShellEnvAppliedKeys());
    return {
      apiKey: envValue,
      source: resolveEnvSourceLabel({
        applied,
        envVars: [customKey],
        label: `${customKey} (models.json marker)`,
      }),
    };
  }
  if (
    customKey === GCP_VERTEX_CREDENTIALS_MARKER &&
    configTargetsGoogleVertex(
      customProviderConfig,
      params.provider,
      params.modelApi,
      effectiveBaseUrl,
    )
  ) {
    // The Vertex ADC marker is intentionally a non-secret marker: it signals "resolve
    // credentials via Application Default Credentials (metadata server / gcloud / ADC
    // file)" rather than carrying a literal key. The per-agent resolver otherwise drops
    // it, so deployments that register Vertex under the "google" provider id (with
    // per-model api "google-vertex") fail with missing-provider-auth even though the
    // Vertex transport resolves ADC at request time. Pass the marker through so the
    // transport performs the real ADC token exchange.
    return {
      apiKey: GCP_VERTEX_CREDENTIALS_MARKER,
      source: "models.json (vertex adc marker)",
    };
  }
  if (
    customProviderConfig &&
    isCustomLocalProviderConfig(customProviderConfig) &&
    (customProviderConfig.api === "openai-completions" || customProviderConfig.api === "ollama") &&
    customProviderConfig.baseUrl &&
    isLocalBaseUrl(customProviderConfig.baseUrl)
  ) {
    return {
      apiKey: customProviderConfig.api === "ollama" ? customKey : CUSTOM_LOCAL_AUTH_MARKER,
      source: "models.json (local marker)",
    };
  }
  return null;
}

/** True when a custom provider has a literal/env/local key available now. */
export function hasUsableCustomProviderApiKey(
  cfg: OpenClawConfig | undefined,
  provider: string,
  env?: NodeJS.ProcessEnv,
  modelApi?: string,
  baseUrl?: string,
): boolean {
  return Boolean(resolveUsableCustomProviderApiKey({ cfg, provider, env, modelApi, baseUrl }));
}

/** True when explicit provider config should outrank profile/environment auth. */
export function shouldPreferExplicitConfigApiKeyAuth(
  cfg: OpenClawConfig | undefined,
  provider: string,
): boolean {
  const providerConfig = resolveProviderConfig(cfg, provider);
  return (
    resolveProviderAuthOverride(cfg, provider) === "api-key" &&
    providerConfig !== undefined &&
    hasExplicitProviderApiKeyConfig(providerConfig)
  );
}

function resolveProviderAuthOverride(
  cfg: OpenClawConfig | undefined,
  provider: string,
): ModelProviderAuthMode | undefined {
  const entry = resolveProviderConfig(cfg, provider);
  const auth = entry?.auth;
  if (auth === "api-key" || auth === "aws-sdk" || auth === "oauth" || auth === "token") {
    return auth;
  }
  return undefined;
}

function shouldUseImplicitAwsSdkAuth(params: {
  cfg: OpenClawConfig | undefined;
  provider: string;
  modelApi: string | undefined;
}): boolean {
  if (params.modelApi !== "bedrock-converse-stream") {
    return false;
  }
  if (normalizeProviderId(params.provider) !== "amazon-bedrock") {
    return false;
  }
  const providerConfig = resolveProviderConfig(params.cfg, params.provider);
  return (
    resolveProviderAuthOverride(params.cfg, params.provider) === undefined &&
    (providerConfig === undefined || !hasExplicitProviderApiKeyConfig(providerConfig))
  );
}

function profileTypeToAuthMode(type: AuthProfileCredential["type"]): ResolvedProviderAuth["mode"] {
  return type === "oauth" ? "oauth" : type === "token" ? "token" : "api-key";
}

type ProviderEntryApiKeyProfileReference =
  | { kind: "none" }
  | { kind: "literal"; apiKey: string; source: string }
  | {
      kind: "profile";
      profileId: string;
      credential: AuthProfileCredential;
      mode: ResolvedProviderAuth["mode"];
    }
  | {
      kind: "profile-incompatible";
      profileId: string;
      credentialProvider: string;
      credentialType: AuthProfileCredential["type"];
      reason: "credential-class" | "provider-binding";
    }
  | { kind: "marker" };

export type ProviderEntryApiKeyBindingResolution =
  | { kind: "none" }
  | { kind: "literal"; apiKey: string; source: string }
  | { kind: "profile-resolved"; auth: ResolvedProviderAuth }
  | {
      kind: "profile-incompatible";
      profileId: string;
      credentialProvider: string;
      credentialType: AuthProfileCredential["type"];
      reason: "credential-class" | "provider-binding";
    }
  | { kind: "profile-unresolved"; profileId: string; error?: unknown };

function normalizeProviderEntryBaseUrlForBinding(baseUrl: string | undefined): string | undefined {
  const trimmed = baseUrl?.trim();
  if (!trimmed) {
    return undefined;
  }
  try {
    const parsed = new URL(trimmed);
    parsed.hash = "";
    parsed.search = "";
    parsed.pathname = parsed.pathname.replace(/\/+$/, "");
    return parsed.toString().replace(/\/+$/, "");
  } catch {
    return trimmed.toLowerCase().replace(/\/+$/, "");
  }
}

function providerEntriesShareBaseUrl(params: {
  cfg?: OpenClawConfig;
  provider: string;
  credentialProvider: string;
}): boolean {
  const providerBaseUrl = normalizeProviderEntryBaseUrlForBinding(
    resolveProviderConfig(params.cfg, params.provider)?.baseUrl,
  );
  const credentialProviderBaseUrl = normalizeProviderEntryBaseUrlForBinding(
    resolveProviderConfig(params.cfg, params.credentialProvider)?.baseUrl,
  );
  return Boolean(
    providerBaseUrl && credentialProviderBaseUrl && providerBaseUrl === credentialProviderBaseUrl,
  );
}

function isBearerProfileCredential(credential: AuthProfileCredential): boolean {
  return credential.type === "api_key" || credential.type === "token";
}

/** True when a bearer auth profile can safely satisfy a provider-entry apiKey reference. */
export function canUseProfileAsProviderEntryApiKey(params: {
  cfg?: OpenClawConfig;
  provider: string;
  credential: AuthProfileCredential;
}): boolean {
  if (!isBearerProfileCredential(params.credential)) {
    return false;
  }
  if (
    isStoredCredentialCompatibleWithAuthProvider({
      cfg: params.cfg,
      provider: params.provider,
      credential: params.credential,
    })
  ) {
    return true;
  }
  // Split-provider entries may intentionally point at the same upstream endpoint
  // with different profile ids. Require a matching configured base URL before
  // allowing a bearer profile to cross provider ids.
  return providerEntriesShareBaseUrl({
    cfg: params.cfg,
    provider: params.provider,
    credentialProvider: params.credential.provider,
  });
}

/** Classifies a provider entry apiKey as literal/profile/marker before resolving secrets. */
export function resolveProviderEntryApiKeyProfileReference(params: {
  cfg?: OpenClawConfig;
  provider: string;
  store: AuthProfileStore;
}): ProviderEntryApiKeyProfileReference {
  const providerConfig = resolveProviderConfig(params.cfg, params.provider);
  if (coerceSecretRef(providerConfig?.apiKey)) {
    return { kind: "none" };
  }
  const perEntryRawKey = normalizeOptionalSecretInput(providerConfig?.apiKey);
  if (!perEntryRawKey) {
    return { kind: "none" };
  }
  if (isNonSecretApiKeyMarker(perEntryRawKey)) {
    return { kind: "marker" };
  }
  const credential = params.store.profiles[perEntryRawKey];
  if (!credential) {
    return { kind: "literal", apiKey: perEntryRawKey, source: "models.json" };
  }
  if (!isBearerProfileCredential(credential)) {
    return {
      kind: "profile-incompatible",
      profileId: perEntryRawKey,
      credentialProvider: credential.provider,
      credentialType: credential.type,
      reason: "credential-class",
    };
  }
  if (
    !canUseProfileAsProviderEntryApiKey({ cfg: params.cfg, provider: params.provider, credential })
  ) {
    return {
      kind: "profile-incompatible",
      profileId: perEntryRawKey,
      credentialProvider: credential.provider,
      credentialType: credential.type,
      reason: "provider-binding",
    };
  }
  return {
    kind: "profile",
    profileId: perEntryRawKey,
    credential,
    mode: profileTypeToAuthMode(credential.type),
  };
}

/** Resolves a provider-entry apiKey profile reference into runtime auth when possible. */
export async function resolveProviderEntryApiKeyBinding(params: {
  cfg?: OpenClawConfig;
  provider: string;
  store: AuthProfileStore;
  agentDir?: string;
}): Promise<ProviderEntryApiKeyBindingResolution> {
  const reference = resolveProviderEntryApiKeyProfileReference(params);
  if (reference.kind === "none" || reference.kind === "marker") {
    return { kind: "none" };
  }
  if (reference.kind === "literal") {
    return reference;
  }
  if (reference.kind === "profile-incompatible") {
    return reference;
  }
  try {
    const resolved = await resolveApiKeyForProfile({
      cfg: params.cfg,
      store: params.store,
      profileId: reference.profileId,
      agentDir: params.agentDir,
    });
    if (!resolved) {
      return { kind: "profile-unresolved", profileId: reference.profileId };
    }
    const resolvedProfileId = resolved.profileId ?? reference.profileId;
    return {
      kind: "profile-resolved",
      auth: {
        apiKey: resolved.apiKey,
        profileId: resolvedProfileId,
        source: `profile:${resolvedProfileId}`,
        mode: resolved.profileType ? profileTypeToAuthMode(resolved.profileType) : reference.mode,
      },
    };
  } catch (err) {
    return { kind: "profile-unresolved", profileId: reference.profileId, error: err };
  }
}

function resolveConfiguredAwsSdkProfileAuth(params: {
  cfg?: OpenClawConfig;
  provider: string;
  profileId: string;
}): ResolvedProviderAuth | null {
  if (!isConfiguredAwsSdkAuthProfileForProvider(params)) {
    return null;
  }
  return {
    ...resolveAwsSdkAuthInfo(),
    profileId: params.profileId,
    source: `profile:${params.profileId}`,
  };
}

function isLocalBaseUrl(baseUrl: string): boolean {
  try {
    let host = normalizeLowercaseStringOrEmpty(new URL(baseUrl).hostname);
    if (host.startsWith("[") && host.endsWith("]")) {
      host = host.slice(1, -1);
    }
    return (
      host === "localhost" ||
      host === "127.0.0.1" ||
      host === "0.0.0.0" ||
      host === "::1" ||
      host === "::ffff:7f00:1" ||
      host === "::ffff:127.0.0.1" ||
      host === "docker.orb.internal" ||
      host === "host.docker.internal" ||
      host === "host.orb.internal" ||
      host.endsWith(".local") ||
      isPrivateIpv4Host(host)
    );
  } catch {
    return false;
  }
}

function isPrivateIpv4Host(host: string): boolean {
  if (!/^\d+\.\d+\.\d+\.\d+$/.test(host)) {
    return false;
  }
  const octets = host.split(".").map((part) => Number.parseInt(part, 10));
  if (octets.some((part) => !Number.isInteger(part) || part < 0 || part > 255)) {
    return false;
  }
  const [a, b] = octets;
  return a === 10 || (a === 172 && b >= 16 && b <= 31) || (a === 192 && b === 168);
}

function hasExplicitProviderApiKeyConfig(providerConfig: ModelProviderConfig): boolean {
  return (
    normalizeOptionalSecretInput(providerConfig.apiKey) !== undefined ||
    coerceSecretRef(providerConfig.apiKey) !== null
  );
}

function isCustomLocalProviderConfig(providerConfig: ModelProviderConfig): boolean {
  return (
    typeof providerConfig.baseUrl === "string" &&
    providerConfig.baseUrl.trim().length > 0 &&
    typeof providerConfig.api === "string" &&
    providerConfig.api.trim().length > 0 &&
    Array.isArray(providerConfig.models) &&
    providerConfig.models.length > 0
  );
}

function isManagedSecretRefApiKeyMarker(apiKey: string | undefined): boolean {
  return apiKey?.trim() === NON_ENV_SECRETREF_MARKER;
}

function hasManagedSecretRefProviderApiKey(
  cfg: OpenClawConfig | undefined,
  provider: string,
): boolean {
  const apiKey = resolveProviderConfig(cfg, provider)?.apiKey;
  const ref = coerceSecretRef(apiKey);
  if (ref) {
    return ref.source !== "env";
  }
  return typeof apiKey === "string" && isManagedSecretRefApiKeyMarker(apiKey);
}

function resolveLiteralProviderConfigApiKeyAuth(params: {
  cfg: OpenClawConfig | undefined;
  provider: string;
}): ResolvedProviderAuth | undefined {
  const apiKey = normalizeOptionalSecretInput(
    resolveProviderConfig(params.cfg, params.provider)?.apiKey,
  );
  if (!apiKey || isNonSecretApiKeyMarker(apiKey)) {
    return undefined;
  }
  return {
    apiKey,
    source: `models.providers.${params.provider}`,
    mode: "api-key",
  };
}

function resolveManagedSecretRefRuntimeProviderAuth(params: {
  cfg: OpenClawConfig | undefined;
  provider: string;
}): ResolvedProviderAuth | undefined {
  if (!hasManagedSecretRefProviderApiKey(params.cfg, params.provider)) {
    return undefined;
  }
  const runtimeConfig = getRuntimeConfigSnapshot();
  const runtimeSourceConfig = getRuntimeConfigSourceSnapshot();
  if (params.cfg && params.cfg !== runtimeConfig && !runtimeSourceConfig) {
    return undefined;
  }
  const applicableConfig = selectApplicableRuntimeConfig({
    inputConfig: params.cfg,
    runtimeConfig,
    runtimeSourceConfig,
  });
  if (!runtimeConfig || applicableConfig !== runtimeConfig) {
    return undefined;
  }
  return resolveLiteralProviderConfigApiKeyAuth({
    cfg: runtimeConfig,
    provider: params.provider,
  });
}

/** True when a custom local provider can use a synthetic no-auth placeholder. */
export function hasSyntheticLocalProviderAuthConfig(params: {
  cfg: OpenClawConfig | undefined;
  provider: string;
}): boolean {
  const providerConfig = resolveProviderConfig(params.cfg, params.provider);
  if (!providerConfig) {
    return false;
  }

  const hasApiConfig =
    Boolean(providerConfig.api?.trim()) ||
    Boolean(providerConfig.baseUrl?.trim()) ||
    (Array.isArray(providerConfig.models) && providerConfig.models.length > 0);
  if (!hasApiConfig) {
    return false;
  }

  const authOverride = resolveProviderAuthOverride(params.cfg, params.provider);
  if (authOverride && authOverride !== "api-key") {
    return false;
  }
  if (!isCustomLocalProviderConfig(providerConfig)) {
    return false;
  }
  if (hasExplicitProviderApiKeyConfig(providerConfig)) {
    return false;
  }
  return Boolean(providerConfig.baseUrl && isLocalBaseUrl(providerConfig.baseUrl));
}

function listProviderSyntheticAuthRefs(params: {
  cfg: OpenClawConfig | undefined;
  provider: string;
  modelApi?: string;
}): string[] {
  const refs = [params.provider];
  const providerConfig = resolveProviderConfig(params.cfg, params.provider);
  if (params.modelApi) {
    refs.push(params.modelApi);
  }
  if (providerConfig?.api) {
    refs.push(providerConfig.api);
  }
  return normalizeUniqueStringEntries(refs.map((ref) => normalizeProviderId(ref)));
}

function shouldResolvePluginSyntheticAuth(params: {
  cfg: OpenClawConfig | undefined;
  provider: string;
  modelApi?: string;
  runtimeLookup?: RuntimeProviderAuthLookup;
}): boolean {
  const syntheticAuthProviderRefs = params.runtimeLookup?.syntheticAuthProviderRefs;
  if (!syntheticAuthProviderRefs) {
    return true;
  }
  const eligibleRefs = new Set(
    normalizeUniqueStringEntries(syntheticAuthProviderRefs.map((ref) => normalizeProviderId(ref))),
  );
  if (eligibleRefs.size === 0) {
    return false;
  }
  return listProviderSyntheticAuthRefs(params).some((ref) => eligibleRefs.has(ref));
}

/** Fast auth-availability check for runtime provider/model selection. */
export function hasRuntimeAvailableProviderAuth(params: {
  provider: string;
  cfg?: OpenClawConfig;
  workspaceDir?: string;
  env?: NodeJS.ProcessEnv;
  allowPluginSyntheticAuth?: boolean;
  runtimeLookup?: RuntimeProviderAuthLookup;
  modelApi?: string;
  /** Selected model base URL, used to detect Vertex routing for AI Studio apis. */
  baseUrl?: string;
}): boolean {
  const provider = normalizeProviderId(params.provider);
  const authOverride = resolveProviderAuthOverride(params.cfg, provider);
  if (authOverride === "aws-sdk") {
    return true;
  }
  // A per-model base URL override outranks the provider config base URL when
  // detecting Vertex routing for the ADC-marker scoping below.
  const effectiveBaseUrl = params.baseUrl ?? resolveProviderConfig(params.cfg, provider)?.baseUrl;
  const envAuth = resolveModelScopedEnvApiKey({
    provider,
    modelApi: params.modelApi,
    baseUrl: effectiveBaseUrl,
    resolve: (resolvedProvider) =>
      resolveEnvApiKey(resolvedProvider, params.env, {
        config: params.cfg,
        workspaceDir: params.workspaceDir,
        ...resolveRuntimeEnvApiKeyLookupOptions({
          provider: resolvedProvider,
          runtimeLookup: params.runtimeLookup,
        }),
      }),
  });
  if (
    envAuth &&
    isAuthModeAllowedForModel({
      provider,
      modelApi: params.modelApi,
      mode: envAuth.source.includes("OAUTH_TOKEN") ? "oauth" : "api-key",
    })
  ) {
    return true;
  }
  if (
    resolveUsableCustomProviderApiKey({
      cfg: params.cfg,
      provider,
      modelApi: params.modelApi,
      baseUrl: effectiveBaseUrl,
      env: params.env,
    })
  ) {
    return true;
  }
  if (resolveManagedSecretRefRuntimeProviderAuth({ cfg: params.cfg, provider })) {
    return true;
  }
  if (hasSyntheticLocalProviderAuthConfig({ cfg: params.cfg, provider })) {
    return true;
  }
  if (
    params.allowPluginSyntheticAuth !== false &&
    shouldResolvePluginSyntheticAuth({
      cfg: params.cfg,
      provider,
      runtimeLookup: params.runtimeLookup,
    }) &&
    resolveSyntheticLocalProviderAuth({ cfg: params.cfg, provider })
  ) {
    return true;
  }
  return false;
}

type SyntheticProviderAuthResolution = {
  auth?: ResolvedProviderAuth;
  blockedOnManagedSecretRef?: boolean;
};

function resolveProviderSyntheticRuntimeAuth(params: {
  cfg: OpenClawConfig | undefined;
  provider: string;
  modelApi?: string;
}): SyntheticProviderAuthResolution {
  const runtimeAuth = resolveManagedSecretRefRuntimeProviderAuth(params);
  if (runtimeAuth) {
    return { auth: runtimeAuth };
  }
  if (hasManagedSecretRefProviderApiKey(params.cfg, params.provider)) {
    return { blockedOnManagedSecretRef: true };
  }

  const resolveFromConfig = (
    config: OpenClawConfig | undefined,
  ): ResolvedProviderAuth | undefined => {
    const providerConfig = resolveProviderConfig(config, params.provider);
    return (
      resolveProviderSyntheticAuthWithPlugin({
        provider: params.provider,
        config,
        context: {
          config,
          provider: params.provider,
          providerConfig,
        },
        modelApi: params.modelApi,
      }) ?? undefined
    );
  };

  const directAuth = resolveFromConfig(params.cfg);
  if (!directAuth) {
    return {};
  }
  if (!isManagedSecretRefApiKeyMarker(directAuth.apiKey)) {
    return { auth: directAuth };
  }

  const runtimeConfig = getRuntimeConfigSnapshot();
  if (!runtimeConfig || runtimeConfig === params.cfg) {
    return { blockedOnManagedSecretRef: true };
  }

  const runtimePluginAuth = resolveFromConfig(runtimeConfig);
  const runtimeApiKey = runtimePluginAuth?.apiKey;
  if (!runtimePluginAuth || !runtimeApiKey || isNonSecretApiKeyMarker(runtimeApiKey)) {
    return { blockedOnManagedSecretRef: true };
  }
  return {
    auth: runtimePluginAuth,
  };
}

function resolveSyntheticLocalProviderAuth(params: {
  cfg: OpenClawConfig | undefined;
  provider: string;
  modelApi?: string;
}): ResolvedProviderAuth | null {
  const syntheticProviderAuth = resolveProviderSyntheticRuntimeAuth(params);
  if (syntheticProviderAuth.auth) {
    return syntheticProviderAuth.auth;
  }
  if (syntheticProviderAuth.blockedOnManagedSecretRef) {
    return null;
  }

  const providerConfig = resolveProviderConfig(params.cfg, params.provider);
  if (!providerConfig) {
    return null;
  }

  // Custom providers pointing at a local server (e.g. llama.cpp, vLLM, LocalAI)
  // typically don't require auth. Synthesize a local key so the auth resolver
  // doesn't reject them when the user left the API key blank during setup.
  if (hasSyntheticLocalProviderAuthConfig(params)) {
    return {
      apiKey: CUSTOM_LOCAL_AUTH_MARKER,
      source: `models.providers.${params.provider} (synthetic local key)`,
      mode: "api-key",
    };
  }

  return null;
}

function resolveEnvSourceLabel(params: {
  applied: Set<string>;
  envVars: string[];
  label: string;
}): string {
  const shellApplied = params.envVars.some((envVar) => params.applied.has(envVar));
  const prefix = shellApplied ? "shell env: " : "env: ";
  return `${prefix}${params.label}`;
}

function resolveAwsSdkAuthInfo(): { mode: "aws-sdk"; source: string } {
  const applied = new Set(getShellEnvAppliedKeys());
  if (process.env.AWS_BEARER_TOKEN_BEDROCK?.trim()) {
    return {
      mode: "aws-sdk",
      source: resolveEnvSourceLabel({
        applied,
        envVars: ["AWS_BEARER_TOKEN_BEDROCK"],
        label: "AWS_BEARER_TOKEN_BEDROCK",
      }),
    };
  }
  if (process.env.AWS_ACCESS_KEY_ID?.trim() && process.env.AWS_SECRET_ACCESS_KEY?.trim()) {
    return {
      mode: "aws-sdk",
      source: resolveEnvSourceLabel({
        applied,
        envVars: ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"],
        label: "AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY",
      }),
    };
  }
  if (process.env.AWS_PROFILE?.trim()) {
    return {
      mode: "aws-sdk",
      source: resolveEnvSourceLabel({
        applied,
        envVars: ["AWS_PROFILE"],
        label: "AWS_PROFILE",
      }),
    };
  }
  return { mode: "aws-sdk", source: "aws-sdk default chain" };
}

function shouldDeferSyntheticProfileAuth(params: {
  cfg: OpenClawConfig | undefined;
  provider: string;
  resolvedApiKey: string | undefined;
  modelApi?: string;
}): boolean {
  const providerConfig = resolveProviderConfig(params.cfg, params.provider);
  return (
    shouldDeferProviderSyntheticProfileAuthWithPlugin({
      provider: params.provider,
      config: params.cfg,
      modelApi: params.modelApi,
      context: {
        config: params.cfg,
        provider: params.provider,
        providerConfig,
        resolvedApiKey: params.resolvedApiKey,
      },
    }) === true
  );
}

function resolveScopedAuthProfileStore(params: {
  agentDir?: string;
  cfg?: OpenClawConfig;
  provider: string;
  profileId?: string;
  preferredProfile?: string;
}): AuthProfileStore {
  return ensureAuthProfileStore(params.agentDir, {
    externalCli: externalCliDiscoveryForProviderAuth(params),
  });
}

/** Resolves the credential that should be used for one provider request. */
export async function resolveApiKeyForProvider(params: {
  provider: string;
  cfg?: OpenClawConfig;
  profileId?: string;
  preferredProfile?: string;
  store?: AuthProfileStore;
  agentDir?: string;
  workspaceDir?: string;
  /** When true, treat profileId as a user-locked selection that must not be
   *  silently overridden by env/config credentials. */
  lockedProfile?: boolean;
  forceRefresh?: boolean;
  credentialPrecedence?: ProviderCredentialPrecedence;
  modelApi?: string;
  /** Selected model base URL, used to detect Vertex routing for AI Studio apis. */
  baseUrl?: string;
}): Promise<ResolvedProviderAuth> {
  const { provider, cfg, profileId, preferredProfile } = params;
  const agentDir = params.agentDir?.trim() || (cfg ? resolveDefaultAgentDir(cfg) : undefined);
  let scopedStore: AuthProfileStore | undefined = params.store;
  // A Vertex model registered under the shared "google" provider must not be
  // dispatched with that provider's stored AI Studio (Gemini) profiles — the
  // Vertex transport rejects them. Such requests resolve their credential from the
  // dedicated "google-vertex" provider (env ADC marker/metadata, configured key, or
  // a google-vertex profile) via the fallback below, so the generic "google"
  // profile order is skipped here.
  const isVertexRequestUnderGenericGoogle =
    isVertexAuthModelRequest(
      params.modelApi,
      params.baseUrl ?? resolveProviderConfig(cfg, provider)?.baseUrl,
    ) && normalizeProviderId(provider) === "google";
  // Every env-credential acceptance below drops a Vertex ADC marker that does
  // not apply to the explicitly selected model, and falls back to the
  // "google-vertex" auth provider for a Vertex model registered under a generic
  // provider id (see resolveModelScopedEnvApiKey / scopeVertexAdcEnvMarker).
  const resolveScopedEnvApiKey = (): EnvApiKeyResult | null =>
    resolveModelScopedEnvApiKey({
      provider,
      modelApi: params.modelApi,
      // A per-model baseUrl override (e.g. an aiplatform Vertex host on a
      // google-generative-ai model) routes the request through Vertex, so prefer it
      // over the provider-level baseUrl when detecting Vertex routing.
      baseUrl: params.baseUrl ?? resolveProviderConfig(cfg, provider)?.baseUrl,
      resolve: (resolvedProvider) =>
        resolveConfigAwareEnvApiKey(cfg, resolvedProvider, params.workspaceDir),
    });

  if (profileId) {
    const awsSdkProfileAuth = resolveConfiguredAwsSdkProfileAuth({ cfg, provider, profileId });
    if (awsSdkProfileAuth) {
      return awsSdkProfileAuth;
    }
    const store =
      params.store ??
      resolveScopedAuthProfileStore({
        agentDir,
        cfg,
        provider,
        profileId,
        preferredProfile,
      });
    // A Vertex model under the shared "google" provider must not be dispatched with
    // that provider's AI Studio (Gemini) profile. The auth controller passes each
    // candidate as an explicit profileId, so guard this branch too: ignore a
    // generic "google" credential and resolve the Vertex request from the dedicated
    // "google-vertex" provider instead (env ADC/metadata, configured key, or a
    // google-vertex profile) via the fallback below.
    if (
      isVertexRequestUnderGenericGoogle &&
      normalizeProviderId(store.profiles[profileId]?.provider ?? provider) !== "google-vertex"
    ) {
      return resolveApiKeyForProvider({ ...params, store, profileId: undefined });
    }
    const resolved = await resolveApiKeyForProfile({
      cfg,
      store,
      profileId,
      agentDir,
      forceRefresh: params.forceRefresh,
    });
    if (!resolved) {
      throw new Error(`No credentials found for profile "${profileId}".`);
    }
    const resolvedProfileId = resolved.profileId ?? profileId;
    const mode = resolved.profileType ?? store.profiles[resolvedProfileId]?.type;
    const result: ResolvedProviderAuth = {
      apiKey: resolved.apiKey,
      profileId: resolvedProfileId,
      source: `profile:${resolvedProfileId}`,
      mode: mode ? profileTypeToAuthMode(mode) : "api-key",
    };
    assertAuthModeAllowedForModel({
      provider,
      modelApi: params.modelApi,
      profileId: resolvedProfileId,
      mode: result.mode,
    });
    // When the resolved key is a provider-owned synthetic profile marker and
    // the caller has not locked this profile, fall through to env/config
    // resolution so provider-owned real credentials take precedence. The auth
    // controller iterates profile candidates and passes each as an explicit
    // profileId, so we cannot assume explicit === user-locked.
    if (
      !params.lockedProfile &&
      shouldDeferSyntheticProfileAuth({
        cfg,
        provider,
        resolvedApiKey: resolved.apiKey,
        modelApi: params.modelApi,
      })
    ) {
      return resolveApiKeyForProvider({
        ...params,
        store,
        profileId: undefined,
        lockedProfile: true,
      }) //
        .catch(() => result);
    }
    return result;
  }

  if (cfg?.auth?.profiles || cfg?.auth?.order) {
    scopedStore ??= resolveScopedAuthProfileStore({
      agentDir,
      cfg,
      provider,
      preferredProfile,
    });
    const configuredProfileOrder = resolveAuthProfileOrder({
      cfg,
      store: scopedStore,
      provider,
      preferredProfile,
    });
    for (const candidate of configuredProfileOrder) {
      const awsSdkProfileAuth = resolveConfiguredAwsSdkProfileAuth({
        cfg,
        provider,
        profileId: candidate,
      });
      if (awsSdkProfileAuth) {
        return awsSdkProfileAuth;
      }
    }
  }

  const authOverride = resolveProviderAuthOverride(cfg, provider);
  if (authOverride === "aws-sdk") {
    return resolveAwsSdkAuthInfo();
  }
  if (shouldUseImplicitAwsSdkAuth({ cfg, provider, modelApi: params.modelApi })) {
    return resolveAwsSdkAuthInfo();
  }

  if (params.credentialPrecedence === "env-first") {
    const envResolved = resolveScopedEnvApiKey();
    if (envResolved) {
      const resolvedMode: ResolvedProviderAuth["mode"] = envResolved.source.includes("OAUTH_TOKEN")
        ? "oauth"
        : "api-key";
      if (
        !isAuthModeAllowedForModel({
          provider,
          modelApi: params.modelApi,
          mode: resolvedMode,
        })
      ) {
        return resolveApiKeyForProvider({ ...params, credentialPrecedence: "profile-first" });
      }
      return {
        apiKey: envResolved.apiKey,
        source: envResolved.source,
        mode: resolvedMode,
      };
    }
  }

  // Resolve stored profile-id references before literal apiKey fallbacks.
  // Matched profile references are terminal so bad bindings cannot silently
  // fall through to a different credential or to the profile id as bearer text.
  scopedStore ??= resolveScopedAuthProfileStore({
    agentDir,
    cfg,
    provider,
    preferredProfile,
  });
  const providerEntryBinding = await resolveProviderEntryApiKeyBinding({
    cfg,
    provider,
    store: scopedStore,
    agentDir,
  });
  if (providerEntryBinding.kind === "profile-resolved") {
    assertAuthModeAllowedForModel({
      provider,
      modelApi: params.modelApi,
      profileId: providerEntryBinding.auth.profileId ?? provider,
      mode: providerEntryBinding.auth.mode,
    });
    return providerEntryBinding.auth;
  }
  if (providerEntryBinding.kind === "profile-incompatible") {
    const reason =
      providerEntryBinding.reason === "credential-class"
        ? "which is not a bearer-style auth class"
        : "which is not compatible with this provider entry's auth binding";
    const action =
      providerEntryBinding.reason === "credential-class"
        ? "Use an api-key or token profile, or set apiKey to a literal bearer token."
        : "Use a compatible provider auth alias, configure the referenced provider entry with the same baseUrl, or set apiKey to a literal bearer token.";
    throw new Error(
      `Per-entry apiKey "${providerEntryBinding.profileId}" for provider "${provider}" references a "${providerEntryBinding.credentialType}" credential for provider "${providerEntryBinding.credentialProvider}", ${reason}. ${action}`,
    );
  }
  if (providerEntryBinding.kind === "profile-unresolved") {
    const cause = providerEntryBinding.error
      ? formatErrorMessage(providerEntryBinding.error)
      : "credential resolution returned no key";
    throw new Error(
      `Per-entry apiKey "${providerEntryBinding.profileId}" for provider "${provider}" matched a stored profile but failed to resolve: ${cause}. Fix the referenced profile or set apiKey to a literal bearer token.`,
    );
  }

  if (shouldPreferExplicitConfigApiKeyAuth(cfg, provider)) {
    const runtimeCustomKey = resolveManagedSecretRefRuntimeProviderAuth({ cfg, provider });
    if (runtimeCustomKey) {
      return runtimeCustomKey;
    }
    const customKey = resolveUsableCustomProviderApiKey({
      cfg,
      provider,
      modelApi: params.modelApi,
      baseUrl: params.baseUrl,
    });
    if (customKey) {
      return {
        apiKey: customKey.apiKey,
        source: customKey.source,
        mode: "api-key",
      };
    }
  }
  const providerConfig = resolveProviderConfig(cfg, provider);
  const configuredLocalKey = resolveUsableCustomProviderApiKey({
    cfg,
    provider,
    modelApi: params.modelApi,
    baseUrl: params.baseUrl,
  });
  if (configuredLocalKey && isNonSecretApiKeyMarker(configuredLocalKey.apiKey)) {
    return {
      apiKey: configuredLocalKey.apiKey,
      source: configuredLocalKey.source,
      mode: "api-key",
    };
  }
  const localMarkerEnv = resolveScopedEnvApiKey();
  if (localMarkerEnv && isNonSecretApiKeyMarker(localMarkerEnv.apiKey)) {
    return {
      apiKey: localMarkerEnv.apiKey,
      source: localMarkerEnv.source,
      mode: "api-key",
    };
  }
  const store =
    scopedStore ??
    resolveScopedAuthProfileStore({
      agentDir,
      cfg,
      provider,
      preferredProfile,
    });
  const order = isVertexRequestUnderGenericGoogle
    ? []
    : resolveAuthProfileOrder({
        cfg,
        store,
        provider,
        preferredProfile,
      });
  let deferredAuthProfileResult: ResolvedProviderAuth | null = null;
  for (const candidate of order) {
    try {
      const awsSdkProfileAuth = resolveConfiguredAwsSdkProfileAuth({
        cfg,
        provider,
        profileId: candidate,
      });
      if (awsSdkProfileAuth) {
        return awsSdkProfileAuth;
      }
      const resolved = await resolveApiKeyForProfile({
        cfg,
        store,
        profileId: candidate,
        agentDir,
        forceRefresh: params.forceRefresh,
      });
      if (resolved) {
        const resolvedProfileId = resolved.profileId ?? candidate;
        const mode = resolved.profileType ?? store.profiles[resolvedProfileId]?.type;
        const resolvedMode: ResolvedProviderAuth["mode"] = mode
          ? profileTypeToAuthMode(mode)
          : "api-key";
        const result: ResolvedProviderAuth = {
          apiKey: resolved.apiKey,
          profileId: resolvedProfileId,
          source: `profile:${resolvedProfileId}`,
          mode: resolvedMode,
        };
        if (
          !isAuthModeAllowedForModel({
            provider,
            modelApi: params.modelApi,
            mode: result.mode,
          })
        ) {
          continue;
        }
        if (
          shouldDeferSyntheticProfileAuth({
            cfg,
            provider,
            resolvedApiKey: resolved.apiKey,
            modelApi: params.modelApi,
          })
        ) {
          deferredAuthProfileResult ??= result;
          continue;
        }
        return result;
      }
    } catch (err) {
      log.debug?.(`auth profile "${candidate}" failed for provider "${provider}": ${String(err)}`);
    }
  }

  const envResolved = resolveScopedEnvApiKey();
  if (envResolved) {
    const resolvedMode: ResolvedProviderAuth["mode"] = envResolved.source.includes("OAUTH_TOKEN")
      ? "oauth"
      : "api-key";
    if (
      isAuthModeAllowedForModel({
        provider,
        modelApi: params.modelApi,
        mode: resolvedMode,
      })
    ) {
      const result: ResolvedProviderAuth = {
        apiKey: envResolved.apiKey,
        source: envResolved.source,
        mode: resolvedMode,
      };
      return result;
    }
  }

  const customKey = resolveUsableCustomProviderApiKey({
    cfg,
    provider,
    modelApi: params.modelApi,
    baseUrl: params.baseUrl,
  });
  if (customKey) {
    const result = { apiKey: customKey.apiKey, source: customKey.source, mode: "api-key" as const };
    return result;
  }

  if (deferredAuthProfileResult) {
    return deferredAuthProfileResult;
  }

  const syntheticLocalAuth = resolveSyntheticLocalProviderAuth({
    cfg,
    provider,
    modelApi: params.modelApi,
  });
  if (syntheticLocalAuth) {
    return syntheticLocalAuth;
  }

  // A Vertex model registered under a non-Vertex provider id (for example the
  // shared "google" provider) resolves its credential from the dedicated
  // "google-vertex" provider. The env-based fallback above already covers the ADC
  // marker / metadata-server env evidence; consult the google-vertex provider's
  // full resolution (its stored profiles and configured keys too) so this path
  // matches both the model-list availability index and the session model registry,
  // which already dispatch a Vertex-under-"google" model with google-vertex auth.
  if (
    isVertexAuthModelRequest(params.modelApi, params.baseUrl ?? providerConfig?.baseUrl) &&
    normalizeProviderId(provider) !== "google-vertex"
  ) {
    try {
      return await resolveApiKeyForProvider({
        ...params,
        provider: "google-vertex",
        profileId: undefined,
        preferredProfile: undefined,
      });
    } catch {
      // Fall through to the requested provider's own missing-auth error below.
    }
  }

  const hasInlineConfiguredModels =
    Array.isArray(providerConfig?.models) && providerConfig.models.length > 0;
  const owningPluginIds = !hasInlineConfiguredModels
    ? resolveOwningPluginIdsForProviderRef({
        provider,
        config: cfg,
      })
    : undefined;
  if (owningPluginIds?.length) {
    const pluginMissingAuthMessage = buildProviderMissingAuthMessageWithPlugin({
      provider,
      config: cfg,
      context: {
        config: cfg,
        agentDir,
        env: process.env,
        provider,
        listProfileIds: (providerId) => listProfilesForProvider(store, providerId),
      },
    });
    if (pluginMissingAuthMessage) {
      throw new ProviderAuthError("missing-provider-auth", provider, pluginMissingAuthMessage);
    }
  }

  const authStorePath = resolveAuthStorePathForDisplay(agentDir);
  const resolvedAgentDir = path.dirname(authStorePath);
  throw new ProviderAuthError(
    "missing-provider-auth",
    provider,
    [
      `No API key found for provider "${provider}".`,
      `Auth store: ${authStorePath} (agentDir: ${resolvedAgentDir}).`,
      `Configure auth for this agent (${formatCliCommand("openclaw agents add <id>")}) or copy only portable static auth profiles from the main agentDir.`,
    ].join(" "),
  );
}

export type ModelAuthMode = "api-key" | "oauth" | "token" | "mixed" | "aws-sdk" | "unknown";

export { resolveEnvApiKey } from "./model-auth-env.js";
export type { EnvApiKeyResult } from "./model-auth-env.js";

/** Reports the strongest configured auth mode for provider-list UI and diagnostics. */
export function resolveModelAuthMode(
  provider?: string,
  cfg?: OpenClawConfig,
  store?: AuthProfileStore,
  options?: { workspaceDir?: string },
): ModelAuthMode | undefined {
  const resolved = provider?.trim();
  if (!resolved) {
    return undefined;
  }

  const authOverride = resolveProviderAuthOverride(cfg, resolved);
  if (authOverride === "aws-sdk") {
    return "aws-sdk";
  }

  const authStore =
    store ??
    resolveScopedAuthProfileStore({
      cfg,
      provider: resolved,
    });
  const profiles = listProfilesForProvider(authStore, resolved);
  if (profiles.length > 0) {
    const modes = new Set(
      profiles
        .map((id) => authStore.profiles[id]?.type)
        .filter((mode): mode is "api_key" | "oauth" | "token" => Boolean(mode)),
    );
    const distinct = ["oauth", "token", "api_key"].filter((k) =>
      modes.has(k as "oauth" | "token" | "api_key"),
    );
    if (distinct.length >= 2) {
      return "mixed";
    }
    if (modes.has("oauth")) {
      return "oauth";
    }
    if (modes.has("token")) {
      return "token";
    }
    if (modes.has("api_key")) {
      return "api-key";
    }
  }

  const envKey = resolveConfigAwareEnvApiKey(cfg, resolved, options?.workspaceDir);
  if (envKey?.apiKey) {
    return envKey.source.includes("OAUTH_TOKEN") ? "oauth" : "api-key";
  }

  if (
    normalizeProviderId(resolved) === "codex" &&
    cliCredentials.readCodexCliCredentialsCached({ ttlMs: 5_000, allowKeychainPrompt: false })
  ) {
    return "oauth";
  }

  if (hasUsableCustomProviderApiKey(cfg, resolved)) {
    return "api-key";
  }

  return "unknown";
}

/** Checks provider auth availability, including profile fallback order. */
export async function hasAvailableAuthForProvider(params: {
  provider: string;
  cfg?: OpenClawConfig;
  preferredProfile?: string;
  store?: AuthProfileStore;
  agentDir?: string;
  workspaceDir?: string;
  modelApi?: string;
  /** Selected model base URL, used to detect Vertex routing for AI Studio apis. */
  baseUrl?: string;
}): Promise<boolean> {
  const { provider, cfg, preferredProfile } = params;
  // A per-model base URL override outranks the provider config base URL when
  // detecting Vertex routing.
  const effectiveBaseUrl = params.baseUrl ?? resolveProviderConfig(cfg, provider)?.baseUrl;
  // A Vertex model under the shared "google" provider is not satisfied by that
  // provider's AI Studio (Gemini) profiles; its availability comes from the
  // dedicated "google-vertex" provider via the fallback below, mirroring
  // resolveApiKeyForProvider.
  const isVertexRequestUnderGenericGoogle =
    isVertexAuthModelRequest(params.modelApi, effectiveBaseUrl) &&
    normalizeProviderId(provider) === "google";

  const authOverride = resolveProviderAuthOverride(cfg, provider);
  if (authOverride === "aws-sdk") {
    return true;
  }
  const envAuth = resolveModelScopedEnvApiKey({
    provider,
    modelApi: params.modelApi,
    baseUrl: effectiveBaseUrl,
    resolve: (resolvedProvider) =>
      resolveConfigAwareEnvApiKey(cfg, resolvedProvider, params.workspaceDir),
  });
  if (
    envAuth &&
    isAuthModeAllowedForModel({
      provider,
      modelApi: params.modelApi,
      mode: envAuth.source.includes("OAUTH_TOKEN") ? "oauth" : "api-key",
    })
  ) {
    return true;
  }
  if (
    resolveUsableCustomProviderApiKey({
      cfg,
      provider,
      modelApi: params.modelApi,
      baseUrl: effectiveBaseUrl,
    })
  ) {
    return true;
  }
  if (resolveSyntheticLocalProviderAuth({ cfg, provider })) {
    return true;
  }
  const store =
    params.store ??
    resolveScopedAuthProfileStore({
      agentDir: params.agentDir,
      cfg,
      provider,
      preferredProfile,
    });
  const order = isVertexRequestUnderGenericGoogle
    ? []
    : resolveAuthProfileOrder({
        cfg,
        store,
        provider,
        preferredProfile,
      });
  for (const candidate of order) {
    try {
      if (resolveConfiguredAwsSdkProfileAuth({ cfg, provider, profileId: candidate })) {
        return true;
      }
      const resolved = await resolveApiKeyForProfile({
        cfg,
        store,
        profileId: candidate,
        agentDir: params.agentDir,
      });
      const mode = resolved?.profileType ?? store.profiles[candidate]?.type;
      if (
        resolved &&
        isAuthModeAllowedForModel({
          provider,
          modelApi: params.modelApi,
          mode: mode ? profileTypeToAuthMode(mode) : "api-key",
        })
      ) {
        return true;
      }
    } catch (err) {
      log.debug?.(`auth profile "${candidate}" failed for provider "${provider}": ${String(err)}`);
    }
  }
  // Mirror resolveApiKeyForProvider: a Vertex model registered under a non-Vertex
  // provider id (for example the shared "google" provider) is available when the
  // dedicated "google-vertex" provider has usable auth (env ADC marker/metadata,
  // configured key, or a stored profile).
  if (
    isVertexAuthModelRequest(params.modelApi, effectiveBaseUrl) &&
    normalizeProviderId(provider) !== "google-vertex"
  ) {
    return hasAvailableAuthForProvider({
      ...params,
      provider: "google-vertex",
      preferredProfile: undefined,
    });
  }
  return false;
}

/** Resolves request credentials from the provider attached to a model descriptor. */
export async function getApiKeyForModel(params: {
  model: Model;
  cfg?: OpenClawConfig;
  profileId?: string;
  preferredProfile?: string;
  store?: AuthProfileStore;
  agentDir?: string;
  workspaceDir?: string;
  lockedProfile?: boolean;
  credentialPrecedence?: ProviderCredentialPrecedence;
}): Promise<ResolvedProviderAuth> {
  return resolveApiKeyForProvider({
    provider: params.model.provider,
    cfg: params.cfg,
    profileId: params.profileId,
    preferredProfile: params.preferredProfile,
    store: params.store,
    agentDir: params.agentDir,
    workspaceDir: params.workspaceDir,
    lockedProfile: params.lockedProfile,
    credentialPrecedence: params.credentialPrecedence,
    modelApi: params.model.api,
    baseUrl: params.model.baseUrl,
  });
}

/** Clears auth for local OpenAI-compatible servers that explicitly use no auth. */
export function applyLocalNoAuthHeaderOverride<T extends Model>(
  model: T,
  auth: ResolvedProviderAuth | null | undefined,
): T {
  if (auth?.apiKey !== CUSTOM_LOCAL_AUTH_MARKER || model.api !== "openai-completions") {
    return model;
  }

  // OpenAI's SDK always generates Authorization from apiKey. Keep the non-secret
  // placeholder so construction succeeds, then clear the header at request build
  // time for local servers that intentionally do not require auth.
  const headers = {
    ...model.headers,
    Authorization: null,
  } as unknown as Record<string, string>;

  return {
    ...model,
    headers,
  };
}

/**
 * When the provider config sets `authHeader: true`, inject an explicit
 * `Authorization: Bearer <apiKey>` header into the model so downstream SDKs
 * (e.g. `@google/genai`) send credentials via the standard HTTP Authorization
 * header instead of vendor-specific headers like `x-goog-api-key`.
 *
 * This is a no-op when `authHeader` is not `true`, when no API key is
 * available, or when the API key is a synthetic marker (e.g. local-server
 * placeholders) rather than a real credential.
 */
export function applyAuthHeaderOverride<T extends Model>(
  model: T,
  auth: ResolvedProviderAuth | null | undefined,
  cfg: OpenClawConfig | undefined,
): T {
  if (!auth?.apiKey) {
    return model;
  }
  // Reject synthetic marker values that are not real credentials.
  if (isNonSecretApiKeyMarker(auth.apiKey)) {
    return model;
  }
  const providerConfig = resolveProviderConfig(cfg, model.provider);
  if (!providerConfig?.authHeader) {
    return model;
  }

  // Strip any existing authorization header (case-insensitive) before
  // injecting the canonical one so we don't produce a comma-joined value.
  const headers: Record<string, string> = {};
  if (model.headers) {
    for (const [key, value] of Object.entries(model.headers)) {
      if (normalizeOptionalLowercaseString(key) !== "authorization") {
        headers[key] = value;
      }
    }
  }
  headers.Authorization = `Bearer ${auth.apiKey}`;

  return {
    ...model,
    headers,
  };
}
