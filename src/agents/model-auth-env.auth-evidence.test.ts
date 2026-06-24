// Verifies manifest auth-evidence resolution, including the env-flag opt-in used
// for ambient credentials (e.g. GCP metadata-server ADC) that cannot be detected
// synchronously.
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterAll, describe, expect, it } from "vitest";
import type { ProviderAuthEvidence } from "../secrets/provider-env-vars.js";
import { resolveEnvApiKey } from "./model-auth-env.js";

// google-vertex declares an explicit env candidate (GOOGLE_CLOUD_API_KEY), which
// otherwise short-circuits the setup-provider fallback. The env-flag evidence is
// evaluated before that early return, so an operator opt-in still resolves.
const candidateMap = { "google-vertex": ["GOOGLE_CLOUD_API_KEY"] };
const aliasMap = {};

const metadataEvidence: ProviderAuthEvidence = {
  type: "env-flag",
  flagEnvVars: ["GOOGLE_VERTEX_USE_GCP_METADATA"],
  requiresAnyEnv: ["GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT"],
  requiresAllEnv: ["GOOGLE_CLOUD_LOCATION"],
  requiresAbsentEnv: ["GOOGLE_APPLICATION_CREDENTIALS", "google_application_credentials"],
  credentialMarker: "gcp-vertex-credentials",
  source: "gcp metadata adc",
};

const authEvidenceMap = { "google-vertex": [metadataEvidence] };

const adcTempDir = fs.mkdtempSync(path.join(os.tmpdir(), "openclaw-adc-lowercase-"));
const adcHomeDir = fs.mkdtempSync(path.join(os.tmpdir(), "openclaw-adc-home-"));

afterAll(() => {
  fs.rmSync(adcTempDir, { recursive: true, force: true });
  fs.rmSync(adcHomeDir, { recursive: true, force: true });
});

function resolve(env: NodeJS.ProcessEnv) {
  return resolveEnvApiKey("google-vertex", env, { aliasMap, candidateMap, authEvidenceMap });
}

describe("resolveEnvApiKey env-flag auth evidence", () => {
  it("returns the credential marker when the opt-in flag is truthy and gating env is present", () => {
    expect(
      resolve({
        GOOGLE_VERTEX_USE_GCP_METADATA: "true",
        GOOGLE_CLOUD_PROJECT: "vertex-project",
        GOOGLE_CLOUD_LOCATION: "global",
      } as NodeJS.ProcessEnv),
    ).toEqual({ apiKey: "gcp-vertex-credentials", source: "gcp metadata adc" });
  });

  it("accepts other truthy flag spellings", () => {
    for (const value of ["1", "on", "YES", "True"]) {
      expect(
        resolve({
          GOOGLE_VERTEX_USE_GCP_METADATA: value,
          GCLOUD_PROJECT: "vertex-project",
          GOOGLE_CLOUD_LOCATION: "global",
        } as NodeJS.ProcessEnv),
      ).toEqual({ apiKey: "gcp-vertex-credentials", source: "gcp metadata adc" });
    }
  });

  it("does not resolve when the opt-in flag is absent or falsy", () => {
    expect(
      resolve({
        GOOGLE_CLOUD_PROJECT: "vertex-project",
        GOOGLE_CLOUD_LOCATION: "global",
      } as NodeJS.ProcessEnv),
    ).toBeNull();
    expect(
      resolve({
        GOOGLE_VERTEX_USE_GCP_METADATA: "false",
        GOOGLE_CLOUD_PROJECT: "vertex-project",
        GOOGLE_CLOUD_LOCATION: "global",
      } as NodeJS.ProcessEnv),
    ).toBeNull();
  });

  it("requires the gating project and location env even with the opt-in", () => {
    expect(
      resolve({
        GOOGLE_VERTEX_USE_GCP_METADATA: "true",
        GOOGLE_CLOUD_LOCATION: "global",
      } as NodeJS.ProcessEnv),
    ).toBeNull();
    expect(
      resolve({
        GOOGLE_VERTEX_USE_GCP_METADATA: "true",
        GOOGLE_CLOUD_PROJECT: "vertex-project",
      } as NodeJS.ProcessEnv),
    ).toBeNull();
  });

  it("suppresses the metadata opt-in when GOOGLE_APPLICATION_CREDENTIALS is set", () => {
    // A stale or invalid GOOGLE_APPLICATION_CREDENTIALS path is loaded first by the
    // auth library, so the metadata-server opt-in must not report Vertex available.
    expect(
      resolve({
        GOOGLE_VERTEX_USE_GCP_METADATA: "true",
        GOOGLE_CLOUD_PROJECT: "vertex-project",
        GOOGLE_CLOUD_LOCATION: "global",
        GOOGLE_APPLICATION_CREDENTIALS: "/var/run/secrets/missing-adc.json",
      } as NodeJS.ProcessEnv),
    ).toBeNull();
  });

  it("suppresses the metadata opt-in when lowercase google_application_credentials is set", () => {
    expect(
      resolve({
        GOOGLE_VERTEX_USE_GCP_METADATA: "true",
        GOOGLE_CLOUD_PROJECT: "vertex-project",
        GOOGLE_CLOUD_LOCATION: "global",
        google_application_credentials: "/var/run/secrets/missing-adc.json",
      } as NodeJS.ProcessEnv),
    ).toBeNull();
  });

  it("resolves file-based ADC from a lowercase google_application_credentials path", () => {
    // google-auth-library reads lowercase google_application_credentials, so a
    // lowercase-only valid file must still resolve as available (consistent with
    // the lowercase metadata-suppression guard) rather than reporting missing auth.
    const adcFile = path.join(adcTempDir, "adc.json");
    fs.writeFileSync(adcFile, "{}", "utf8");
    const fileEvidence: ProviderAuthEvidence = {
      type: "local-file-with-env",
      fileEnvVar: "google_application_credentials",
      requiresAnyEnv: ["GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT"],
      requiresAllEnv: ["GOOGLE_CLOUD_LOCATION"],
      credentialMarker: "gcp-vertex-credentials",
      source: "gcloud adc",
    };
    expect(
      resolveEnvApiKey(
        "google-vertex",
        {
          google_application_credentials: adcFile,
          GOOGLE_CLOUD_PROJECT: "vertex-project",
          GOOGLE_CLOUD_LOCATION: "global",
        } as NodeJS.ProcessEnv,
        { aliasMap, candidateMap, authEvidenceMap: { "google-vertex": [fileEvidence] } },
      ),
    ).toEqual({ apiKey: "gcp-vertex-credentials", source: "gcloud adc" });
  });

  it("prefers an explicit env API key over the opt-in marker", () => {
    expect(
      resolve({
        GOOGLE_CLOUD_API_KEY: "literal-key", // pragma: allowlist secret
        GOOGLE_VERTEX_USE_GCP_METADATA: "true",
        GOOGLE_CLOUD_PROJECT: "vertex-project",
        GOOGLE_CLOUD_LOCATION: "global",
      } as NodeJS.ProcessEnv),
    ).toEqual({ apiKey: "literal-key", source: "env: GOOGLE_CLOUD_API_KEY" });
  });
});

describe("resolveEnvApiKey ADC file-fallback precedence", () => {
  // Mirror the production google-vertex manifest evidence ordering: explicit
  // uppercase file, explicit lowercase file, then the default ADC file used only
  // when neither explicit credentials env var is set, then the metadata opt-in.
  const fullEvidence: readonly ProviderAuthEvidence[] = [
    {
      type: "local-file-with-env",
      fileEnvVar: "GOOGLE_APPLICATION_CREDENTIALS",
      requiresAnyEnv: ["GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT"],
      requiresAllEnv: ["GOOGLE_CLOUD_LOCATION"],
      credentialMarker: "gcp-vertex-credentials",
      source: "gcloud adc",
    },
    {
      type: "local-file-with-env",
      fileEnvVar: "google_application_credentials",
      requiresAnyEnv: ["GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT"],
      requiresAllEnv: ["GOOGLE_CLOUD_LOCATION"],
      requiresAbsentEnv: ["GOOGLE_APPLICATION_CREDENTIALS"],
      credentialMarker: "gcp-vertex-credentials",
      source: "gcloud adc",
    },
    {
      type: "local-file-with-env",
      fallbackPaths: ["${HOME}/.config/gcloud/application_default_credentials.json"],
      requiresAnyEnv: ["GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT"],
      requiresAllEnv: ["GOOGLE_CLOUD_LOCATION"],
      requiresAbsentEnv: ["GOOGLE_APPLICATION_CREDENTIALS", "google_application_credentials"],
      credentialMarker: "gcp-vertex-credentials",
      source: "gcloud adc",
    },
    metadataEvidence,
  ];

  const defaultAdcFile = path.join(
    adcHomeDir,
    ".config",
    "gcloud",
    "application_default_credentials.json",
  );
  fs.mkdirSync(path.dirname(defaultAdcFile), { recursive: true });
  fs.writeFileSync(defaultAdcFile, "{}", "utf8");

  function resolveFull(env: NodeJS.ProcessEnv) {
    return resolveEnvApiKey("google-vertex", env, {
      aliasMap,
      candidateMap,
      authEvidenceMap: { "google-vertex": fullEvidence },
    });
  }

  it("uses the default ADC file when no explicit credentials env var is set", () => {
    expect(
      resolveFull({
        HOME: adcHomeDir,
        GOOGLE_CLOUD_PROJECT: "vertex-project",
        GOOGLE_CLOUD_LOCATION: "global",
      } as NodeJS.ProcessEnv),
    ).toEqual({ apiKey: "gcp-vertex-credentials", source: "gcloud adc" });
  });

  it("does not fall back to the default ADC file when lowercase google_application_credentials points at a missing file", () => {
    // google-auth-library checks GOOGLE_APPLICATION_CREDENTIALS || its lowercase
    // variant before the default ADC path and fails on a bad path instead of
    // falling through, so a present default file must not mask the broken one.
    expect(
      resolveFull({
        HOME: adcHomeDir,
        google_application_credentials: path.join(adcHomeDir, "missing-adc.json"),
        GOOGLE_CLOUD_PROJECT: "vertex-project",
        GOOGLE_CLOUD_LOCATION: "global",
      } as NodeJS.ProcessEnv),
    ).toBeNull();
  });

  it("does not fall back to the default ADC file when uppercase GOOGLE_APPLICATION_CREDENTIALS points at a missing file", () => {
    expect(
      resolveFull({
        HOME: adcHomeDir,
        GOOGLE_APPLICATION_CREDENTIALS: path.join(adcHomeDir, "missing-adc.json"),
        GOOGLE_CLOUD_PROJECT: "vertex-project",
        GOOGLE_CLOUD_LOCATION: "global",
      } as NodeJS.ProcessEnv),
    ).toBeNull();
  });

  it("does not honor the lowercase ADC file when a stale uppercase GOOGLE_APPLICATION_CREDENTIALS is also set", () => {
    // google-auth-library reads GOOGLE_APPLICATION_CREDENTIALS first and throws on
    // its bad path without trying the lowercase variant, so a valid lowercase file
    // must not report Vertex available when the uppercase var is set to a stale path.
    const lowercaseAdc = path.join(adcHomeDir, "lowercase-adc.json");
    fs.writeFileSync(lowercaseAdc, "{}", "utf8");
    expect(
      resolveFull({
        HOME: adcHomeDir,
        GOOGLE_APPLICATION_CREDENTIALS: path.join(adcHomeDir, "missing-adc.json"),
        google_application_credentials: lowercaseAdc,
        GOOGLE_CLOUD_PROJECT: "vertex-project",
        GOOGLE_CLOUD_LOCATION: "global",
      } as NodeJS.ProcessEnv),
    ).toBeNull();
  });
});
