import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterAll, describe, expect, it } from "vitest";
import { resolveGoogleVertexConfigApiKey } from "./vertex-adc.js";

// A HOME with no gcloud ADC file, so file-based ADC detection is deterministically
// false and only the metadata opt-in can supply the marker.
const HOME_WITHOUT_ADC = path.join(os.tmpdir(), "openclaw-google-vertex-no-adc-file");

function envWithoutAdcFile(overrides: NodeJS.ProcessEnv = {}): NodeJS.ProcessEnv {
  return {
    HOME: HOME_WITHOUT_ADC,
    GOOGLE_CLOUD_PROJECT: "vertex-project",
    GOOGLE_CLOUD_LOCATION: "global",
    ...overrides,
  };
}

describe("resolveGoogleVertexConfigApiKey metadata-server ADC", () => {
  it("emits the marker for metadata-server ADC opt-in without a credentials file", () => {
    expect(
      resolveGoogleVertexConfigApiKey(
        envWithoutAdcFile({ GOOGLE_VERTEX_USE_GCP_METADATA: "true" }),
      ),
    ).toBe("gcp-vertex-credentials");
  });

  it("accepts the numeric opt-in value", () => {
    expect(
      resolveGoogleVertexConfigApiKey(envWithoutAdcFile({ GOOGLE_VERTEX_USE_GCP_METADATA: "1" })),
    ).toBe("gcp-vertex-credentials");
  });

  it("accepts a case-insensitive opt-in value", () => {
    expect(
      resolveGoogleVertexConfigApiKey(
        envWithoutAdcFile({ GOOGLE_VERTEX_USE_GCP_METADATA: "TRUE" }),
      ),
    ).toBe("gcp-vertex-credentials");
  });

  // Parity with the env-flag auth-evidence path and the documented contract,
  // which accept all shared truthy spellings (1/true/yes/on).
  it.each(["yes", "on", "Yes", "ON"])("accepts the truthy opt-in value %s", (value) => {
    expect(
      resolveGoogleVertexConfigApiKey(envWithoutAdcFile({ GOOGLE_VERTEX_USE_GCP_METADATA: value })),
    ).toBe("gcp-vertex-credentials");
  });

  it("does not emit the marker without the opt-in when no credentials file is present", () => {
    expect(resolveGoogleVertexConfigApiKey(envWithoutAdcFile())).toBeUndefined();
  });

  it("does not accept the metadata opt-in when GOOGLE_APPLICATION_CREDENTIALS points at a missing file", () => {
    // google-auth-library loads GOOGLE_APPLICATION_CREDENTIALS before probing the
    // metadata server and throws on a bad path, so the request would fail despite
    // the opt-in; availability must not report Vertex as usable here.
    expect(
      resolveGoogleVertexConfigApiKey(
        envWithoutAdcFile({
          GOOGLE_VERTEX_USE_GCP_METADATA: "true",
          GOOGLE_APPLICATION_CREDENTIALS: path.join(HOME_WITHOUT_ADC, "missing-adc.json"),
        }),
      ),
    ).toBeUndefined();
  });

  it("does not accept the metadata opt-in when lowercase google_application_credentials is set", () => {
    // google-auth-library also reads the lowercase variant before probing the
    // metadata server, so an invalid lowercase path must likewise preempt the
    // metadata opt-in instead of falsely reporting Vertex as usable.
    expect(
      resolveGoogleVertexConfigApiKey(
        envWithoutAdcFile({
          GOOGLE_VERTEX_USE_GCP_METADATA: "true",
          google_application_credentials: path.join(HOME_WITHOUT_ADC, "missing-adc.json"),
        }),
      ),
    ).toBeUndefined();
  });

  it("ignores an unrelated opt-in value", () => {
    expect(
      resolveGoogleVertexConfigApiKey(envWithoutAdcFile({ GOOGLE_VERTEX_USE_GCP_METADATA: "no" })),
    ).toBeUndefined();
  });

  it("still requires project env even with the metadata opt-in", () => {
    expect(
      resolveGoogleVertexConfigApiKey({
        HOME: HOME_WITHOUT_ADC,
        GOOGLE_CLOUD_LOCATION: "global",
        GOOGLE_VERTEX_USE_GCP_METADATA: "true",
      }),
    ).toBeUndefined();
  });

  it("still requires location env even with the metadata opt-in", () => {
    expect(
      resolveGoogleVertexConfigApiKey({
        HOME: HOME_WITHOUT_ADC,
        GOOGLE_CLOUD_PROJECT: "vertex-project",
        GOOGLE_VERTEX_USE_GCP_METADATA: "true",
      }),
    ).toBeUndefined();
  });

  it("accepts GCLOUD_PROJECT as the project source", () => {
    expect(
      resolveGoogleVertexConfigApiKey({
        HOME: HOME_WITHOUT_ADC,
        GCLOUD_PROJECT: "vertex-project",
        GOOGLE_CLOUD_LOCATION: "global",
        GOOGLE_VERTEX_USE_GCP_METADATA: "true",
      }),
    ).toBe("gcp-vertex-credentials");
  });
});

describe("resolveGoogleVertexConfigApiKey file ADC precedence", () => {
  // A HOME that DOES contain a default ADC file, so we can assert the default file
  // is used only when neither explicit credentials env var is set.
  const adcHome = fs.mkdtempSync(path.join(os.tmpdir(), "openclaw-vertex-default-adc-"));
  const defaultAdc = path.join(
    adcHome,
    ".config",
    "gcloud",
    "application_default_credentials.json",
  );
  fs.mkdirSync(path.dirname(defaultAdc), { recursive: true });
  fs.writeFileSync(defaultAdc, JSON.stringify({ type: "authorized_user" }), "utf8");

  afterAll(() => {
    fs.rmSync(adcHome, { recursive: true, force: true });
  });

  it("uses the default ADC file when no explicit credentials env var is set", () => {
    expect(
      resolveGoogleVertexConfigApiKey({
        HOME: adcHome,
        GOOGLE_CLOUD_PROJECT: "vertex-project",
        GOOGLE_CLOUD_LOCATION: "global",
      }),
    ).toBe("gcp-vertex-credentials");
  });

  it("does not fall back to the default ADC file when lowercase google_application_credentials points at a missing file", () => {
    // google-auth-library reads GOOGLE_APPLICATION_CREDENTIALS || its lowercase
    // variant before the default ADC path and fails on a bad path rather than
    // falling through, so a present default file must not mask the broken one.
    expect(
      resolveGoogleVertexConfigApiKey({
        HOME: adcHome,
        google_application_credentials: path.join(adcHome, "missing.json"),
        GOOGLE_CLOUD_PROJECT: "vertex-project",
        GOOGLE_CLOUD_LOCATION: "global",
      }),
    ).toBeUndefined();
  });

  it("does not fall back to the default ADC file when uppercase GOOGLE_APPLICATION_CREDENTIALS points at a missing file", () => {
    expect(
      resolveGoogleVertexConfigApiKey({
        HOME: adcHome,
        GOOGLE_APPLICATION_CREDENTIALS: path.join(adcHome, "missing.json"),
        GOOGLE_CLOUD_PROJECT: "vertex-project",
        GOOGLE_CLOUD_LOCATION: "global",
      }),
    ).toBeUndefined();
  });
});
