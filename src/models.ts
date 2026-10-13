/**
 * Model Configuration
 * Maps model names to encodings and context limits
 *
 * Context limits are synced with tiktoken-rs:
 * https://github.com/zurawiki/tiktoken-rs/blob/main/tiktoken-rs/src/model.rs
 */

import type { EncodingName, ModelConfig, ModelFamily } from "./types.js";
import {
  findContextSize,
  getExactContextSize,
} from "./core/model-to-encoding.js";

/**
 * Helper to create model config with context size from tiktoken-rs
 */
function createModelConfig(
  name: string,
  encoding: EncodingName,
  maxOutputTokens: number,
  family: ModelFamily,
  contextLimitOverride?: number,
): ModelConfig {
  return {
    name,
    encoding,
    contextLimit: contextLimitOverride ?? getExactContextSize(name),
    maxOutputTokens,
    family,
  };
}

/**
 * Model context limits and configurations
 * Context limits from tiktoken-rs, max output tokens from OpenAI docs
 */
export const MODEL_CONFIGS: Record<string, ModelConfig> = {
  // ==========================================================================
  // GPT-5 Series (context from tiktoken-rs)
  // ==========================================================================
  "gpt-5.4": createModelConfig("gpt-5.4", "o200k_base", 16384, "gpt-5"),
  "gpt-5.4-pro": createModelConfig("gpt-5.4-pro", "o200k_base", 16384, "gpt-5"),
  "gpt-5.4-mini": createModelConfig(
    "gpt-5.4-mini",
    "o200k_base",
    16384,
    "gpt-5",
  ),
  "gpt-5.4-nano": createModelConfig(
    "gpt-5.4-nano",
    "o200k_base",
    4096,
    "gpt-5",
  ),
  "gpt-5.2": createModelConfig("gpt-5.2", "o200k_base", 16384, "gpt-5"),
  "gpt-5.2-pro": createModelConfig("gpt-5.2-pro", "o200k_base", 16384, "gpt-5"),
  "gpt-5": createModelConfig("gpt-5", "o200k_base", 16384, "gpt-5"),
  "gpt-5-mini": createModelConfig("gpt-5-mini", "o200k_base", 16384, "gpt-5"),
  "gpt-5-nano": createModelConfig("gpt-5-nano", "o200k_base", 4096, "gpt-5"),
  "gpt-5-turbo": createModelConfig("gpt-5-turbo", "o200k_base", 16384, "gpt-5"),

  // ==========================================================================
  // Codex Series
  // ==========================================================================
  "gpt-5.3-codex": createModelConfig(
    "gpt-5.3-codex",
    "o200k_base",
    32768,
    "codex",
  ),
  "gpt-5.3-codex-spark": createModelConfig(
    "gpt-5.3-codex-spark",
    "o200k_base",
    32768,
    "codex",
  ),
  "gpt-5.2-codex": createModelConfig(
    "gpt-5.2-codex",
    "o200k_base",
    32768,
    "codex",
  ),
  "gpt-5.1-codex": createModelConfig(
    "gpt-5.1-codex",
    "o200k_base",
    32768,
    "codex",
  ),
  "gpt-5.1-codex-mini": createModelConfig(
    "gpt-5.1-codex-mini",
    "o200k_base",
    32768,
    "codex",
  ),
  "gpt-5-codex": createModelConfig("gpt-5-codex", "o200k_base", 32768, "codex"),
  "codex-mini": createModelConfig("codex-mini", "o200k_base", 32768, "codex"),

  // ==========================================================================
  // o-Series Reasoning Models (200K context from tiktoken-rs)
  // ==========================================================================
  o1: createModelConfig("o1", "o200k_base", 32768, "o-series"),
  "o1-mini": createModelConfig("o1-mini", "o200k_base", 16384, "o-series"),
  "o1-preview": createModelConfig(
    "o1-preview",
    "o200k_base",
    16384,
    "o-series",
  ),
  o3: createModelConfig("o3", "o200k_base", 32768, "o-series"),
  "o3-mini": createModelConfig("o3-mini", "o200k_base", 16384, "o-series"),
  "o3-pro": createModelConfig("o3-pro", "o200k_base", 65536, "o-series"),
  "o4-mini": createModelConfig("o4-mini", "o200k_base", 16384, "o-series"),

  // ==========================================================================
  // GPT-4.1 Series (1M+ context from tiktoken-rs!)
  // ==========================================================================
  "gpt-4.1": createModelConfig("gpt-4.1", "o200k_base", 16384, "gpt-4.1"),
  "gpt-4.1-mini": createModelConfig(
    "gpt-4.1-mini",
    "o200k_base",
    16384,
    "gpt-4.1",
  ),
  "gpt-4.1-nano": createModelConfig(
    "gpt-4.1-nano",
    "o200k_base",
    4096,
    "gpt-4.1",
  ),

  // ==========================================================================
  // GPT-4.5 Preview
  // ==========================================================================
  "gpt-4.5-preview": createModelConfig(
    "gpt-4.5-preview",
    "o200k_base",
    16384,
    "gpt-4o",
  ),

  // ==========================================================================
  // GPT-4o Series (128K context)
  // ==========================================================================
  "gpt-4o": createModelConfig("gpt-4o", "o200k_base", 16384, "gpt-4o"),
  "gpt-4o-mini": createModelConfig(
    "gpt-4o-mini",
    "o200k_base",
    16384,
    "gpt-4o",
  ),
  "chatgpt-4o-latest": createModelConfig(
    "chatgpt-4o-latest",
    "o200k_base",
    16384,
    "gpt-4o",
  ),

  // ==========================================================================
  // GPT-4 Legacy Series
  // ==========================================================================
  "gpt-4-turbo": createModelConfig("gpt-4-turbo", "cl100k_base", 4096, "gpt-4"),
  "gpt-4-turbo-preview": createModelConfig(
    "gpt-4-turbo-preview",
    "cl100k_base",
    4096,
    "gpt-4",
  ),
  "gpt-4": createModelConfig("gpt-4", "cl100k_base", 4096, "gpt-4"),
  "gpt-4-32k": createModelConfig("gpt-4-32k", "cl100k_base", 4096, "gpt-4"),

  // ==========================================================================
  // GPT-3.5 Series
  // ==========================================================================
  "gpt-3.5-turbo": createModelConfig(
    "gpt-3.5-turbo",
    "cl100k_base",
    4096,
    "gpt-3.5",
  ),
  "gpt-3.5-turbo-16k": createModelConfig(
    "gpt-3.5-turbo-16k",
    "cl100k_base",
    4096,
    "gpt-3.5",
  ),

  // ==========================================================================
  // Embedding Models
  // ==========================================================================
  "text-embedding-ada-002": createModelConfig(
    "text-embedding-ada-002",
    "cl100k_base",
    0,
    "embedding",
    8192,
  ),
  "text-embedding-3-small": createModelConfig(
    "text-embedding-3-small",
    "cl100k_base",
    0,
    "embedding",
    8191,
  ),
  "text-embedding-3-large": createModelConfig(
    "text-embedding-3-large",
    "cl100k_base",
    0,
    "embedding",
    8191,
  ),

  // ==========================================================================
  // Anthropic Claude Models
  //
  // IMPORTANT: Claude uses a proprietary tokenizer that is NOT publicly available.
  // The "claude_estimation" encoding provides SAFE estimates that intentionally
  // over-count tokens (25% safety margin) to prevent API truncation issues.
  //
  // Research findings (see claude-tokenizer-research.md):
  // - Claude 3+ uses ~22,000 token vocabulary (smaller than OpenAI's 100K-200K)
  // - Claude produces 16-30% MORE tokens than GPT-4 for equivalent content
  // - Average ~3.5 characters per token (vs GPT-4's ~4)
  //
  // For exact Claude token counts, use Anthropic's /v1/messages/count_tokens API.
  // @see https://docs.anthropic.com/en/docs/build-with-claude/token-counting
  // ==========================================================================

  // Claude 4.5 Series
  "claude-4-5-opus": createModelConfig(
    "claude-4-5-opus",
    "claude_estimation",
    32768,
    "claude",
    200_000,
  ),
  "claude-4.5-opus": createModelConfig(
    "claude-4.5-opus",
    "claude_estimation",
    32768,
    "claude",
    200_000,
  ),
  "claude-4-5-sonnet": createModelConfig(
    "claude-4-5-sonnet",
    "claude_estimation",
    16384,
    "claude",
    200_000,
  ),
  "claude-4.5-sonnet": createModelConfig(
    "claude-4.5-sonnet",
    "claude_estimation",
    16384,
    "claude",
    200_000,
  ),
  "claude-4-5-haiku": createModelConfig(
    "claude-4-5-haiku",
    "claude_estimation",
    8192,
    "claude",
    200_000,
  ),
  "claude-4.5-haiku": createModelConfig(
    "claude-4.5-haiku",
    "claude_estimation",
    8192,
    "claude",
    200_000,
  ),

  // Claude 4.1 Series
  "claude-4-1-opus": createModelConfig(
    "claude-4-1-opus",
    "claude_estimation",
    32768,
    "claude",
    200_000,
  ),
  "claude-4.1-opus": createModelConfig(
    "claude-4.1-opus",
    "claude_estimation",
    32768,
    "claude",
    200_000,
  ),
  "claude-4-1-sonnet": createModelConfig(
    "claude-4-1-sonnet",
    "claude_estimation",
    16384,
    "claude",
    200_000,
  ),
  "claude-4.1-sonnet": createModelConfig(
    "claude-4.1-sonnet",
    "claude_estimation",
    16384,
    "claude",
    200_000,
  ),

  // Claude 4 Series
  "claude-4-opus": createModelConfig(
    "claude-4-opus",
    "claude_estimation",
    32768,
    "claude",
    200_000,
  ),
  "claude-4-sonnet": createModelConfig(
    "claude-4-sonnet",
    "claude_estimation",
    16384,
    "claude",
    200_000,
  ),

  // Claude 3.5 Series
  "claude-3-5-sonnet-20241022": createModelConfig(
    "claude-3-5-sonnet-20241022",
    "claude_estimation",
    8192,
    "claude",
    200_000,
  ),
  "claude-3-5-sonnet-latest": createModelConfig(
    "claude-3-5-sonnet-latest",
    "claude_estimation",
    8192,
    "claude",
    200_000,
  ),
  "claude-3-5-sonnet": createModelConfig(
    "claude-3-5-sonnet",
    "claude_estimation",
    8192,
    "claude",
    200_000,
  ),
  "claude-3-5-haiku-20241022": createModelConfig(
    "claude-3-5-haiku-20241022",
    "claude_estimation",
    8192,
    "claude",
    200_000,
  ),
  "claude-3-5-haiku-latest": createModelConfig(
    "claude-3-5-haiku-latest",
    "claude_estimation",
    8192,
    "claude",
    200_000,
  ),
  "claude-3-5-haiku": createModelConfig(
    "claude-3-5-haiku",
    "claude_estimation",
    8192,
    "claude",
    200_000,
  ),

  // Claude 3 Series
  "claude-3-opus-20240229": createModelConfig(
    "claude-3-opus-20240229",
    "claude_estimation",
    4096,
    "claude",
    200_000,
  ),
  "claude-3-opus-latest": createModelConfig(
    "claude-3-opus-latest",
    "claude_estimation",
    4096,
    "claude",
    200_000,
  ),
  "claude-3-opus": createModelConfig(
    "claude-3-opus",
    "claude_estimation",
    4096,
    "claude",
    200_000,
  ),
  "claude-3-sonnet-20240229": createModelConfig(
    "claude-3-sonnet-20240229",
    "claude_estimation",
    4096,
    "claude",
    200_000,
  ),
  "claude-3-sonnet": createModelConfig(
    "claude-3-sonnet",
    "claude_estimation",
    4096,
    "claude",
    200_000,
  ),
  "claude-3-haiku-20240307": createModelConfig(
    "claude-3-haiku-20240307",
    "claude_estimation",
    4096,
    "claude",
    200_000,
  ),
  "claude-3-haiku": createModelConfig(
    "claude-3-haiku",
    "claude_estimation",
    4096,
    "claude",
    200_000,
  ),

  // Claude 2 Series (Legacy)
  "claude-2.1": createModelConfig(
    "claude-2.1",
    "claude_estimation",
    4096,
    "claude",
    200_000,
  ),
  "claude-2.0": createModelConfig(
    "claude-2.0",
    "claude_estimation",
    4096,
    "claude",
    100_000,
  ),

  // Claude Instant (Legacy)
  "claude-instant-1.2": createModelConfig(
    "claude-instant-1.2",
    "claude_estimation",
    4096,
    "claude",
    100_000,
  ),

  // ==========================================================================
  // DeepSeek Models (approximation with cl100k_base)
  // ==========================================================================
  "deepseek-chat": createModelConfig(
    "deepseek-chat",
    "cl100k_base",
    4096,
    "deepseek",
    32_768,
  ),
  "deepseek-coder": createModelConfig(
    "deepseek-coder",
    "cl100k_base",
    4096,
    "deepseek",
    16_384,
  ),
  "deepseek-reasoner": createModelConfig(
    "deepseek-reasoner",
    "cl100k_base",
    8192,
    "deepseek",
    65_536,
  ),

  // ==========================================================================
  // Gemini Models (approximation with cl100k_base)
  // ==========================================================================
  "gemini-1.5-pro": createModelConfig(
    "gemini-1.5-pro",
    "cl100k_base",
    8192,
    "gemini",
    1_000_000,
  ),
  "gemini-1.5-flash": createModelConfig(
    "gemini-1.5-flash",
    "cl100k_base",
    8192,
    "gemini",
    1_000_000,
  ),
  "gemini-2.0-flash": createModelConfig(
    "gemini-2.0-flash",
    "cl100k_base",
    8192,
    "gemini",
    1_000_000,
  ),
  "gemini-2.0-flash-thinking": createModelConfig(
    "gemini-2.0-flash-thinking",
    "cl100k_base",
    16384,
    "gemini",
    1_000_000,
  ),
};

/**
 * Model name aliases for flexibility
 */
export const MODEL_ALIASES: Record<string, string> = {
  // Common aliases
  gpt4: "gpt-4",
  gpt4o: "gpt-4o",
  "gpt4o-mini": "gpt-4o-mini",
  "gpt-4-omni": "gpt-4o",
  "gpt-4-omni-mini": "gpt-4o-mini",

  // GPT-5 aliases
  gpt5: "gpt-5",
  "gpt5-mini": "gpt-5-mini",
  "gpt5-nano": "gpt-5-nano",

  // GPT-3.5 aliases
  "gpt-35-turbo": "gpt-3.5-turbo",
  gpt35: "gpt-3.5-turbo",
  chatgpt: "gpt-3.5-turbo",

  // DeepSeek aliases
  deepseek: "deepseek-chat",
  "deepseek-v3": "deepseek-chat",

  // Gemini aliases
  gemini: "gemini-1.5-flash",
  "gemini-pro": "gemini-1.5-pro",
  "gemini-flash": "gemini-1.5-flash",

  // Claude aliases - default to latest versions
  claude: "claude-4-5-sonnet",
  "claude-latest": "claude-4-5-sonnet",
  "claude-sonnet": "claude-4-5-sonnet",
  "claude-haiku": "claude-4-5-haiku",
  "claude-opus": "claude-4-5-opus",

  // Claude 4.5 aliases
  "claude4.5": "claude-4-5-sonnet",
  "claude-45": "claude-4-5-sonnet",
  "claude-45-sonnet": "claude-4-5-sonnet",
  "claude-45-opus": "claude-4-5-opus",
  "claude-45-haiku": "claude-4-5-haiku",

  // Claude 4.1 aliases
  "claude4.1": "claude-4-1-sonnet",
  "claude-41": "claude-4-1-sonnet",
  "claude-41-sonnet": "claude-4-1-sonnet",
  "claude-41-opus": "claude-4-1-opus",

  // Claude 4 aliases
  claude4: "claude-4-sonnet",
  "claude-4": "claude-4-sonnet",

  // Claude 3.5 aliases
  "claude3.5": "claude-3-5-sonnet",
  "claude-3.5": "claude-3-5-sonnet",
  "claude-35-sonnet": "claude-3-5-sonnet",
  "claude-35-haiku": "claude-3-5-haiku",

  // Claude 3 aliases
  claude3: "claude-3-opus",
  "claude-3": "claude-3-opus",

  // Legacy aliases
  "claude-instant": "claude-instant-1.2",
  "claude-2": "claude-2.1",
};

/**
 * Get model configuration by name
 * Handles aliases and partial matches
 *
 * @param modelName - Model name or alias
 * @returns Model configuration or undefined
 */
export function getModelConfig(modelName: string): ModelConfig | undefined {
  const normalizedName = modelName.toLowerCase().trim();

  // Fine-tuned models resolve via their base model
  if (normalizedName.startsWith("ft:")) {
    const rest = normalizedName.slice(3);
    return getModelConfig(rest.split(":")[0] ?? rest);
  }

  // Direct match
  if (MODEL_CONFIGS[normalizedName]) {
    return MODEL_CONFIGS[normalizedName];
  }

  // Check aliases
  const aliasedName = MODEL_ALIASES[normalizedName];
  if (aliasedName && MODEL_CONFIGS[aliasedName]) {
    return MODEL_CONFIGS[aliasedName];
  }

  // Truncated names: first model the input is a prefix of
  for (const [key, config] of Object.entries(MODEL_CONFIGS)) {
    if (key.startsWith(normalizedName)) {
      return config;
    }
  }

  // Versioned names (gpt-4-0613, o1-mini-2024-09-12): longest known prefix wins
  let bestKey = "";
  for (const key of Object.keys(MODEL_CONFIGS)) {
    if (normalizedName.startsWith(key) && key.length > bestKey.length) {
      bestKey = key;
    }
  }
  return bestKey ? MODEL_CONFIGS[bestKey] : undefined;
}

/**
 * Get encoding name for a model
 *
 * @param modelName - Model name
 * @returns Encoding name, defaults to o200k_base for unknown models
 */
export function getEncodingForModel(modelName: string): EncodingName {
  const config = getModelConfig(modelName);
  return config?.encoding ?? "o200k_base";
}

/**
 * Get context limit for a model
 *
 * @param modelName - Model name
 * @returns Context limit in tokens
 */
export function getModelContextLimit(modelName: string): number {
  const config = getModelConfig(modelName);
  // Unconfigured families with a known window (e.g. gpt-6) before the GPT-4o-class default
  return config?.contextLimit ?? findContextSize(modelName) ?? 128000;
}

/**
 * Get max output tokens for a model
 *
 * @param modelName - Model name
 * @returns Max output tokens
 */
export function getModelMaxOutputTokens(modelName: string): number {
  const config = getModelConfig(modelName);
  return config?.maxOutputTokens ?? 4096; // Default to standard output limit
}

/**
 * Get model family
 *
 * @param modelName - Model name
 * @returns Model family classification
 */
export function getModelFamily(modelName: string): ModelFamily {
  const config = getModelConfig(modelName);
  return config?.family ?? "unknown";
}

/**
 * Check if a model uses the newer o200k_base encoding
 *
 * @param modelName - Model name
 * @returns True if model uses o200k_base
 */
export function usesO200kEncoding(modelName: string): boolean {
  return getEncodingForModel(modelName) === "o200k_base";
}

/**
 * Check if a model uses the Claude estimation encoding
 *
 * Claude models use a proprietary tokenizer. This encoding provides
 * "safe" estimates that intentionally over-count tokens to prevent
 * API truncation issues.
 *
 * @param modelName - Model name
 * @returns True if model uses claude_estimation
 */
export function usesClaudeEstimation(modelName: string): boolean {
  return getEncodingForModel(modelName) === "claude_estimation";
}

/**
 * Check if a model is a Claude model
 *
 * @param modelName - Model name
 * @returns True if model is from Anthropic Claude family
 */
export function isClaudeModel(modelName: string): boolean {
  const config = getModelConfig(modelName);
  return config?.family === "claude";
}

/**
 * List all known models
 *
 * @returns Array of model names
 */
export function listModels(): string[] {
  return Object.keys(MODEL_CONFIGS);
}

/**
 * List models by family
 *
 * @param family - Model family to filter
 * @returns Array of model configurations
 */
export function listModelsByFamily(family: ModelFamily): ModelConfig[] {
  return Object.values(MODEL_CONFIGS).filter(
    (config) => config.family === family,
  );
}
