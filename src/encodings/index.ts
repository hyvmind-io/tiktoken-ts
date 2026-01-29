/**
 * Encoding Exports
 * Central export for all encoding implementations
 */

// cl100k_base (GPT-4, GPT-3.5-turbo)
export {
  createCL100kTokenizer,
  ENCODING_NAME as CL100K_ENCODING_NAME,
  COMMON_TOKEN_PATTERNS as CL100K_TOKEN_PATTERNS,
  AVERAGE_CHARS_PER_TOKEN as CL100K_AVG_CHARS,
  CONTENT_TYPE_MULTIPLIERS as CL100K_CONTENT_MULTIPLIERS,
  estimateTokensForContent as estimateCL100kTokens,
} from "./cl100k-base.js";

// o200k_base (GPT-4o, GPT-4.1, GPT-5)
export {
  createO200kTokenizer,
  ENCODING_NAME as O200K_ENCODING_NAME,
  COMMON_TOKEN_PATTERNS as O200K_TOKEN_PATTERNS,
  AVERAGE_CHARS_PER_TOKEN as O200K_AVG_CHARS,
  CONTENT_TYPE_MULTIPLIERS as O200K_CONTENT_MULTIPLIERS,
  LANGUAGE_EFFICIENCY as O200K_LANGUAGE_EFFICIENCY,
  estimateTokensForContent as estimateO200kTokens,
  estimateTokensForLanguage as estimateO200kTokensForLanguage,
} from "./o200k-base.js";

// p50k_base (Legacy Codex models)
export {
  createP50kTokenizer,
  ENCODING_NAME as P50K_ENCODING_NAME,
  AVERAGE_CHARS_PER_TOKEN as P50K_AVG_CHARS,
  CONTENT_TYPE_MULTIPLIERS as P50K_CONTENT_MULTIPLIERS,
  estimateTokensForContent as estimateP50kTokens,
} from "./p50k-base.js";

// claude_estimation (Anthropic Claude models - estimation only)
export {
  createClaudeEstimationTokenizer,
  ENCODING_NAME as CLAUDE_ENCODING_NAME,
  AVERAGE_CHARS_PER_TOKEN as CLAUDE_AVG_CHARS,
  ESTIMATED_VOCAB_SIZE as CLAUDE_VOCAB_SIZE,
  SAFETY_MULTIPLIER as CLAUDE_SAFETY_MULT,
  CONTENT_TYPE_MULTIPLIERS as CLAUDE_CONTENT_MULTIPLIERS,
  estimateClaudeTokens,
  estimateClaudeTokensConservative,
  includesSafetyMultiplier as claudeIncludesSafetyMultiplier,
} from "./claude-estimation.js";
