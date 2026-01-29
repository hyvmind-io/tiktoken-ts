/**
 * Claude Estimation Encoding
 *
 * IMPORTANT: This is an ESTIMATION encoding, not an exact BPE tokenizer.
 *
 * Claude (Anthropic) uses a proprietary tokenizer that is NOT publicly available.
 * This encoding provides "safe" estimates that intentionally over-count tokens
 * to prevent API truncation issues when working with Claude models.
 *
 * Based on research findings (see claude-tokenizer-research.md):
 * - Claude 3+ uses ~22,000 token vocabulary (much smaller than OpenAI's 100K-200K)
 * - Claude produces 16-30% MORE tokens than GPT-4 for equivalent content:
 *   - English articles: +16%
 *   - Math equations: +21%
 *   - Python code: +30%
 * - Average ~3.5 characters per token (vs GPT-4's ~4)
 *
 * This encoding applies a 1.25x safety multiplier to ensure estimates err on
 * the side of over-counting, preventing API truncation.
 *
 * For EXACT Claude token counts, use Anthropic's official API:
 * @see https://docs.anthropic.com/en/docs/build-with-claude/token-counting
 */

import { BPETokenizer, CLAUDE_SAFETY_MULTIPLIER } from "../bpe.js";
import type { Tokenizer, EncodingName } from "../types.js";

/**
 * Encoding name constant
 */
export const ENCODING_NAME: EncodingName = "claude_estimation";

/**
 * Create a Claude estimation tokenizer instance
 *
 * This tokenizer provides SAFE estimates for Claude models by:
 * 1. Using cl100k_base patterns as a base (70% vocabulary overlap with Claude 1/2)
 * 2. Applying a 1.25x safety multiplier to account for Claude's higher token usage
 *
 * The estimates intentionally err on over-counting to prevent API truncation.
 *
 * @returns Tokenizer instance configured for Claude estimation
 *
 * @example
 * ```typescript
 * import { createClaudeEstimationTokenizer } from "tiktoken-ts";
 *
 * const tokenizer = createClaudeEstimationTokenizer();
 * const count = tokenizer.countTokens("Hello, Claude!");
 * // Returns a SAFE estimate (intentionally higher than actual)
 * ```
 */
export function createClaudeEstimationTokenizer(): Tokenizer {
  const bpe = new BPETokenizer(ENCODING_NAME);

  return {
    encodingName: ENCODING_NAME,
    encode: (text: string) => bpe.encode(text),
    decode: (tokens: number[]) => bpe.decode(tokens),
    countTokens: (text: string) => bpe.countTokens(text),
  };
}

/**
 * Average characters per token for Claude models.
 *
 * Research indicates Claude averages ~3.5 characters per token,
 * which is lower (less efficient) than GPT-4's ~4 characters per token.
 */
export const AVERAGE_CHARS_PER_TOKEN = 3.5;

/**
 * Estimated vocabulary size for Claude 3+.
 *
 * Based on reverse-engineering research by Sander Land, Claude 3's
 * vocabulary is estimated at ~22,000 tokens - remarkably small compared
 * to Mistral's 32K, GPT-4's 100K, or LLaMA 3's 128K.
 */
export const ESTIMATED_VOCAB_SIZE = 22000;

/**
 * Safety multiplier applied to token estimates.
 *
 * This multiplier ensures estimates err on over-counting to prevent
 * API truncation. Based on research showing Claude produces 16-30%
 * more tokens than GPT-4:
 * - English: +16%
 * - Math: +21%
 * - Code: +30%
 *
 * We use 1.25 (25%) as a safe middle ground.
 */
export const SAFETY_MULTIPLIER = CLAUDE_SAFETY_MULTIPLIER;

/**
 * Token estimation adjustments for different content types.
 *
 * Claude's tokenizer has different efficiency characteristics than OpenAI's:
 * - Generally less efficient (more tokens per character)
 * - Especially less efficient for code (+30% vs GPT-4)
 * - CJK handling varies by input vs output context
 *
 * These multipliers are applied ON TOP of the base safety multiplier.
 */
export const CONTENT_TYPE_MULTIPLIERS: Record<string, number> = {
  prose: 1.0, // Base English text
  code: 1.1, // Code is particularly inefficient on Claude (+30% vs GPT-4)
  json: 1.0, // JSON follows prose patterns
  markdown: 1.0, // Markdown similar to prose
  html: 0.95, // HTML tags slightly more efficient
  math: 1.1, // Math equations use more tokens on Claude (+21% vs GPT-4)
};

/**
 * Get estimated token count with content-type awareness for Claude.
 *
 * The returned estimate includes both:
 * 1. The base safety multiplier (1.25x)
 * 2. Content-type specific adjustments
 *
 * @param text - Input text
 * @param contentType - Type of content (affects estimation accuracy)
 * @returns Safe estimated token count (intentionally over-counts)
 *
 * @example
 * ```typescript
 * import { estimateClaudeTokens } from "tiktoken-ts";
 *
 * // For English prose
 * const count = estimateClaudeTokens("Hello, world!", "prose");
 *
 * // For Python code (applies additional code multiplier)
 * const codeCount = estimateClaudeTokens(pythonCode, "code");
 * ```
 */
export function estimateClaudeTokens(
  text: string,
  contentType: keyof typeof CONTENT_TYPE_MULTIPLIERS = "prose",
): number {
  const tokenizer = createClaudeEstimationTokenizer();
  const baseCount = tokenizer.countTokens(text); // Already includes safety multiplier
  const contentMultiplier = CONTENT_TYPE_MULTIPLIERS[contentType] ?? 1.0;

  return Math.max(1, Math.ceil(baseCount * contentMultiplier));
}

/**
 * Get a CONSERVATIVE (high) token estimate for Claude.
 *
 * This function returns the highest reasonable estimate, suitable for:
 * - Calculating worst-case API costs
 * - Ensuring prompts fit within context limits
 * - Setting very safe max_tokens values
 *
 * It applies the maximum content multiplier (1.1 for code/math) on top
 * of the base safety multiplier (1.25x), resulting in ~1.375x over-estimation.
 *
 * @param text - Input text
 * @returns Conservative (high) token estimate
 */
export function estimateClaudeTokensConservative(text: string): number {
  return estimateClaudeTokens(text, "code"); // Uses highest multiplier
}

/**
 * Check if Claude token estimation applies safety multiplier.
 *
 * This is a convenience function that always returns true, confirming
 * that Claude estimates include a safety margin.
 *
 * @returns Always true (Claude estimation always includes safety multiplier)
 */
export function includesSafetyMultiplier(): boolean {
  return true;
}
