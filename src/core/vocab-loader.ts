/**
 * Vocabulary Loader
 *
 * Handles loading and parsing of .tiktoken vocabulary files.
 * These files contain base64-encoded tokens with their ranks.
 *
 * Format: Each line is "base64_token rank"
 * Example: "SGVsbG8= 12345" means "Hello" has rank 12345
 */

import {
  type Vocabulary,
  type ReverseVocabulary,
  type Rank,
  bytesToKey,
  keyToBytes,
} from "./byte-pair-encoding.js";

// Re-export for convenience
export { keyToBytes };

/**
 * Base64 decode (works in both Node.js and browser)
 */
function base64Decode(base64: string): Uint8Array {
  // Use built-in atob for browser compatibility
  // Node.js 18+ also supports atob globally
  if (typeof atob === "function") {
    const binaryString = atob(base64);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }
    return bytes;
  }

  // Fallback for older Node.js (should not be needed with Node 18+)
  const buffer = Buffer.from(base64, "base64");
  return new Uint8Array(buffer);
}

/**
 * Base64 encode (works in both Node.js and browser)
 */
export function base64Encode(bytes: Uint8Array): string {
  if (typeof btoa === "function") {
    let binaryString = "";
    for (let i = 0; i < bytes.length; i++) {
      binaryString += String.fromCharCode(bytes[i]);
    }
    return btoa(binaryString);
  }

  // Fallback for older Node.js
  return Buffer.from(bytes).toString("base64");
}

/**
 * Parse a .tiktoken vocabulary file content
 *
 * @param content - The raw content of the .tiktoken file
 * @returns Vocabulary map and reverse vocabulary map
 */
export function parseVocabulary(content: string): {
  encoder: Vocabulary;
  decoder: ReverseVocabulary;
} {
  const encoder: Vocabulary = new Map();
  const decoder: ReverseVocabulary = new Map();

  const lines = content.split("\n");

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) continue;

    const spaceIndex = trimmed.indexOf(" ");
    if (spaceIndex === -1) continue;

    const base64Token = trimmed.slice(0, spaceIndex);
    const rankStr = trimmed.slice(spaceIndex + 1);

    try {
      const tokenBytes = base64Decode(base64Token);
      const rank = parseInt(rankStr, 10);

      if (!isNaN(rank)) {
        const key = bytesToKey(tokenBytes);
        encoder.set(key, rank);
        decoder.set(rank, tokenBytes);
      }
    } catch {
      // Skip malformed lines
      continue;
    }
  }

  return { encoder, decoder };
}

/**
 * Vocabulary URLs for each encoding
 */
export const VOCABULARY_URLS: Record<string, string> = {
  r50k_base:
    "https://openaipublic.blob.core.windows.net/encodings/r50k_base.tiktoken",
  p50k_base:
    "https://openaipublic.blob.core.windows.net/encodings/p50k_base.tiktoken",
  p50k_edit:
    "https://openaipublic.blob.core.windows.net/encodings/p50k_base.tiktoken", // Uses same vocab
  cl100k_base:
    "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
  o200k_base:
    "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken",
  o200k_harmony:
    "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken", // Uses same vocab
};

/**
 * Cache for loaded vocabularies
 */
const vocabularyCache: Map<
  string,
  { encoder: Vocabulary; decoder: ReverseVocabulary }
> = new Map();

/**
 * Load vocabulary from URL (with caching)
 *
 * @param encodingName - Name of the encoding
 * @returns Promise resolving to vocabulary maps
 */
export async function loadVocabularyFromUrl(
  encodingName: string,
): Promise<{ encoder: Vocabulary; decoder: ReverseVocabulary }> {
  // Check cache first
  const cached = vocabularyCache.get(encodingName);
  if (cached) {
    return cached;
  }

  const url = VOCABULARY_URLS[encodingName];
  if (!url) {
    throw new Error(`Unknown encoding: ${encodingName}`);
  }

  // Fetch the vocabulary file
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(
      `Failed to fetch vocabulary for ${encodingName}: ${response.statusText}`,
    );
  }

  const content = await response.text();
  const vocab = parseVocabulary(content);

  // Cache the result
  vocabularyCache.set(encodingName, vocab);

  return vocab;
}

/**
 * Load vocabulary from a string (for embedded vocabularies)
 *
 * @param encodingName - Name of the encoding (for caching)
 * @param content - The vocabulary file content
 * @returns Vocabulary maps
 */
export function loadVocabularyFromString(
  encodingName: string,
  content: string,
): { encoder: Vocabulary; decoder: ReverseVocabulary } {
  // Check cache first
  const cached = vocabularyCache.get(encodingName);
  if (cached) {
    return cached;
  }

  const vocab = parseVocabulary(content);

  // Cache the result
  vocabularyCache.set(encodingName, vocab);

  return vocab;
}

/**
 * Create special tokens map
 *
 * @param specialTokens - Map of special token strings to ranks
 * @returns Maps for encoding and decoding special tokens
 */
export function createSpecialTokenMaps(specialTokens: Record<string, Rank>): {
  encoder: Map<string, Rank>;
  decoder: Map<Rank, Uint8Array>;
} {
  const encoder = new Map<string, Rank>();
  const decoder = new Map<Rank, Uint8Array>();

  for (const [token, rank] of Object.entries(specialTokens)) {
    encoder.set(token, rank);
    // Convert string to bytes for decoder
    const textEncoder = new TextEncoder();
    decoder.set(rank, textEncoder.encode(token));
  }

  return { encoder, decoder };
}

/**
 * Clear the vocabulary cache
 * Useful for testing or memory management
 */
export function clearVocabularyCache(): void {
  vocabularyCache.clear();
}

/**
 * Get vocabulary from cache (sync, returns undefined if not loaded)
 */
export function getVocabularyFromCache(
  encodingName: string,
): { encoder: Vocabulary; decoder: ReverseVocabulary } | undefined {
  return vocabularyCache.get(encodingName);
}

/**
 * Check if a vocabulary is cached
 */
export function isVocabularyCached(encodingName: string): boolean {
  return vocabularyCache.has(encodingName);
}
