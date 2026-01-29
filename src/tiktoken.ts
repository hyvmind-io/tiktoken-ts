/**
 * High-Level Tiktoken API
 *
 * Provides easy-to-use tokenization APIs with automatic vocabulary loading.
 *
 * This module is the main entry point for most users. It provides:
 * - Lazy loading of vocabularies
 * - Caching of tokenizer instances
 * - Support for both sync and async operations
 * - Model name resolution
 */

import {
  CoreBPE,
  type Rank,
  type Vocabulary,
  type ReverseVocabulary,
  getEncodingDefinition,
  type EncodingDefinition,
  loadVocabularyFromUrl,
  loadVocabularyFromString,
  getVocabularyFromCache,
  isVocabularyCached,
  getTokenizerForModel,
} from "./core/index.js";

/**
 * Tiktoken encoding instance
 *
 * This class wraps CoreBPE and provides:
 * - Lazy initialization
 * - Convenient API matching tiktoken-rs
 * - Error handling for unloaded vocabularies
 */
export class Tiktoken {
  /** The encoding name */
  public readonly name: string;

  /** The encoding definition */
  private readonly definition: EncodingDefinition;

  /** The underlying CoreBPE instance (lazy loaded) */
  private coreBPE: CoreBPE | null = null;

  /** Promise for async initialization */
  private initPromise: Promise<void> | null = null;

  /** Whether the vocabulary has been loaded */
  private isLoaded = false;

  /**
   * Create a new Tiktoken instance
   *
   * Note: The vocabulary is NOT loaded until you call load() or one of the
   * encoding methods with a loaded vocabulary.
   *
   * @param encodingName - The encoding name (e.g., "cl100k_base", "o200k_base")
   */
  constructor(encodingName: string) {
    const definition = getEncodingDefinition(encodingName);
    if (!definition) {
      throw new Error(`Unknown encoding: ${encodingName}`);
    }

    this.name = encodingName;
    this.definition = definition;

    // Check if vocabulary is already cached
    if (isVocabularyCached(encodingName)) {
      this.initFromCache();
    }
  }

  /**
   * Initialize from cached vocabulary
   */
  private initFromCache(): void {
    const vocab = getVocabularyFromCache(this.name);
    if (vocab) {
      this.initCoreBPE(vocab.encoder, vocab.decoder);
    }
  }

  /**
   * Initialize the CoreBPE instance
   */
  private initCoreBPE(encoder: Vocabulary, _decoder: ReverseVocabulary): void {
    // Note: decoder is provided by vocab loader but CoreBPE builds its own from encoder
    // The _decoder parameter is kept for API consistency with vocabulary loading

    // Create special tokens map
    const specialTokensEncoder = new Map<string, Rank>();
    for (const [token, rank] of Object.entries(this.definition.specialTokens)) {
      specialTokensEncoder.set(token, rank);
    }

    this.coreBPE = new CoreBPE(
      encoder,
      specialTokensEncoder,
      this.definition.pattern,
    );
    this.isLoaded = true;
  }

  /**
   * Load the vocabulary from URL
   *
   * @returns Promise that resolves when loaded
   */
  async load(): Promise<void> {
    if (this.isLoaded) {
      return;
    }

    if (this.initPromise) {
      return this.initPromise;
    }

    this.initPromise = (async () => {
      const vocab = await loadVocabularyFromUrl(this.name);
      this.initCoreBPE(vocab.encoder, vocab.decoder); // decoder used internally in initCoreBPE
    })();

    return this.initPromise;
  }

  /**
   * Load the vocabulary from a string (for embedded vocabularies)
   *
   * @param content - The vocabulary file content
   */
  loadFromString(content: string): void {
    if (this.isLoaded) {
      return;
    }

    const vocab = loadVocabularyFromString(this.name, content);
    this.initCoreBPE(vocab.encoder, vocab.decoder);
  }

  /**
   * Ensure the vocabulary is loaded
   */
  private ensureLoaded(): CoreBPE {
    if (!this.coreBPE) {
      throw new Error(
        `Vocabulary not loaded for ${this.name}. Call load() first or use the async API.`,
      );
    }
    return this.coreBPE;
  }

  /**
   * Check if the vocabulary is loaded
   */
  get loaded(): boolean {
    return this.isLoaded;
  }

  // =========================================================================
  // Encoding Methods
  // =========================================================================

  /**
   * Encode text into token IDs without handling special tokens
   *
   * @param text - Text to encode
   * @returns Array of token IDs
   */
  encodeOrdinary(text: string): Rank[] {
    return this.ensureLoaded().encodeOrdinary(text);
  }

  /**
   * Encode text into token IDs
   *
   * @param text - Text to encode
   * @param allowedSpecial - Special tokens to allow (default: none)
   * @returns Array of token IDs
   */
  encode(text: string, allowedSpecial?: Set<string> | "all"): Rank[] {
    const bpe = this.ensureLoaded();

    if (allowedSpecial === "all") {
      return bpe.encodeWithSpecialTokens(text);
    }

    const allowed = allowedSpecial ?? new Set<string>();
    return bpe.encode(text, allowed)[0];
  }

  /**
   * Encode text with all special tokens allowed
   *
   * @param text - Text to encode
   * @returns Array of token IDs
   */
  encodeWithSpecialTokens(text: string): Rank[] {
    return this.ensureLoaded().encodeWithSpecialTokens(text);
  }

  // =========================================================================
  // Decoding Methods
  // =========================================================================

  /**
   * Decode token IDs back to text
   *
   * @param tokens - Array of token IDs
   * @returns Decoded text
   */
  decode(tokens: Rank[]): string {
    return this.ensureLoaded().decode(tokens);
  }

  /**
   * Decode token IDs to raw bytes
   *
   * @param tokens - Array of token IDs
   * @returns Decoded bytes
   */
  decodeBytes(tokens: Rank[]): Uint8Array {
    return this.ensureLoaded().decodeBytes(tokens);
  }

  // =========================================================================
  // Utility Methods
  // =========================================================================

  /**
   * Count tokens in text (without returning the tokens)
   *
   * @param text - Text to count
   * @returns Number of tokens
   */
  countTokens(text: string): number {
    return this.encodeOrdinary(text).length;
  }

  /**
   * Get the vocabulary size (excluding special tokens)
   */
  get vocabSize(): number {
    return this.ensureLoaded().vocabSize;
  }

  /**
   * Get the total vocabulary size (including special tokens)
   */
  get totalVocabSize(): number {
    return this.ensureLoaded().totalVocabSize;
  }

  /**
   * Get all special tokens
   */
  getSpecialTokens(): Set<string> {
    return this.ensureLoaded().getSpecialTokens();
  }

  /**
   * Check if a token is a special token
   */
  isSpecialToken(token: Rank): boolean {
    return this.ensureLoaded().isSpecialToken(token);
  }

  /**
   * Get the definition for this encoding
   */
  getDefinition(): EncodingDefinition {
    return this.definition;
  }
}

// =========================================================================
// Factory Functions
// =========================================================================

/**
 * Cache of Tiktoken instances
 */
const tiktokenCache: Map<string, Tiktoken> = new Map();

/**
 * Get or create a Tiktoken instance for an encoding
 *
 * @param encodingName - Encoding name
 * @returns Tiktoken instance (may not be loaded yet)
 */
export function getEncoding(encodingName: string): Tiktoken {
  let instance = tiktokenCache.get(encodingName);

  if (!instance) {
    instance = new Tiktoken(encodingName);
    tiktokenCache.set(encodingName, instance);
  }

  return instance;
}

/**
 * Get or create a Tiktoken instance for a model
 *
 * @param modelName - Model name
 * @returns Tiktoken instance (may not be loaded yet)
 * @throws Error if no encoding is found for the model
 */
export function getEncodingForModel(modelName: string): Tiktoken {
  const encoding = getTokenizerForModel(modelName);
  if (!encoding) {
    // Default to o200k_base for unknown models (most modern models use this)
    return getEncoding("o200k_base");
  }
  return getEncoding(encoding);
}

/**
 * Get a loaded Tiktoken instance for an encoding (async)
 *
 * @param encodingName - Encoding name
 * @returns Promise resolving to a loaded Tiktoken instance
 */
export async function getEncodingAsync(
  encodingName: string,
): Promise<Tiktoken> {
  const instance = getEncoding(encodingName);
  await instance.load();
  return instance;
}

/**
 * Get a loaded Tiktoken instance for a model (async)
 *
 * @param modelName - Model name
 * @returns Promise resolving to a loaded Tiktoken instance
 */
export async function getEncodingForModelAsync(
  modelName: string,
): Promise<Tiktoken> {
  const instance = getEncodingForModel(modelName);
  await instance.load();
  return instance;
}

/**
 * Clear the Tiktoken instance cache
 */
export function clearTiktokenCache(): void {
  tiktokenCache.clear();
}

// =========================================================================
// Convenience Functions
// =========================================================================

/**
 * Encode text using a specific encoding (async)
 *
 * @param text - Text to encode
 * @param encodingName - Encoding name
 * @returns Promise resolving to token IDs
 */
export async function encodeAsync(
  text: string,
  encodingName: string = "o200k_base",
): Promise<Rank[]> {
  const tiktoken = await getEncodingAsync(encodingName);
  return tiktoken.encodeOrdinary(text);
}

/**
 * Decode tokens using a specific encoding (async)
 *
 * @param tokens - Token IDs
 * @param encodingName - Encoding name
 * @returns Promise resolving to decoded text
 */
export async function decodeAsync(
  tokens: Rank[],
  encodingName: string = "o200k_base",
): Promise<string> {
  const tiktoken = await getEncodingAsync(encodingName);
  return tiktoken.decode(tokens);
}

/**
 * Count tokens in text (async)
 *
 * @param text - Text to count
 * @param encodingName - Encoding name
 * @returns Promise resolving to token count
 */
export async function countTokensAsync(
  text: string,
  encodingName: string = "o200k_base",
): Promise<number> {
  const tiktoken = await getEncodingAsync(encodingName);
  return tiktoken.countTokens(text);
}

/**
 * Count tokens for a model (async)
 *
 * @param text - Text to count
 * @param modelName - Model name
 * @returns Promise resolving to token count
 */
export async function countTokensForModelAsync(
  text: string,
  modelName: string,
): Promise<number> {
  const tiktoken = await getEncodingForModelAsync(modelName);
  return tiktoken.countTokens(text);
}
