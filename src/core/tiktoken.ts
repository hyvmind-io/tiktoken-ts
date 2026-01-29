/**
 * Core Tiktoken BPE Implementation
 *
 * This is an EXACT port of the CoreBPE struct from tiktoken-rs.
 * Reference: https://github.com/zurawiki/tiktoken-rs/blob/main/tiktoken-rs/src/vendor_tiktoken.rs
 *
 * Provides the main tokenization API:
 * - encode_ordinary(text) - Encode without special tokens
 * - encode(text, allowed_special) - Encode with optional special tokens
 * - encode_with_special_tokens(text) - Encode with all special tokens
 * - decode(tokens) - Decode tokens to text
 * - decode_bytes(tokens) - Decode tokens to raw bytes
 */

import {
  type Vocabulary,
  type ReverseVocabulary,
  type Rank,
  bytePairEncode,
  bytesToKey,
  keyToBytes,
} from "./byte-pair-encoding.js";

/**
 * Error thrown when a token cannot be decoded
 */
export class DecodeKeyError extends Error {
  public readonly token: Rank;

  constructor(token: Rank) {
    super(`Invalid token for decoding: ${token}`);
    this.name = "DecodeKeyError";
    this.token = token;
  }
}

/**
 * Escape a string for use in a regex
 */
function escapeRegex(str: string): string {
  return str.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/**
 * CoreBPE - The main tokenizer class
 *
 * This class implements the complete BPE tokenization algorithm,
 * matching the behavior of tiktoken-rs exactly.
 */
export class CoreBPE {
  /** The vocabulary mapping byte sequences to ranks */
  private readonly encoder: Vocabulary;

  /** The special tokens mapping strings to ranks */
  private readonly specialTokensEncoder: Map<string, Rank>;

  /** Reverse vocabulary mapping ranks to byte sequences */
  private readonly decoder: ReverseVocabulary;

  /** Reverse special tokens mapping ranks to byte sequences */
  private readonly specialTokensDecoder: Map<Rank, Uint8Array>;

  /** Compiled regex for splitting text */
  private readonly regex: RegExp;

  /** Compiled regex for finding special tokens */
  private readonly specialRegex: RegExp | null;

  /** Text encoder for converting strings to bytes */
  private readonly textEncoder: InstanceType<typeof TextEncoder>;

  /** Text decoder for converting bytes to strings */
  private readonly textDecoder: InstanceType<typeof TextDecoder>;

  /**
   * Create a new CoreBPE instance
   *
   * @param encoder - Vocabulary mapping byte sequences to ranks
   * @param specialTokensEncoder - Special tokens mapping strings to ranks
   * @param pattern - Regex pattern for splitting text into pieces
   */
  constructor(
    encoder: Vocabulary,
    specialTokensEncoder: Map<string, Rank>,
    pattern: string,
  ) {
    this.encoder = encoder;
    this.specialTokensEncoder = specialTokensEncoder;
    this.textEncoder = new TextEncoder();
    this.textDecoder = new TextDecoder("utf-8", { fatal: false });

    // Build the decoder (reverse of encoder)
    this.decoder = new Map();
    for (const [key, rank] of encoder) {
      this.decoder.set(rank, keyToBytes(key));
    }

    // Verify encoder and decoder have same size
    if (encoder.size !== this.decoder.size) {
      throw new Error(
        "Encoder and decoder must be of equal length; maybe you had duplicate token indices in your encoder?",
      );
    }

    // Build special tokens decoder
    this.specialTokensDecoder = new Map();
    for (const [token, rank] of specialTokensEncoder) {
      this.specialTokensDecoder.set(rank, this.textEncoder.encode(token));
    }

    // Compile the regex pattern
    // Note: JavaScript regex flags differ from Rust - we use 'gu' for global + unicode
    this.regex = new RegExp(pattern, "gu");

    // Build special tokens regex
    if (specialTokensEncoder.size > 0) {
      const escapedTokens = Array.from(specialTokensEncoder.keys())
        .map(escapeRegex)
        .join("|");
      this.specialRegex = new RegExp(escapedTokens, "g");
    } else {
      this.specialRegex = null;
    }
  }

  /**
   * Decode tokens to raw bytes
   *
   * The bytes are not guaranteed to be valid UTF-8.
   *
   * @param tokens - Array of token ranks to decode
   * @returns Decoded bytes
   * @throws DecodeKeyError if a token is not found
   */
  decodeBytes(tokens: Rank[]): Uint8Array {
    const parts: Uint8Array[] = [];
    let totalLength = 0;

    for (const token of tokens) {
      let tokenBytes = this.decoder.get(token);
      if (tokenBytes === undefined) {
        tokenBytes = this.specialTokensDecoder.get(token);
      }
      if (tokenBytes === undefined) {
        throw new DecodeKeyError(token);
      }
      parts.push(tokenBytes);
      totalLength += tokenBytes.length;
    }

    // Concatenate all parts
    const result = new Uint8Array(totalLength);
    let offset = 0;
    for (const part of parts) {
      result.set(part, offset);
      offset += part.length;
    }

    return result;
  }

  /**
   * Decode tokens to a string
   *
   * @param tokens - Array of token ranks to decode
   * @returns Decoded string
   * @throws DecodeKeyError if a token is not found
   */
  decode(tokens: Rank[]): string {
    const bytes = this.decodeBytes(tokens);
    return this.textDecoder.decode(bytes);
  }

  /**
   * Encode text without handling special tokens
   *
   * This is the core encoding logic. Special tokens are treated as regular text.
   *
   * @param text - Text to encode
   * @returns Array of token ranks
   */
  encodeOrdinary(text: string): Rank[] {
    const tokens: Rank[] = [];

    // Reset regex state
    this.regex.lastIndex = 0;

    let match;
    while ((match = this.regex.exec(text)) !== null) {
      const piece = match[0];
      const pieceBytes = this.textEncoder.encode(piece);
      const key = bytesToKey(pieceBytes);

      // Check if the piece is a single token
      const directRank = this.encoder.get(key);
      if (directRank !== undefined) {
        tokens.push(directRank);
      } else {
        // Apply BPE
        const bpeTokens = bytePairEncode(pieceBytes, this.encoder);
        tokens.push(...bpeTokens);
      }
    }

    return tokens;
  }

  /**
   * Encode text with special token handling
   *
   * @param text - Text to encode
   * @param allowedSpecial - Set of special tokens that are allowed
   * @returns Tuple of [tokens, lastPieceTokenLen]
   */
  encode(text: string, allowedSpecial: Set<string>): [Rank[], number] {
    if (!this.specialRegex || allowedSpecial.size === 0) {
      // No special tokens to handle
      return [this.encodeOrdinary(text), 0];
    }

    const tokens: Rank[] = [];
    let start = 0;
    let lastPieceTokenLen = 0;

    while (start < text.length) {
      // Find the next special token
      let nextSpecial: { index: number; token: string } | null = null;
      let searchStart = start;

      while (searchStart < text.length) {
        this.specialRegex.lastIndex = searchStart;
        const match = this.specialRegex.exec(text);

        if (!match) {
          break;
        }

        const matchedToken = match[0];
        if (allowedSpecial.has(matchedToken)) {
          nextSpecial = { index: match.index, token: matchedToken };
          break;
        }

        // Token not allowed, continue searching
        searchStart = match.index + 1;
      }

      const end = nextSpecial ? nextSpecial.index : text.length;

      // Encode the text before the special token
      if (end > start) {
        const segment = text.slice(start, end);

        // Reset regex state
        this.regex.lastIndex = 0;

        let match;
        while ((match = this.regex.exec(segment)) !== null) {
          const piece = match[0];
          const pieceBytes = this.textEncoder.encode(piece);
          const key = bytesToKey(pieceBytes);

          const directRank = this.encoder.get(key);
          if (directRank !== undefined) {
            lastPieceTokenLen = 1;
            tokens.push(directRank);
          } else {
            const bpeTokens = bytePairEncode(pieceBytes, this.encoder);
            lastPieceTokenLen = bpeTokens.length;
            tokens.push(...bpeTokens);
          }
        }
      }

      // Handle the special token
      if (nextSpecial) {
        const specialRank = this.specialTokensEncoder.get(nextSpecial.token);
        if (specialRank !== undefined) {
          tokens.push(specialRank);
          lastPieceTokenLen = 0;
        }
        start = nextSpecial.index + nextSpecial.token.length;
      } else {
        break;
      }
    }

    return [tokens, lastPieceTokenLen];
  }

  /**
   * Encode text with all special tokens allowed
   *
   * @param text - Text to encode
   * @returns Array of token ranks
   */
  encodeWithSpecialTokens(text: string): Rank[] {
    const allowedSpecial = this.getSpecialTokens();
    return this.encode(text, allowedSpecial)[0];
  }

  /**
   * Get all special tokens
   *
   * @returns Set of special token strings
   */
  getSpecialTokens(): Set<string> {
    return new Set(this.specialTokensEncoder.keys());
  }

  /**
   * Get the vocabulary size (excluding special tokens)
   */
  get vocabSize(): number {
    return this.encoder.size;
  }

  /**
   * Get the total vocabulary size (including special tokens)
   */
  get totalVocabSize(): number {
    return this.encoder.size + this.specialTokensEncoder.size;
  }

  /**
   * Check if a token rank is a special token
   */
  isSpecialToken(token: Rank): boolean {
    return this.specialTokensDecoder.has(token);
  }

  /**
   * Get the byte representation of a token
   */
  getTokenBytes(token: Rank): Uint8Array | undefined {
    return this.decoder.get(token) ?? this.specialTokensDecoder.get(token);
  }

  /**
   * Get the rank of a byte sequence
   */
  getRank(bytes: Uint8Array): Rank | undefined {
    const key = bytesToKey(bytes);
    return this.encoder.get(key);
  }

  /**
   * Get the rank of a special token
   */
  getSpecialTokenRank(token: string): Rank | undefined {
    return this.specialTokensEncoder.get(token);
  }
}
