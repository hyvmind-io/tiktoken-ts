/**
 * Tokenizer Utilities
 * Helper functions for tokenization
 */

/**
 * Convert a string to UTF-8 bytes
 *
 * @param text - Input text
 * @returns Uint8Array of UTF-8 bytes
 */
export function stringToBytes(text: string): Uint8Array {
  const encoder = new TextEncoder();
  return encoder.encode(text);
}

/**
 * Convert UTF-8 bytes to string
 *
 * @param bytes - UTF-8 bytes
 * @returns Decoded string
 */
export function bytesToString(bytes: Uint8Array | number[]): string {
  const decoder = new TextDecoder("utf-8", { fatal: false });
  return decoder.decode(
    bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes),
  );
}

/**
 * Convert a byte to its hexadecimal representation
 *
 * @param byte - Single byte value (0-255)
 * @returns Two-character hex string
 */
export function byteToHex(byte: number): string {
  return byte.toString(16).padStart(2, "0");
}

/**
 * Convert hexadecimal string to byte
 *
 * @param hex - Two-character hex string
 * @returns Byte value (0-255)
 */
export function hexToByte(hex: string): number {
  return parseInt(hex, 16);
}

/**
 * Check if a character is a whitespace character
 *
 * @param char - Single character
 * @returns True if whitespace
 */
export function isWhitespace(char: string): boolean {
  return /\s/.test(char);
}

/**
 * Check if a character is a letter
 *
 * @param char - Single character
 * @returns True if letter
 */
export function isLetter(char: string): boolean {
  return /\p{L}/u.test(char);
}

/**
 * Check if a character is a digit
 *
 * @param char - Single character
 * @returns True if digit
 */
export function isDigit(char: string): boolean {
  return /\p{N}/u.test(char);
}

/**
 * Check if a character is punctuation
 *
 * @param char - Single character
 * @returns True if punctuation
 */
export function isPunctuation(char: string): boolean {
  return /\p{P}/u.test(char);
}

/**
 * Check if a string contains only ASCII characters
 *
 * @param text - Input text
 * @returns True if ASCII only
 */
export function isAscii(text: string): boolean {
  // eslint-disable-next-line no-control-regex
  return /^[\x00-\x7F]*$/.test(text);
}

/**
 * Count the number of Unicode code points in a string
 * Handles surrogate pairs correctly
 *
 * @param text - Input text
 * @returns Number of code points
 */
export function countCodePoints(text: string): number {
  // Using the spread operator to handle surrogate pairs
  return [...text].length;
}

/**
 * Split text into words using Unicode-aware boundaries
 *
 * @param text - Input text
 * @returns Array of words and whitespace
 */
export function splitIntoWords(text: string): string[] {
  // Split on word boundaries while preserving whitespace
  const segments: string[] = [];
  let current = "";
  let inWord = false;

  for (const char of text) {
    const charIsWord = isLetter(char) || isDigit(char) || char === "'";

    if (charIsWord !== inWord) {
      if (current) {
        segments.push(current);
      }
      current = char;
      inWord = charIsWord;
    } else {
      current += char;
    }
  }

  if (current) {
    segments.push(current);
  }

  return segments;
}

/**
 * Escape special regex characters in a string
 *
 * @param text - Input text
 * @returns Escaped string safe for regex
 */
export function escapeRegex(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/**
 * Clamp a number between min and max values
 *
 * @param value - Value to clamp
 * @param min - Minimum value
 * @param max - Maximum value
 * @returns Clamped value
 */
export function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

/**
 * Calculate the percentage of a value
 *
 * @param value - Base value
 * @param percentage - Percentage (0-1)
 * @returns Calculated percentage value
 */
export function percentage(value: number, pct: number): number {
  return Math.round(value * pct);
}

/**
 * Create a hash from a string (for vocabulary lookup)
 * Simple FNV-1a hash for performance
 *
 * @param text - Input text
 * @returns 32-bit hash value
 */
export function hashString(text: string): number {
  let hash = 0x811c9dc5; // FNV offset basis
  const bytes = stringToBytes(text);

  for (const byte of bytes) {
    hash ^= byte;
    hash = Math.imul(hash, 0x01000193); // FNV prime
  }

  return hash >>> 0; // Convert to unsigned
}

/**
 * Count occurrences of a substring in text
 *
 * @param text - Text to search in
 * @param substring - Substring to count
 * @returns Number of occurrences
 */
export function countOccurrences(text: string, substring: string): number {
  if (!substring) return 0;

  let count = 0;
  let position = 0;

  while ((position = text.indexOf(substring, position)) !== -1) {
    count++;
    position += substring.length;
  }

  return count;
}

/**
 * Normalize whitespace in text (collapse multiple spaces)
 *
 * @param text - Input text
 * @returns Text with normalized whitespace
 */
export function normalizeWhitespace(text: string): string {
  return text.replace(/\s+/g, " ").trim();
}

/**
 * Check if text contains CJK (Chinese/Japanese/Korean) characters
 *
 * @param text - Input text
 * @returns True if contains CJK characters
 */
export function containsCJK(text: string): boolean {
  // CJK Unified Ideographs and related blocks
  return /[\u4e00-\u9fff\u3400-\u4dbf\u{20000}-\u{2a6df}\u{2a700}-\u{2b73f}\u{2b740}-\u{2b81f}\u{2b820}-\u{2ceaf}\u{2ceb0}-\u{2ebef}\u{30000}-\u{3134f}\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]/u.test(
    text,
  );
}

/**
 * Check if text contains emoji
 *
 * @param text - Input text
 * @returns True if contains emoji
 */
export function containsEmoji(text: string): boolean {
  return /\p{Emoji}/u.test(text);
}

/**
 * Estimate token multiplier based on text characteristics
 * Used for non-English text where token count can vary significantly
 *
 * @param text - Input text
 * @returns Multiplier to apply to base token estimate
 */
export function getTextComplexityMultiplier(text: string): number {
  let multiplier = 1.0;

  // CJK characters typically use more tokens per character
  if (containsCJK(text)) {
    multiplier *= 1.5;
  }

  // Emoji can use multiple tokens
  if (containsEmoji(text)) {
    multiplier *= 1.2;
  }

  // Non-ASCII text generally uses more tokens
  if (!isAscii(text)) {
    multiplier *= 1.1;
  }

  return multiplier;
}
