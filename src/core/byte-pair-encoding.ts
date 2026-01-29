/**
 * Byte-Pair Encoding Core Algorithm
 *
 * This is an EXACT port of the BPE algorithm from tiktoken-rs.
 * Reference: https://github.com/zurawiki/tiktoken-rs/blob/main/tiktoken-rs/src/vendor_tiktoken.rs
 *
 * The algorithm works as follows:
 * 1. Split input text into pieces using a regex pattern
 * 2. For each piece, convert to bytes and apply BPE merges
 * 3. Look up each merged byte sequence in the vocabulary to get token IDs
 */

/**
 * Token rank type (same as tiktoken-rs)
 */
export type Rank = number;

/**
 * Vocabulary type: maps byte sequences to ranks
 */
export type Vocabulary = Map<string, Rank>;

/**
 * Reverse vocabulary: maps ranks to byte sequences
 */
export type ReverseVocabulary = Map<Rank, Uint8Array>;

/**
 * Convert a byte array to a string key for Map lookup
 * This is necessary because JavaScript Maps use reference equality for arrays
 */
export function bytesToKey(bytes: Uint8Array): string {
  // Use a more efficient encoding for the key
  // Each byte becomes a character in the string
  let key = "";
  for (let i = 0; i < bytes.length; i++) {
    key += String.fromCharCode(bytes[i]);
  }
  return key;
}

/**
 * Convert a string key back to bytes
 */
export function keyToBytes(key: string): Uint8Array {
  const bytes = new Uint8Array(key.length);
  for (let i = 0; i < key.length; i++) {
    bytes[i] = key.charCodeAt(i);
  }
  return bytes;
}

/**
 * Internal BPE merge algorithm
 *
 * This is the core of the BPE algorithm. It finds the lowest-ranked pair
 * of adjacent bytes and merges them, repeating until no more merges are possible.
 *
 * @param ranks - The vocabulary mapping byte sequences to ranks
 * @param piece - The byte sequence to process
 * @returns Array of (start_index, rank) tuples representing the merge points
 */
function _bytePairMerge(
  ranks: Vocabulary,
  piece: Uint8Array,
): Array<[number, Rank]> {
  // This is a vector of (start, rank).
  // The rank is of the pair starting at position start.
  const parts: Array<[number, Rank]> = [];

  // Note that we hash bytes when indexing into `ranks`, not token pairs. As long as we train BPE
  // the way we currently do, this is equivalent. An easy way to break this would be to decouple
  // merge priority from token index or to prevent specific token merges.
  let minRank: [Rank, number] = [
    Number.MAX_SAFE_INTEGER,
    Number.MAX_SAFE_INTEGER,
  ];

  for (let i = 0; i < piece.length - 1; i++) {
    const pairKey = bytesToKey(piece.slice(i, i + 2));
    const rank = ranks.get(pairKey) ?? Number.MAX_SAFE_INTEGER;
    if (rank < minRank[0]) {
      minRank = [rank, i];
    }
    parts.push([i, rank]);
  }
  parts.push([piece.length - 1, Number.MAX_SAFE_INTEGER]);
  parts.push([piece.length, Number.MAX_SAFE_INTEGER]);

  const getRank = (parts: Array<[number, Rank]>, i: number): Rank => {
    if (i + 3 < parts.length) {
      // Similar to `piece[i..i + 2]` above. The +3 is because we haven't yet deleted
      // parts[i + 1], see comment in the main loop.
      const start = parts[i][0];
      const end = parts[i + 3][0];
      const pairKey = bytesToKey(piece.slice(start, end));
      return ranks.get(pairKey) ?? Number.MAX_SAFE_INTEGER;
    } else {
      return Number.MAX_SAFE_INTEGER;
    }
  };

  // If you have n parts and m merges, this does O(mn) work.
  // We could do something with a heap and do O(m log n) work.
  // n is often very small so considerations like cache-locality outweigh the algorithmic
  // complexity downsides of the `parts` vector.
  while (minRank[0] !== Number.MAX_SAFE_INTEGER) {
    const i = minRank[1];
    // Update parts[i] and parts[i - 1] before removing parts[i + 1], since
    // `parts.splice(i + 1, 1)` will thrash the cache.
    if (i > 0) {
      parts[i - 1][1] = getRank(parts, i - 1);
    }
    parts[i][1] = getRank(parts, i);
    parts.splice(i + 1, 1);

    minRank = [Number.MAX_SAFE_INTEGER, Number.MAX_SAFE_INTEGER];
    for (let j = 0; j < parts.length - 1; j++) {
      const rank = parts[j][1];
      if (rank < minRank[0]) {
        minRank = [rank, j];
      }
    }
  }

  return parts;
}

/**
 * Encode a byte sequence using BPE
 *
 * @param piece - The byte sequence to encode
 * @param ranks - The vocabulary mapping byte sequences to ranks
 * @returns Array of token ranks
 */
export function bytePairEncode(piece: Uint8Array, ranks: Vocabulary): Rank[] {
  if (piece.length === 1) {
    const key = bytesToKey(piece);
    const rank = ranks.get(key);
    if (rank === undefined) {
      throw new Error(`Unknown byte: ${piece[0]}`);
    }
    return [rank];
  }

  const parts = _bytePairMerge(ranks, piece);

  // Convert parts to token ranks
  const tokens: Rank[] = [];
  for (let i = 0; i < parts.length - 1; i++) {
    const start = parts[i][0];
    const end = parts[i + 1][0];
    const tokenBytes = piece.slice(start, end);
    const key = bytesToKey(tokenBytes);
    const rank = ranks.get(key);
    if (rank === undefined) {
      throw new Error(`Unknown token: ${Array.from(tokenBytes).join(",")}`);
    }
    tokens.push(rank);
  }

  return tokens;
}

/**
 * Split a byte sequence into BPE tokens (returns byte slices, not ranks)
 *
 * @param piece - The byte sequence to split
 * @param ranks - The vocabulary mapping byte sequences to ranks
 * @returns Array of byte slices
 */
export function bytePairSplit(
  piece: Uint8Array,
  ranks: Vocabulary,
): Uint8Array[] {
  if (piece.length <= 1) {
    return [piece];
  }

  const parts = _bytePairMerge(ranks, piece);
  const splits: Uint8Array[] = [];

  for (let i = 0; i < parts.length - 1; i++) {
    const start = parts[i][0];
    const end = parts[i + 1][0];
    splits.push(piece.slice(start, end));
  }

  return splits;
}
