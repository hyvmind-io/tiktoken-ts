/**
 * Core Tiktoken Module
 *
 * This module provides the exact BPE implementation from tiktoken-rs,
 * ported to TypeScript for use in both Node.js and browser environments.
 */

// Byte-Pair Encoding algorithm
export {
  type Rank,
  type Vocabulary,
  type ReverseVocabulary,
  bytesToKey,
  keyToBytes,
  bytePairEncode,
  bytePairSplit,
} from "./byte-pair-encoding.js";

// Vocabulary loading
export {
  parseVocabulary,
  loadVocabularyFromUrl,
  loadVocabularyFromString,
  createSpecialTokenMaps,
  clearVocabularyCache,
  getVocabularyFromCache,
  isVocabularyCached,
  VOCABULARY_URLS,
  base64Encode,
  keyToBytes as vocabKeyToBytes,
} from "./vocab-loader.js";

// Core BPE tokenizer
export { CoreBPE, DecodeKeyError } from "./tiktoken.js";

// Encoding definitions
export {
  type EncodingDefinition,
  SPECIAL_TOKENS,
  R50K_BASE,
  P50K_BASE,
  P50K_EDIT,
  CL100K_BASE,
  O200K_BASE,
  O200K_HARMONY,
  ENCODING_DEFINITIONS,
  getEncodingDefinition,
  listEncodingNames,
} from "./encoding-definitions.js";

// Model to encoding mappings
export {
  type TokenizerName,
  getTokenizerForModel,
  getContextSize,
  getExactContextSize,
  EXACT_CONTEXT_SIZES,
} from "./model-to-encoding.js";
