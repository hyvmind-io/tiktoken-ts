# Inside Claude's tokenizer: What we know and what remains hidden

**Anthropic has not publicly disclosed Claude 3's tokenization algorithm, vocabulary size, or technical design—a deliberate competitive decision.** However, reverse-engineering efforts by academic researchers and community developers have revealed substantial details: Claude likely uses a modified BPE (Byte-Pair Encoding) tokenizer with an unusually small vocabulary of approximately **22,000 tokens**, produces **16-30% more tokens** than GPT-4's cl100k_base for equivalent content, and exhibits several anomalous behaviors suggesting it's not a standard BPE implementation. The only official method for accurate Claude 3+ token counting is Anthropic's API endpoint.

## The official picture: What Anthropic has disclosed

Anthropic provides a TypeScript tokenizer package (`@anthropic-ai/tokenizer`) with a critical caveat displayed prominently in its README: "As of the Claude 3 models, this algorithm is no longer accurate, but can be used as a very rough approximation." This confirms Claude 3 introduced a fundamentally different tokenizer from earlier versions.

For **Claude 1 and 2 models**, Anthropic's official tokenizer uses BPE with a **~65,000 token vocabulary** (plus five special tokens). The tokenizer configuration lives in a `claude.json` file containing `bpe_ranks` (base64-encoded merge rankings), `special_tokens`, and a regex pattern (`pat_str`) for pre-tokenization. It uses NFKC normalization and is tiktoken-compatible. According to analysis by Zhen Wang, **70% of Claude 1/2's vocabulary** (approximately 45,200 tokens) overlaps with GPT-4's cl100k_base model.

For **Claude 3+**, Anthropic offers only a token counting API endpoint (`/v1/messages/count_tokens`) that returns total counts without tokenization details. Their interpretability research explicitly acknowledges this opacity: "Some of the decisions in this publication reflect this, such as not reporting the size of the model...and using a simplified tokenizer." According to Hacker News discussions, Anthropic has stated that releasing a new tokenizer is "not on their current roadmap."

## Reverse engineering reveals unusual properties

The most significant reverse-engineering effort comes from **Javier Rando and Florian Tramèr** at ETH Zurich's SPY Lab. Their technique exploits Claude's streaming behavior: by asking Claude to repeat text with `max_tokens=1`, they could extract individual tokens one at a time. Their key findings:

The API returns `number_of_text_tokens + 3` for any successful tokenization, suggesting three special tokens (likely beginning/end-of-turn markers). For example, "Security by obscurity is not a good idea." tokenizes to `['Security', ' by', ' obsc', 'urity', ' is', ' not', ' a', ' good', ' idea', '.']`—10 tokens. Anthropic has since patched the streaming to no longer directly reveal tokens in network traffic, though the `max_tokens=1` technique still works.

**Sander Land's extensive analysis** on the Token Contributions newsletter uncovered deeper anomalies. Through exhaustive search, he estimates Claude 3's vocabulary at only **~22,000 unique tokens**—remarkably small compared to Mistral's 32K, GPT-4's 100K, or LLaMA 3's 128K. Despite this, **8,311 of the 10,000 most common English words** tokenize as single tokens in Claude 3, versus only 7,682 in the combined vocabularies of LLaMA 3, Mistral, Cohere, and Gemma.

Strange behaviors emerged from testing. Repeated characters show non-linear patterns: `J` = 1 token, `JJ` = 2 tokens, `JJJ` = 3 tokens, but `JJJJ` inexplicably produces **5 tokens**. Similarly, `A` = 1, `AA` = 1, `AAA` = 2, but `AAAA` = 4 tokens (not the expected 2). The common English suffix "ent" requires **two tokens** in Claude despite being a single subword in standard BPE tokenizers.

Input and output tokenization also differ asymmetrically. A single digit like "1" costs 1 token as input but 2 tokens as output. CJK characters behave differently between input and output contexts. Land concludes this "looks like a fairly standard byte-based BPE, although the Korean token counts point to something a little more handcrafted for certain Unicode ranges"—with evidence of randomness "reminiscent of Unigram or BPE-Dropout methods."

## How Claude compares to GPT and other tokenizers

Claude's tokenizer is measurably **less efficient** than GPT-4's cl100k_base across all content types:

| Content Type | Claude Token Overhead vs GPT-4 |
|--------------|-------------------------------|
| English articles | +16% |
| Math equations | +21% |
| Python code | +30% |

This efficiency gap directly impacts costs. For equivalent workloads, Claude requires 16-30% more tokens, translating to higher API expenses for enterprise deployments. The smaller vocabulary forces more text to be represented through subword combinations rather than single tokens.

Character-to-token ratios differ accordingly: Claude averages approximately **3.5 characters per token**, while GPT-4 averages closer to 4 characters per token. Both approximate 0.75 words per token for standard English prose—Anthropic's documentation states "100K tokens ≈ 75,000 words."

Academic research analyzing BPE tokenizers estimates Claude's training data composition at **57.5% code, 38.8% English, and 3.7% other languages**. Paradoxically, despite this heavy code training, Claude's smaller vocabulary makes code tokenization less efficient than GPT models optimized for programming content.

## Available tools and approximations for token counting

No publicly available tokenizer exactly matches Claude 3+ behavior. The practical options:

**For Claude 1/2 models:** The `Xenova/claude-tokenizer` on Hugging Face provides a `GPT2TokenizerFast`-compatible implementation adapted from Anthropic's SDK. Usage: `tokenizer = GPT2TokenizerFast.from_pretrained('Xenova/claude-tokenizer')`. This has been downloaded by 31 HuggingFace Spaces and works reliably for pre-Claude-3 applications.

**For Claude 3+ models:** Several community tools exist:
- **javirandor/anthropic-tokenizer** (147 GitHub stars): Python scripts using the `max_tokens=1` reverse-engineering technique
- **claude-tokenizer.vercel.app**: Web tool that calls Anthropic's official API
- **tokencost library** (AgentOps-AI): Falls back to tiktoken cl100k_base for offline estimation

For rough offline approximations without API calls, developers use tiktoken's `p50k_base` with a 1.2x multiplier, or the simple heuristic of dividing character count by 3.5. Both approaches have "much higher error rates" according to community testing but work for order-of-magnitude estimates.

## Technical findings from code analysis

The official Claude 1/2 tokenizer structure reveals implementation details applicable to understanding the tokenizer family:

```python
{
  "bpe_ranks": "! 5 [base64-encoded merge tokens...]",  # Offset starts at 5
  "special_tokens": {...},
  "pat_str": "..."  # Regex pattern for pre-tokenization
}
```

This format is tiktoken-compatible, allowing integration via:
```python
import tiktoken
enc = tiktoken.Encoding(
    name="claude_tokenizer",
    pat_str=config['pat_str'],
    mergeable_ranks=decode_claude_bpe(config),
    special_tokens=config['special_tokens']
)
```

Multiple community implementations exist across languages: Rust crates (`19h/claude_tokenizer`, `Jellyfishboy/claude-tokenizer`), .NET packages (`LLMSharp.Anthropic.Tokenizer`), and JavaScript versions via Transformers.js. All work only for Claude 1/2 models.

## Conclusion

Claude's tokenization remains among the most opaque of major LLM providers. The key technical facts: Claude 3+ likely uses a **modified BPE tokenizer** with an unusually small **~22,000 token vocabulary**, exhibits **whole-word preferences** that bypass normal subword tokenization, and shows anomalous behaviors suggesting handcrafted modifications or BPE-Dropout-style randomness. The **three-token end sequence** and **input/output asymmetry** for certain character types further distinguish it from standard implementations.

For practical work, developers must choose between API-based counting (accurate but requires network calls) or multiplier-based estimation (offline but imprecise). The 16-30% efficiency gap versus GPT-4 has real cost implications for high-volume applications. Until Anthropic changes course on disclosure—which they've indicated isn't planned—the community's reverse-engineering efforts remain the primary source of technical understanding about Claude's tokenization internals.