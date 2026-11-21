# PaliGemma 在 HF 上是 gated，集群没有 token。用 openpi 缓存里 big_vision 的 paligemma_tokenizer.model 造一个 HF 格式分词器。
import sys
from transformers import GemmaTokenizerFast, GemmaTokenizer
sp, out = sys.argv[1], sys.argv[2]
tok = GemmaTokenizer(vocab_file=sp)
fast = GemmaTokenizerFast(vocab_file=sp, from_slow=True) if False else None
print("base vocab", len(tok))
tok.save_pretrained(out)
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained(out, padding_side="right")
t.add_special_tokens({"additional_special_tokens": ["<image>"]})
print("type", type(t).__name__, "image id", t.convert_tokens_to_ids("<image>"), "loc0000", t.convert_tokens_to_ids("<loc0000>"), "bos", t.bos_token_id, "pad", t.pad_token_id)
print("ids", t("put eggplant into yellow basket\n", add_special_tokens=False).input_ids)
