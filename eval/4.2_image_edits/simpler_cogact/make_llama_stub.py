"""CogACT 的 LLM 骨干 llama2-7b-pure 在推理模式下只从 HF 取 meta-llama/Llama-2-7b-hf 的 config.json 和分词器
（权重全部来自 CogACT-Base.pt）。该仓库需授权、集群没有 token，所以用本地已有的 OpenVLA（公开、MIT）仓库里
随附的 Llama-2 分词器文件，加上 Llama-2-7b 的标准结构参数，拼一个本地目录。权重形状不对会在 load_state_dict 时报错。"""
import json, os, shutil, sys, hashlib
SRC = os.path.expanduser("~/hf_cache/hub/models--moojink--openvla-7b-oft-finetuned-libero-spatial-object-goal-10/snapshots/638918f3d1c2e43a39a8a20772bdb8b91835e4b7")
DST = os.path.expanduser("$PROJ_ROOT/bench2_cogact/llama2_stub")
os.makedirs(DST, exist_ok=True)
for f in ["tokenizer.model", "tokenizer.json", "special_tokens_map.json", "added_tokens.json"]:
    shutil.copyfile(os.path.join(SRC, f), os.path.join(DST, f))
tc = json.load(open(os.path.join(SRC, "tokenizer_config.json")))
for k in ["auto_map", "processor_class"]: tc.pop(k, None)
json.dump(tc, open(os.path.join(DST, "tokenizer_config.json"), "w"), indent=2)
cfg = {"architectures": ["LlamaForCausalLM"], "bos_token_id": 1, "eos_token_id": 2, "hidden_act": "silu",
       "hidden_size": 4096, "initializer_range": 0.02, "intermediate_size": 11008, "max_position_embeddings": 4096,
       "model_type": "llama", "num_attention_heads": 32, "num_hidden_layers": 32, "num_key_value_heads": 32,
       "pretraining_tp": 1, "rms_norm_eps": 1e-05, "rope_scaling": None, "tie_word_embeddings": False,
       "torch_dtype": "float16", "use_cache": True, "vocab_size": 32000}
json.dump(cfg, open(os.path.join(DST, "config.json"), "w"), indent=2)
print("tokenizer.model md5", hashlib.md5(open(os.path.join(DST, "tokenizer.model"), "rb").read()).hexdigest())
from transformers import AutoTokenizer, LlamaTokenizer, AutoConfig
tf = AutoTokenizer.from_pretrained(DST, model_max_length=2048, padding_side="right")
ts = LlamaTokenizer.from_pretrained(DST)
p = "In: What action should the robot take to pick coke can?\nOut: "
print(type(tf).__name__, "fast", tf(p).input_ids)
print(type(ts).__name__, "slow", ts(p).input_ids)
print("same", tf(p).input_ids == ts(p).input_ids, "pad", tf.pad_token, tf.pad_token_id, "len", len(tf))
print(AutoConfig.from_pretrained(DST))
