from .deepseek_v32 import DeepseekV32ForCausalLM
from .llama import LlamaForCausalLM
from .mini_cpm4 import MiniCPMForCausalLM
from .qwen3 import Qwen3ForCausalLM
from .qwen3_moe import Qwen3MoeForCausalLM

model_dict = {
    "DeepseekV32ForCausalLM": DeepseekV32ForCausalLM,
    "DeepseekV3ForCausalLM": DeepseekV32ForCausalLM,
    "LlamaForCausalLM": LlamaForCausalLM,
    "Qwen2ForCausalLM": Qwen3ForCausalLM,
    "Qwen3ForCausalLM": Qwen3ForCausalLM,
    "Qwen3MoeForCausalLM": Qwen3MoeForCausalLM,
    "MiniCPMForCausalLM": MiniCPMForCausalLM,
}
