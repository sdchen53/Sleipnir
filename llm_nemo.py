# %%
from nemo.collections import llm
import nemo_run as run


pretrain = llm.qwen2_7b.pretrain_recipe(
    name="qwen2_7b_pretraining",
    dir=f"/path/to/checkpoints",
    num_nodes=2,
    num_gpus_per_node=8,
)

# pretrain = llm.qwen25_72b.pretrain_recipe(
#     name="qwen25_72b_pretraining",
#     dir=f"./checkpoints",
#     num_nodes=1,
#     num_gpus_per_node=1,
# )

# # To override the data argument
# dataloader = a_function_that_configures_your_custom_dataset(
#     global_batch_size=global_batch_size,
#     micro_batch_size=micro_batch_size,
#     seq_length=pretrain.model.config.seq_length,
# )
# pretrain.data = dataloader

if __name__ == "__main__":

    llm.import_ckpt(model=llm.Qwen3Model(llm.Qwen3Config8B()), source='hf://Qwen/Qwen3-8B')



    # recipe = llm.qwen25_500m.finetune_recipe(
    #     name="qwen2.5_500m_finetuning",
    #     dir=f"./checkpoints",
    #     num_nodes=1,
    #     num_gpus_per_node=1,
    #     peft_scheme='lora',  # 'lora', 'none'
    #     packed_sequence=False,
    # )

    # # %%

    # run.run(recipe, executor=run.LocalExecutor())


