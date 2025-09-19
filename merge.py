from nemo.collections import llm

if __name__ == '__main__':
    llm.peft.merge_lora(
        lora_checkpoint_path="/data/finetuned-models/mistral-7b-instruct-dolly/default/2025-09-18_17-46-27/checkpoints/model_name=0--val_loss=1.44-step=49-consumed_samples=800.0-last",
        output_path="/data/finetuned-models/mistral-merged",
    )
