import asyncio
import os
import json
from transformer_lens import HookedTransformer
from sae_lens import SAE
import sae_bench.sae_bench_utils.activation_collection as activation_collection
import sae_bench.sae_bench_utils.dataset_utils as dataset_utils
import sae_bench.sae_bench_utils.general_utils as general_utils
from sae_bench.evals.autointerp.eval_config import AutoInterpEvalConfig
from sae_bench.evals.autointerp.main import AutoInterp

async def generate_explanations():
    MODEL_NAME = "gemma-2-2b"
    HOOK_LAYER = 12
    HOOK_NAME = f"blocks.{HOOK_LAYER}.hook_resid_post"
    DEVICE = "cuda:0"
    dtype = torch.bfloat16

    # Configuration
    model_name = MODEL_NAME  # Change as needed

    # Load API key
    with open("openai_api_key.txt") as f:
        api_key = f.read().strip()

    device = general_utils.setup_environment()

    # Create config with scoring disabled
    config = AutoInterpEvalConfig(
        model_name=model_name,
        scoring=False,  # Disable scoring
        # n_latents=100,  # Adjust as needed
    )
    config.llm_batch_size = 32
    config.llm_dtype = "bfloat16"

    # Load model and SAE
    llm_dtype = general_utils.str_to_dtype(config.llm_dtype)
    model = HookedTransformer.from_pretrained_no_processing(
        model_name, device=device, dtype=llm_dtype
    )

    # sae = SAE.from_pretrained(release=sae_release, sae_id=sae_id, device=device)
    # sae = sae.to(device=device, dtype=llm_dtype)

    file_name = "gemma-2-2b-it_5M_layer12_largebuf.h5-result-baseline-1048576-65536-1-4096-20.npy"
    L0 = int(file_name.split("-")[-1].split(".")[0])
    D = np.load("/mnt/sisl-llm-embeddings/"+file_name)
    ksvd = KSVD(D, L0, MODEL_NAME, HOOK_LAYER, DEVICE, dtype)


    # Load tokenized dataset
    tokenized_dataset = dataset_utils.load_and_tokenize_dataset(
        config.dataset_name,
        config.llm_context_size,
        config.total_tokens,
        model.tokenizer,
    ).to(device)

    # Get sparsity
    sae_sparsity = activation_collection.get_feature_activation_sparsity(
        tokenized_dataset,
        model,
        sae,
        config.llm_batch_size,
        sae.cfg.hook_layer,
        sae.cfg.hook_name,
        mask_bos_pad_eos_tokens=True,
    )

    # Create AutoInterp instance
    autointerp = AutoInterp(
        cfg=config,
        model=model,
        sae=sae,
        tokenized_dataset=tokenized_dataset,
        sparsity=sae_sparsity,
        api_key=api_key,
        device=device,
    )

    # Generate explanations
    results = await autointerp.run()

    # Extract just the explanations
    explanations = {
        latent: result["explanation"]
        for latent, result in results.items()
    }

    # Save to disk
    output_file = f"explanations_{MODEL_NAME}_{D_FILE_NAME}.json"
    with open(output_file, 'w') as f:
        json.dump(explanations, f, indent=2)

    print(f"Saved {len(explanations)} explanations to {output_file}")

    # Also save a readable text format
    text_file = output_file.replace('.json', '.txt')
    with open(text_file, 'w') as f:
        for latent, explanation in explanations.items():
            f.write(f"Latent {latent}: {explanation}\n")

    print(f"Also saved readable format to {text_file}")

if __name__ == "__main__":
    asyncio.run(generate_explanations())
