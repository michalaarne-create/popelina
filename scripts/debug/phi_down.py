from huggingface_hub import hf_hub_download

files = [
    "llava-phi-3-8b.Q4_K_M.gguf",
    "llava-phi-3-8b-mmproj-f16.gguf",
    "clip-vit-l14-336-f16.gguf"
]

for f in files:
    path = hf_hub_download(
        repo_id="xtuner/llava-phi-3-8b-gguf",
        filename=f,
        local_dir="C:/llava/models"
    )
    print("✅ Pobrano:", path)
