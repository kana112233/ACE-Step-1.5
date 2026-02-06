# GPU Compatibility Guide

ACE-Step 1.5 automatically adapts to your GPU's available VRAM, adjusting generation limits and LM model availability accordingly. The system detects GPU memory at startup and configures optimal settings.

## GPU Tier Configuration

| VRAM | Tier | LM Mode | Max Duration | Max Batch Size | LM Memory Allocation |
|------|------|---------|--------------|----------------|---------------------|
| ≤4GB | Tier 1 | Not available | 3 min | 1 | - |
| 4-6GB | Tier 2 | Not available | 6 min | 1 | - |
| 6-8GB | Tier 3 | 0.6B (optional) | With LM: 4 min / Without: 6 min | With LM: 1 / Without: 2 | 3GB |
| 8-12GB | Tier 4 | 0.6B (optional) | With LM: 4 min / Without: 6 min | With LM: 2 / Without: 4 | 3GB |
| 12-16GB | Tier 5 | 0.6B / 1.7B | With LM: 4 min / Without: 6 min | With LM: 2 / Without: 4 | 0.6B: 3GB, 1.7B: 8GB |
| 16-24GB | Tier 6 | 0.6B / 1.7B / 4B | 8 min | With LM: 4 / Without: 8 | 0.6B: 3GB, 1.7B: 8GB, 4B: 12GB |
| ≥24GB | Unlimited | All models | 10 min | 8 | Unrestricted |

## Notes

- **Default settings** are automatically configured based on detected GPU memory
- **LM Mode** refers to the Language Model used for Chain-of-Thought generation and audio understanding
- **Flash Attention**, **CPU Offload**, **Compile**, and **Quantization** are enabled by default for optimal performance
- If you request a duration or batch size exceeding your GPU's limits, a warning will be displayed and values will be clamped
- **Constrained Decoding**: When LM is initialized, the LM's duration generation is also constrained to the GPU tier's maximum duration limit, preventing out-of-memory errors during CoT generation
- For GPUs with ≤6GB VRAM, LM initialization is disabled by default to preserve memory for the DiT model
- You can manually override settings via command-line arguments or the Gradio UI

> **Community Contributions Welcome**: The GPU tier configurations above are based on our testing across common hardware. If you find that your device's actual performance differs from these parameters (e.g., can handle longer durations or larger batch sizes), we welcome you to conduct more thorough testing and submit a PR to optimize these configurations in `acestep/gpu_config.py`. Your contributions help improve the experience for all users!

## Memory Optimization Tips

1. **Low VRAM (<8GB)**: Use DiT-only mode without LM initialization for maximum duration
2. **Medium VRAM (8-16GB)**: Use the 0.6B LM model for best balance of quality and memory
3. **High VRAM (>16GB)**: Enable larger LM models (1.7B/4B) for better audio understanding and generation quality

## Debug Mode: Simulating Different GPU Configurations

For testing and development, you can simulate different GPU memory sizes using the `MAX_CUDA_VRAM` environment variable:

```bash
# Simulate a 4GB GPU (Tier 1)
MAX_CUDA_VRAM=4 uv run acestep

# Simulate an 8GB GPU (Tier 4)
MAX_CUDA_VRAM=8 uv run acestep

# Simulate a 12GB GPU (Tier 5)
MAX_CUDA_VRAM=12 uv run acestep

# Simulate a 16GB GPU (Tier 6)
MAX_CUDA_VRAM=16 uv run acestep
```

This is useful for:
- Testing GPU tier configurations on high-end hardware
- Verifying that warnings and limits work correctly for each tier
- Developing and testing new GPU configuration parameters before submitting a PR

## Troubleshooting

### Flash Attention Issues

**Problem:** GPU compatibility errors, crashes, or "CUDA error" messages during inference.

**Symptoms:**
- Application crashes when initializing models
- Error messages mentioning "flash_attn" or "flash attention"
- CUDA errors during generation

**Solution:**

1. **For Windows Portable Package Users:**
   
   Edit `start_gradio_ui.bat` and add this line in the "Service Configuration" section:
   ```batch
   set USE_FLASH_ATTENTION=--use_flash_attention false
   ```
   Save the file and restart the application.

2. **For Command Line Users:**
   
   Launch with flash attention disabled:
   ```bash
   # Using uv
   uv run acestep --use_flash_attention false
   
   # Using Python directly
   python acestep/acestep_v15_pipeline.py --use_flash_attention false
   ```

3. **For Gradio UI Users (non-portable):**
   
   If the Service Configuration tab is visible:
   - Uncheck the "Use Flash Attention" checkbox
   - Click "Initialize Service"

**Why this happens:**
Flash Attention requires specific GPU architectures (Ampere or newer for NVIDIA) and may not be compatible with all GPUs. Disabling it will use the standard attention mechanism, which is slower but more compatible.

### Service Configuration Tab Missing

**Problem:** Cannot find the Service Configuration tab in the Gradio UI.

**Cause:** The service was pre-initialized on startup (common in Windows Portable version).

**Solutions:**

1. **Configure via `start_gradio_ui.bat` (Windows Portable):**
   
   Edit the file and modify settings directly:
   ```batch
   set USE_FLASH_ATTENTION=--use_flash_attention false
   set CONFIG_PATH=--config_path acestep-v15-turbo
   set LM_MODEL_PATH=--lm_model_path acestep-5Hz-lm-0.6B
   set OFFLOAD_TO_CPU=--offload_to_cpu true
   ```

2. **Disable auto-initialization to show the tab:**
   
   In `start_gradio_ui.bat`, comment out or remove:
   ```batch
   REM set INIT_SERVICE=--init_service true
   ```
   Or change to:
   ```batch
   set INIT_SERVICE=
   ```
   Then restart the application.

3. **Launch without pre-initialization:**
   ```bash
   # Don't use --init_service flag
   uv run acestep
   ```

See the [Service Configuration section in the Gradio Guide](./GRADIO_GUIDE.md#service-configuration) for more details.

