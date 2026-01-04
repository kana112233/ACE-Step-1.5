"""
Playground Handler
Wraps AceStepHandler and LLMHandler to provide a simple interface for the Playground UI.
"""
import os
import sys
import traceback
from typing import Optional, Tuple, List, Dict, Any

import torch

# Add project root to sys.path to allow importing acestep modules
current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler
except ImportError as e:
    print(f"Error importing acestep modules: {e}")
    print(f"sys.path: {sys.path}")
    raise


class PlaygroundHandler:
    """
    Handler for the Playground.
    Wraps AceStepHandler and LLMHandler to provide a simple interface for the UI.
    """
    
    def __init__(self):
        self.dit_handler = AceStepHandler()
        self.llm_handler = LLMHandler()
        
    # =========================================================================
    # Model Availability
    # =========================================================================
    
    def get_available_llm_models(self) -> List[str]:
        """Get list of available LLM models"""
        return self.llm_handler.get_available_5hz_lm_models()
    
    def get_available_dit_models(self) -> List[str]:
        """Get list of available DiT models"""
        return self.dit_handler.get_available_acestep_v15_models()
    
    # =========================================================================
    # LLM Section
    # =========================================================================
    
    def initialize_llm(
        self,
        lm_model_path: str,
        backend: str = "vllm",
        device: str = "auto"
    ) -> str:
        """
        Initialize LLM model.
        
        Args:
            lm_model_path: Model directory name (e.g., "acestep-5Hz-lm-xxx")
            backend: Backend type ("vllm" or "pt")
            device: Device type ("auto", "cuda", "cpu")
            
        Returns:
            Status message string
        """
        try:
            # Get checkpoint directory (auto-detect from project root)
            current_file_path = os.path.abspath(__file__)
            actual_project_root = os.path.dirname(os.path.dirname(current_file_path))
            checkpoint_dir = os.path.join(actual_project_root, "checkpoints")
            
            msg, success = self.llm_handler.initialize(
                checkpoint_dir=checkpoint_dir,
                lm_model_path=lm_model_path,
                backend=backend,
                device=device,
                offload_to_cpu=False,
                dtype=None  # Auto-detect based on device
            )
            return msg
        except Exception as e:
            return f"❌ Error initializing LLM: {str(e)}\n{traceback.format_exc()}"
    
    def generate_llm_codes(
        self,
        caption: str,
        lyrics: str,
        temperature: float = 0.85,
        cfg_scale: float = 2.0,
        negative_prompt: str = "NO USER INPUT",
        top_k: Optional[int] = None,
        top_p: Optional[float] = 0.9,
        repetition_penalty: float = 1.0,
        metadata_temperature: float = 0.85,
        codes_temperature: float = 1.0,
        target_duration: Optional[float] = None,
        user_metadata: Optional[Dict[str, str]] = None,
    ) -> Tuple[Dict, str, str]:
        """
        Generate audio codes using LLM.
        
        Args:
            caption: Music description
            lyrics: Song lyrics
            temperature: Sampling temperature
            cfg_scale: Classifier-free guidance scale
            negative_prompt: Negative prompt
            top_k: Top-K sampling (optional, 0 or None to disable)
            top_p: Top-P (nucleus) sampling (1.0 to disable)
            repetition_penalty: Repetition penalty
            metadata_temperature: Temperature for metadata generation
            codes_temperature: Temperature for codes generation
            target_duration: Target duration in seconds
            user_metadata: User-provided metadata dict (bpm, keyscale, timesignature)
            
        Returns:
            Tuple of (metadata_dict, audio_codes_string, status_message)
        """
        if not self.llm_handler.llm_initialized:
            return {}, "", "❌ LLM not initialized. Please load the LLM model first."
        
        try:
            # Convert top_k: 0 means None (disabled)
            top_k_value = None if (top_k is None or top_k == 0) else int(top_k)
            # Convert top_p: 1.0 means None (disabled)
            top_p_value = None if (top_p is None or top_p >= 1.0) else top_p
            
            # Use generate_with_stop_condition with infer_type='llm_dit' for full generation
            metadata, audio_codes, status = self.llm_handler.generate_with_stop_condition(
                caption=caption,
                lyrics=lyrics,
                infer_type="llm_dit",  # Full generation (metadata + audio codes)
                temperature=temperature,
                cfg_scale=cfg_scale,
                negative_prompt=negative_prompt,
                top_k=top_k_value,
                top_p=top_p_value,
                repetition_penalty=repetition_penalty,
                use_constrained_decoding=True,
                constrained_decoding_debug=False,
                target_duration=target_duration,
                user_metadata=user_metadata,
            )
            return metadata, audio_codes, status
        except Exception as e:
            return {}, "", f"❌ Error generating codes: {str(e)}\n{traceback.format_exc()}"
    
    # =========================================================================
    # ACEStep Section
    # =========================================================================
    
    def initialize_dit(
        self,
        config_path: str,
        device: str = "auto"
    ) -> str:
        """
        Initialize DiT model.
        
        Args:
            config_path: Model config directory name (e.g., "acestep-v15-turbo")
            device: Device type ("auto", "cuda", "cpu")
            
        Returns:
            Status message string
        """
        try:
            # Get project root (auto-detect)
            current_file_path = os.path.abspath(__file__)
            actual_project_root = os.path.dirname(os.path.dirname(current_file_path))
            
            msg, success = self.dit_handler.initialize_service(
                project_root=actual_project_root,
                config_path=config_path,
                device=device,
                use_flash_attention=self.dit_handler.is_flash_attention_available(),
                compile_model=False,
                offload_to_cpu=False,
                offload_dit_to_cpu=False,
                quantization=None
            )
            return msg
        except Exception as e:
            return f"❌ Error initializing DiT: {str(e)}\n{traceback.format_exc()}"
    
    def generate_instruction(
        self,
        task_type: str,
        track_name: Optional[str] = None,
        complete_track_classes: Optional[List[str]] = None
    ) -> str:
        """
        Generate instruction string based on task type.
        
        Args:
            task_type: Task type (generate, repaint, cover, add, complete, extract)
            track_name: Track name for add/extract tasks
            complete_track_classes: Track classes for complete task
            
        Returns:
            Instruction string
        """
        # Map UI task names to internal task names
        task_map = {
            "generate": "text2music",
            "repaint": "repaint",
            "cover": "cover",
            "add": "lego",
            "complete": "complete",
            "extract": "extract",
        }
        internal_task = task_map.get(task_type, "text2music")
        
        return self.dit_handler.generate_instruction(
            task_type=internal_task,
            track_name=track_name,
            complete_track_classes=complete_track_classes
        )
    
    def generate_audio(
        self,
        # Task type
        task_type: str,
        # Text inputs
        caption: str,
        lyrics: str,
        audio_codes: str,
        # Inference settings
        inference_steps: int,
        guidance_scale: float,
        seed: int,
        # Reference audio (for style guidance)
        reference_audio_path: Optional[str],
        # Source audio (for repaint, cover, add, complete, extract tasks)
        source_audio_path: Optional[str],
        # Repaint parameters
        repainting_start: float,
        repainting_end: float,
        # Cover parameters
        audio_cover_strength: float,
        # Meta
        bpm: Optional[int],
        key_scale: str,
        time_signature: str,
        vocal_language: str,
        # Advanced parameters
        use_adg: bool,
        cfg_interval_start: float,
        cfg_interval_end: float,
        audio_format: str,
        use_tiled_decode: bool,
        # Track parameters (for add/complete tasks)
        track_type: Optional[str] = None,
        # Progress callback
        progress=None
    ) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], str, str]:
        """
        Generate audio using DiT model.
        
        Returns:
            Tuple of (first_audio, second_audio, source_audio, mixed_1, mixed_2, status, actual_texts)
        """
        if self.dit_handler.model is None:
            return None, None, None, None, None, "❌ DiT model not initialized. Please load the DiT model first.", ""
        
        try:
            # Map UI task names to internal task names
            task_map = {
                "generate": "text2music",
                "repaint": "repaint",
                "cover": "cover",
                "add": "lego",
                "complete": "complete",
                "extract": "extract",
            }
            internal_task = task_map.get(task_type, "text2music")
            
            # Generate instruction based on task type
            instruction = self.dit_handler.generate_instruction(
                task_type=internal_task,
                track_name=track_type,  # For lego/extract tasks
                complete_track_classes=[track_type] if track_type and internal_task == "complete" else None
            )
            
            # Determine if using random seed
            use_random_seed = (seed == -1)
            
            # Call generate_music
            result = self.dit_handler.generate_music(
                captions=caption,
                lyrics=lyrics,
                bpm=bpm,
                key_scale=key_scale,
                time_signature=time_signature,
                vocal_language=vocal_language,
                inference_steps=inference_steps,
                guidance_scale=guidance_scale,
                use_random_seed=use_random_seed,
                seed=seed,
                reference_audio=reference_audio_path,  # Reference audio for style
                audio_duration=-1,  # Use default or derive from src_audio
                batch_size=2,
                src_audio=source_audio_path if internal_task in ["repaint", "cover", "lego", "complete", "extract"] else None,
                audio_code_string=audio_codes,
                repainting_start=repainting_start,
                repainting_end=repainting_end,
                instruction=instruction,
                audio_cover_strength=audio_cover_strength,
                task_type=internal_task,
                use_adg=use_adg,
                cfg_interval_start=cfg_interval_start,
                cfg_interval_end=cfg_interval_end,
                audio_format=audio_format,
                lm_temperature=1.0,
                use_tiled_decode=use_tiled_decode,
                progress=progress
            )
            
            # Unpack result from generate_music:
            # (first_audio, second_audio, all_audio_paths, generation_info, status_message, 
            #  seed_value_for_ui, actual_texts, align_score_1, align_text_1, align_plot_1, 
            #  align_score_2, align_text_2, align_plot_2)
            if result and len(result) >= 5:
                first_audio = result[0]  # first_audio
                second_audio = result[1]  # second_audio
                status = result[4]  # status_message
                actual_texts = result[6] if len(result) > 6 else ""
                actual_texts_str = ""
                if actual_texts and len(actual_texts) > 0:
                    actual_texts_str = actual_texts[0].replace("\\n", "\n")
                
                # For add/complete tasks, create mixed audio (source + generated)
                source_out = None
                mixed_1 = None
                mixed_2 = None
                
                if internal_task in ["lego", "complete"] and source_audio_path:
                    source_out = source_audio_path
                    mixed_1, mixed_2 = self._mix_audio(source_audio_path, first_audio, second_audio, audio_format)
                
                return first_audio, second_audio, source_out, mixed_1, mixed_2, status, actual_texts_str
            else:
                return None, None, None, None, None, "❌ Unexpected result format", ""
            
        except Exception as e:
            return None, None, None, None, None, f"❌ Error generating audio: {str(e)}\n{traceback.format_exc()}", ""
    
    def _mix_audio(
        self, 
        source_path: str, 
        audio1_path: Optional[str], 
        audio2_path: Optional[str],
        audio_format: str = "mp3"
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Mix source audio with generated audio.
        
        Args:
            source_path: Path to source audio file
            audio1_path: Path to first generated audio
            audio2_path: Path to second generated audio
            audio_format: Output audio format
            
        Returns:
            Tuple of (mixed_1_path, mixed_2_path)
        """
        try:
            from pydub import AudioSegment
            import tempfile
            
            source = AudioSegment.from_file(source_path)
            mixed_1_path = None
            mixed_2_path = None
            
            if audio1_path:
                gen1 = AudioSegment.from_file(audio1_path)
                # Overlay: mix the two audio tracks together
                mixed1 = source.overlay(gen1)
                mixed_1_path = tempfile.NamedTemporaryFile(
                    suffix=f".{audio_format}", 
                    delete=False
                ).name
                mixed1.export(mixed_1_path, format=audio_format)
            
            if audio2_path:
                gen2 = AudioSegment.from_file(audio2_path)
                mixed2 = source.overlay(gen2)
                mixed_2_path = tempfile.NamedTemporaryFile(
                    suffix=f".{audio_format}", 
                    delete=False
                ).name
                mixed2.export(mixed_2_path, format=audio_format)
            
            return mixed_1_path, mixed_2_path
        except ImportError:
            print("Warning: pydub not installed. Cannot mix audio. Install with: pip install pydub")
            return None, None
        except Exception as e:
            print(f"Error mixing audio: {e}")
            return None, None
    
    # =========================================================================
    # Status Properties
    # =========================================================================
    
    @property
    def is_llm_initialized(self) -> bool:
        """Check if LLM is initialized"""
        return self.llm_handler.llm_initialized
    
    @property
    def is_dit_initialized(self) -> bool:
        """Check if DiT model is initialized"""
        return self.dit_handler.model is not None
