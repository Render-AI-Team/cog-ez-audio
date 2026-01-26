import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Optional

# Cog imports
try:
    from cog import BasePredictor, Input, Path as CogPath
except ImportError:
    raise ImportError("Cog package is required. Install with: pip install cog")

import soundfile as sf
from api.ezaudio import EzAudio
from api.controlnet import EzAudio_ControlNet


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Initialize models lazily - will be loaded on first use
        self.ezaudio = None
        self.controlnet = None

    def predict(
        self,
        mode: str = Input(
            description="Generation mode",
            choices=["text_to_audio", "audio_editing", "audio_controlnet"],
            default="text_to_audio",
        ),
        prompt: str = Input(
            description="Text prompt for audio generation",
            default="a dog barking in the distance",
        ),
        model_size: str = Input(
            description="Model size for text-to-audio",
            choices=["s3_l", "s3_xl"],
            default="s3_xl",
        ),
        length: float = Input(
            description="Duration of generated audio in seconds (1-30)",
            default=10.0,
            ge=1.0,
            le=30.0,
        ),
        guidance_scale: float = Input(
            description="Guidance scale for generation quality",
            default=5.0,
            ge=0.0,
            le=20.0,
        ),
        guidance_rescale: float = Input(
            description="Guidance rescale for better conditioning",
            default=0.75,
            ge=0.0,
            le=1.0,
        ),
        ddim_steps: int = Input(
            description="Number of DDIM sampling steps",
            default=100,
            ge=10,
            le=200,
        ),
        eta: float = Input(
            description="Eta parameter for DDIM scheduler",
            default=1.0,
            ge=0.0,
            le=1.0,
        ),
        seed: int = Input(
            description="Random seed for reproducibility (-1 for random)",
            default=-1,
        ),
        # Audio editing parameters
        edit_audio: Optional[CogPath] = Input(
            description="Audio file for editing mode",
            default=None,
        ),
        mask_start: float = Input(
            description="Mask start time in seconds for editing",
            default=1.0,
            ge=0.0,
        ),
        mask_length: float = Input(
            description="Mask duration in seconds for editing",
            default=5.0,
            ge=0.1,
        ),
        boundary: float = Input(
            description="Boundary duration for smooth editing",
            default=2.0,
            ge=0.0,
        ),
        # ControlNet parameters
        controlnet_type: str = Input(
            description="Type of control for ControlNet mode",
            choices=["energy"],
            default="energy",
        ),
        reference_audio: Optional[CogPath] = Input(
            description="Reference audio file for ControlNet mode",
            default=None,
        ),
    ) -> CogPath:
        """Run a single prediction on the model"""
        
        # Validate seed
        if seed == -1:
            random_seed = None
            randomize_seed = True
        else:
            random_seed = seed
            randomize_seed = False

        # Generate audio based on mode
        if mode == "text_to_audio":
            return self._text_to_audio(
                prompt=prompt,
                model_size=model_size,
                length=length,
                guidance_scale=guidance_scale,
                guidance_rescale=guidance_rescale,
                ddim_steps=ddim_steps,
                eta=eta,
                random_seed=random_seed,
                randomize_seed=randomize_seed,
            )
        
        elif mode == "audio_editing":
            if edit_audio is None:
                raise ValueError("edit_audio parameter required for audio_editing mode")
            return self._audio_editing(
                prompt=prompt,
                edit_audio=str(edit_audio),
                mask_start=mask_start,
                mask_length=mask_length,
                boundary=boundary,
                guidance_scale=guidance_scale,
                guidance_rescale=guidance_rescale,
                ddim_steps=ddim_steps,
                eta=eta,
                random_seed=random_seed,
                randomize_seed=randomize_seed,
            )
        
        elif mode == "audio_controlnet":
            if reference_audio is None:
                raise ValueError("reference_audio parameter required for audio_controlnet mode")
            return self._controlnet_generation(
                prompt=prompt,
                controlnet_type=controlnet_type,
                reference_audio=str(reference_audio),
            )

    def _text_to_audio(
        self,
        prompt: str,
        model_size: str,
        length: float,
        guidance_scale: float,
        guidance_rescale: float,
        ddim_steps: int,
        eta: float,
        random_seed: Optional[int],
        randomize_seed: bool,
    ) -> CogPath:
        """Generate audio from text"""
        if self.ezaudio is None:
            print(f"Loading EzAudio model ({model_size})...")
            self.ezaudio = EzAudio(model_name=model_size, device=self.device)
        
        print(f"Generating audio for prompt: {prompt}")
        sr, audio = self.ezaudio.generate_audio(
            prompt,
            length=length,
            guidance_scale=guidance_scale,
            guidance_rescale=guidance_rescale,
            ddim_steps=ddim_steps,
            eta=eta,
            random_seed=random_seed,
            randomize_seed=randomize_seed,
        )
        
        # Save output
        output_path = "/tmp/output.wav"
        sf.write(output_path, audio, sr)
        return CogPath(output_path)

    def _audio_editing(
        self,
        prompt: str,
        edit_audio: str,
        mask_start: float,
        mask_length: float,
        boundary: float,
        guidance_scale: float,
        guidance_rescale: float,
        ddim_steps: int,
        eta: float,
        random_seed: Optional[int],
        randomize_seed: bool,
    ) -> CogPath:
        """Edit/inpaint audio with text guidance"""
        if self.ezaudio is None:
            print("Loading EzAudio model (s3_xl)...")
            self.ezaudio = EzAudio(model_name="s3_xl", device=self.device)
        
        print(f"Editing audio with prompt: {prompt}")
        sr, audio = self.ezaudio.editing_audio(
            prompt,
            boundary=boundary,
            gt_file=edit_audio,
            mask_start=mask_start,
            mask_length=mask_length,
            guidance_scale=guidance_scale,
            guidance_rescale=guidance_rescale,
            ddim_steps=ddim_steps,
            eta=eta,
            random_seed=random_seed,
            randomize_seed=randomize_seed,
        )
        
        # Save output
        output_path = "/tmp/output.wav"
        sf.write(output_path, audio, sr)
        return CogPath(output_path)

    def _controlnet_generation(
        self,
        prompt: str,
        controlnet_type: str,
        reference_audio: str,
    ) -> CogPath:
        """Generate audio with ControlNet conditioning"""
        if self.controlnet is None:
            print(f"Loading EzAudio-ControlNet model ({controlnet_type})...")
            self.controlnet = EzAudio_ControlNet(
                model_name=controlnet_type,
                device=self.device
            )
        
        print(f"Generating audio with ControlNet: {prompt}")
        sr, audio = self.controlnet.generate_audio(
            prompt,
            audio_path=reference_audio,
        )
        
        # Save output
        output_path = "/tmp/output.wav"
        sf.write(output_path, audio, sr)
        return CogPath(output_path)
