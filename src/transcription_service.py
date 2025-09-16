"""
Transcription service for WhisperX-based audio transcription with speaker diarization
Adapted for the LLM service
"""

import os
import torch
from typing import Optional, Dict, Any, List
from pathlib import Path

from config import Config
from simple_logger import log_action
from storage_service import StorageService


# Simple logging setup for the LLM service
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TranscriptionService:
    """Service for handling audio transcription with WhisperX"""
    
    def __init__(self, config: Config, use_gpu: bool = True):
        """
        Initialize transcription service
        
        Args:
            config: Application configuration
            use_gpu: Whether to use GPU acceleration if available
        """
        self.config = config
        
        # Determine device and compute type based on availability and preference
        if use_gpu and torch.cuda.is_available():
            self.device = config.DEVICE if config.DEVICE in ['cuda', 'cpu'] else "cuda"
            self.compute_type = config.COMPUTE_TYPE
            logger.info("GPU acceleration enabled for transcription")
        else:
            self.device = "cpu"
            self.compute_type = "int8"
            if use_gpu:
                logger.warning("GPU requested but not available, falling back to CPU")
            else:
                logger.info("Using CPU for transcription")
        
        # Initialize WhisperX model lazily
        self.model = None
        self.model_a = None
        self.metadata = None

        # Initialize storage service
        self.storage_service = StorageService(config.OUTPUT_DIR)

        logger.info(f"TranscriptionService initialized with device: {self.device}")
    
    def _load_model(self):
        """Lazy load the WhisperX model"""
        if self.model is None:
            try:
                import whisperx
                logger.info(f"Loading WhisperX model: {self.config.WHISPER_MODEL}")
                
                self.model = whisperx.load_model(
                    self.config.WHISPER_MODEL, 
                    self.device, 
                    compute_type=self.compute_type
                )
                
                # Load alignment model
                self.model_a, self.metadata = whisperx.load_align_model(
                    language_code="en", 
                    device=self.device
                )
                
                logger.info("WhisperX models loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load WhisperX model: {e}")
                raise RuntimeError(f"Model loading failed: {e}")
    
    def transcribe_audio(
        self, 
        audio_path: str | Path, 
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        batch_size: int = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Transcribe audio file with optional speaker diarization
        
        Args:
            audio_path: Path to audio file
            min_speakers: Minimum number of speakers for diarization
            max_speakers: Maximum number of speakers for diarization
            batch_size: Batch size for processing
            verbose: Whether to log detailed progress
            
        Returns:
            Dictionary containing transcription results with segments and metadata
        """
        audio_path = str(audio_path)
        batch_size = batch_size or self.config.BATCH_SIZE
        
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        log_action(f"Starting audio transcription for file: {audio_path}")
        
        if verbose:
            logger.info(f"Starting transcription of: {audio_path}")
            logger.info(f"Device: {self.device}, Batch size: {batch_size}")
            if min_speakers or max_speakers:
                logger.info(f"Speaker diarization enabled: min={min_speakers}, max={max_speakers}")
        
        try:
            # Lazy load model
            self._load_model()
            
            import whisperx
            
            # Load audio
            audio = whisperx.load_audio(audio_path)
            
            # Transcribe
            result = self.model.transcribe(audio, batch_size=batch_size)
            
            # Align
            result = whisperx.align(
                result["segments"], 
                self.model_a, 
                self.metadata, 
                audio, 
                self.device, 
                return_char_alignments=False
            )
            
            # Diarization if requested
            if min_speakers or max_speakers:
                try:
                    diarize_model = whisperx.DiarizationPipeline(
                        use_auth_token=self.config.HF_TOKEN, 
                        device=self.device
                    )
                    
                    diarize_segments = diarize_model(
                        audio, 
                        min_speakers=min_speakers, 
                        max_speakers=max_speakers
                    )
                    
                    result = whisperx.assign_word_speakers(diarize_segments, result)
                    
                except Exception as e:
                    logger.warning(f"Speaker diarization failed: {e}")
            
            # Add metadata about transcription process
            result['metadata'] = {
                'device_used': self.device,
                'compute_type': self.compute_type,
                'model': self.config.WHISPER_MODEL,
                'batch_size': batch_size,
                'speakers_detected': self._count_unique_speakers(result),
                'audio_file': audio_path
            }
            
            if verbose:
                logger.info(f"Transcription completed successfully")
                logger.info(f"Language detected: {result.get('language', 'unknown')}")
                logger.info(f"Segments generated: {len(result.get('segments', []))}")
                if result['metadata']['speakers_detected'] > 0:
                    logger.info(f"Speakers detected: {result['metadata']['speakers_detected']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Transcription failed for {audio_path}: {str(e)}")
            raise RuntimeError(f"Transcription failed: {str(e)}")
    
    def _count_unique_speakers(self, result: Dict[str, Any]) -> int:
        """Count unique speakers in the transcription result"""
        speakers = set()
        for segment in result.get('segments', []):
            if 'speaker' in segment:
                speakers.add(segment['speaker'])
            # Also check words for speaker info
            for word in segment.get('words', []):
                if 'speaker' in word:
                    speakers.add(word['speaker'])
        return len(speakers)
    
    def is_available(self) -> bool:
        """Check if transcription service is available (all dependencies installed)"""
        try:
            import whisperx
            import torch
            return True
        except ImportError as e:
            logger.warning(f"Transcription service not available: {e}")
            return False
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get information about the compute device being used"""
        info = {
            'device': self.device,
            'compute_type': self.compute_type,
            'torch_version': torch.__version__,
        }
        
        if self.device == 'cuda':
            info.update({
                'cuda_available': torch.cuda.is_available(),
                'cuda_version': torch.version.cuda,
                'gpu_count': torch.cuda.device_count(),
                'current_gpu': torch.cuda.current_device() if torch.cuda.is_available() else None,
                'gpu_name': torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                'gpu_memory': {
                    'allocated': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
                    'cached': torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
                }
            })
        
        return info

    def format_for_llm(
        self,
        transcription_result: Dict[str, Any],
        output_format: str = "simple",
        include_speakers: bool = False,
        merge_consecutive_speakers: bool = True,
        remove_filler_words: bool = False
    ) -> Dict[str, Any]:
        """
        Format transcription result for LLM consumption

        Args:
            transcription_result: The full transcription result from WhisperX
            output_format: Format type ("simple", "speaker", "structured", "markdown")
            include_speakers: Whether to include speaker information
            merge_consecutive_speakers: Whether to merge consecutive segments from same speaker
            remove_filler_words: Whether to remove common filler words

        Returns:
            Dictionary with LLM-optimized format
        """
        segments = transcription_result.get('segments', [])
        language = transcription_result.get('language', 'unknown')

        if not segments:
            return {
                "success": True,
                "text": "",
                "language": language,
                "metadata": {"word_count": 0}
            }

        # Process segments based on format
        if output_format == "simple":
            return self._format_simple(segments, language, remove_filler_words)
        elif output_format == "speaker" or include_speakers:
            return self._format_speaker_aware(segments, language, merge_consecutive_speakers, remove_filler_words)
        elif output_format == "structured":
            return self._format_structured(segments, language, merge_consecutive_speakers, remove_filler_words)
        elif output_format == "markdown":
            return self._format_markdown(segments, language, merge_consecutive_speakers, remove_filler_words)
        else:
            return self._format_simple(segments, language, remove_filler_words)

    def _format_simple(self, segments: List[Dict], language: str, remove_filler_words: bool = False) -> Dict[str, Any]:
        """Format as simple concatenated text"""
        texts = []

        for segment in segments:
            text = segment.get('text', '').strip()
            if text:
                if remove_filler_words:
                    text = self._remove_filler_words(text)
                texts.append(text)

        full_text = ' '.join(texts).strip()

        return {
            "success": True,
            "text": full_text,
            "language": language,
            "metadata": {
                "word_count": len(full_text.split()) if full_text else 0,
                "segment_count": len(segments),
                "format": "simple"
            }
        }

    def _format_speaker_aware(
        self,
        segments: List[Dict],
        language: str,
        merge_consecutive: bool = True,
        remove_filler_words: bool = False
    ) -> Dict[str, Any]:
        """Format with speaker information"""
        speakers = {}
        current_speaker = None
        current_text = []

        for segment in segments:
            text = segment.get('text', '').strip()
            speaker = segment.get('speaker', 'UNKNOWN')

            if not text:
                continue

            if remove_filler_words:
                text = self._remove_filler_words(text)

            if merge_consecutive and speaker == current_speaker:
                current_text.append(text)
            else:
                # Save previous speaker's text
                if current_speaker and current_text:
                    if current_speaker not in speakers:
                        speakers[current_speaker] = []
                    speakers[current_speaker].append(' '.join(current_text))

                # Start new speaker
                current_speaker = speaker
                current_text = [text]

        # Don't forget the last speaker
        if current_speaker and current_text:
            if current_speaker not in speakers:
                speakers[current_speaker] = []
            speakers[current_speaker].append(' '.join(current_text))

        # Merge all text parts for each speaker
        for speaker in speakers:
            speakers[speaker] = ' '.join(speakers[speaker])

        return {
            "success": True,
            "speakers": speakers,
            "language": language,
            "metadata": {
                "speaker_count": len(speakers),
                "segment_count": len(segments),
                "format": "speaker"
            }
        }

    def _format_structured(
        self,
        segments: List[Dict],
        language: str,
        merge_consecutive: bool = True,
        remove_filler_words: bool = False
    ) -> Dict[str, Any]:
        """Format as structured conversation blocks"""
        blocks = []
        current_speaker = None
        current_text = []

        for segment in segments:
            text = segment.get('text', '').strip()
            speaker = segment.get('speaker', 'UNKNOWN')

            if not text:
                continue

            if remove_filler_words:
                text = self._remove_filler_words(text)

            if merge_consecutive and speaker == current_speaker:
                current_text.append(text)
            else:
                # Save previous speaker's block
                if current_speaker and current_text:
                    blocks.append({
                        "speaker": current_speaker,
                        "text": ' '.join(current_text)
                    })

                # Start new speaker
                current_speaker = speaker
                current_text = [text]

        # Don't forget the last speaker
        if current_speaker and current_text:
            blocks.append({
                "speaker": current_speaker,
                "text": ' '.join(current_text)
            })

        return {
            "success": True,
            "blocks": blocks,
            "language": language,
            "metadata": {
                "block_count": len(blocks),
                "segment_count": len(segments),
                "format": "structured"
            }
        }

    def _format_markdown(
        self,
        segments: List[Dict],
        language: str,
        merge_consecutive: bool = True,
        remove_filler_words: bool = False
    ) -> Dict[str, Any]:
        """Format as markdown with speaker headings"""
        markdown_parts = []
        current_speaker = None
        current_text = []
        speaker_counter = {}

        for segment in segments:
            text = segment.get('text', '').strip()
            speaker = segment.get('speaker', 'UNKNOWN')

            if not text:
                continue

            if remove_filler_words:
                text = self._remove_filler_words(text)

            if merge_consecutive and speaker == current_speaker:
                current_text.append(text)
            else:
                # Save previous speaker's section
                if current_speaker and current_text:
                    # Create readable speaker name
                    if current_speaker not in speaker_counter:
                        speaker_counter[current_speaker] = len(speaker_counter) + 1

                    speaker_name = f"Speaker {speaker_counter[current_speaker]}"
                    markdown_parts.append(f"## {speaker_name}\n\n{' '.join(current_text)}\n")

                # Start new speaker
                current_speaker = speaker
                current_text = [text]

        # Don't forget the last speaker
        if current_speaker and current_text:
            if current_speaker not in speaker_counter:
                speaker_counter[current_speaker] = len(speaker_counter) + 1

            speaker_name = f"Speaker {speaker_counter[current_speaker]}"
            markdown_parts.append(f"## {speaker_name}\n\n{' '.join(current_text)}\n")

        markdown_text = '\n'.join(markdown_parts).strip()

        return {
            "success": True,
            "text": markdown_text,
            "language": language,
            "metadata": {
                "speaker_count": len(speaker_counter),
                "segment_count": len(segments),
                "format": "markdown"
            }
        }

    def _remove_filler_words(self, text: str) -> str:
        """Remove common filler words"""
        filler_words = {
            'um', 'uh', 'er', 'ah', 'em', 'hmm', 'mm', 'mhm',
            'erm', 'uh-huh', 'mm-hmm', 'eh', 'eh?'
        }

        words = text.split()
        filtered_words = [word for word in words if word.lower().strip('.,!?;:') not in filler_words]
        return ' '.join(filtered_words)

    def save_transcription_to_disk(
        self,
        media_filename: str,
        transcription_result: Dict[str, Any],
        llm_result: Dict[str, Any]
    ) -> Optional[str]:
        """
        Save transcription to persistent storage

        Args:
            media_filename: Original media filename/URL
            transcription_result: Raw transcription result from WhisperX
            llm_result: Formatted LLM result

        Returns:
            Path to saved file if successful, None if failed
        """
        try:
            # Determine transcription text based on format
            transcription_text = self._extract_transcription_text(llm_result)

            # Combine metadata from both results
            combined_metadata = self._combine_metadata(transcription_result, llm_result)

            # Save to disk
            file_path = self.storage_service.save_transcription(
                media_filename=media_filename,
                transcription_text=transcription_text,
                metadata=combined_metadata,
                language=transcription_result.get('language')
            )

            logger.info(f"Transcription saved to disk: {file_path}")
            return file_path

        except Exception as e:
            logger.error(f"Failed to save transcription to disk: {str(e)}")
            return None

    def _extract_transcription_text(self, llm_result: Dict[str, Any]) -> str:
        """
        Extract formatted transcription text from LLM result

        Args:
            llm_result: Formatted LLM result

        Returns:
            Formatted transcription text with speaker indicators if applicable
        """
        # Handle different output formats
        if "text" in llm_result:
            return llm_result["text"]
        elif "speakers" in llm_result:
            # Format speakers as numbered sections
            speakers_dict = llm_result["speakers"]
            if speakers_dict:
                parts = []
                speaker_mapping = {}
                speaker_counter = 1

                for speaker_id, text in speakers_dict.items():
                    if speaker_id not in speaker_mapping:
                        speaker_mapping[speaker_id] = speaker_counter
                        speaker_counter += 1

                    speaker_num = speaker_mapping[speaker_id]
                    parts.append(f"Speaker {speaker_num}: {text}")

                return "\n\n".join(parts)
        elif "blocks" in llm_result:
            # Format blocks as conversation
            blocks = llm_result["blocks"]
            if blocks:
                parts = []
                speaker_mapping = {}
                speaker_counter = 1

                for block in blocks:
                    speaker_id = block.get("speaker", "UNKNOWN")
                    text = block.get("text", "")

                    if speaker_id not in speaker_mapping:
                        speaker_mapping[speaker_id] = speaker_counter
                        speaker_counter += 1

                    speaker_num = speaker_mapping[speaker_id]
                    parts.append(f"Speaker {speaker_num}: {text}")

                return "\n\n".join(parts)

        return ""

    def _combine_metadata(
        self,
        transcription_result: Dict[str, Any],
        llm_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Combine metadata from transcription and LLM results

        Args:
            transcription_result: Raw transcription result
            llm_result: Formatted LLM result

        Returns:
            Combined metadata dictionary
        """
        combined = {}

        # Add transcription metadata
        if "metadata" in transcription_result:
            combined.update(transcription_result["metadata"])

        # Add LLM metadata
        if "metadata" in llm_result:
            llm_metadata = llm_result["metadata"]
            combined.update(llm_metadata)

        # Add format-specific information
        if "speakers" in llm_result:
            combined["speaker_count"] = len(llm_result["speakers"]) if llm_result["speakers"] else 0
        if "blocks" in llm_result:
            combined["block_count"] = len(llm_result["blocks"]) if llm_result["blocks"] else 0

        return combined