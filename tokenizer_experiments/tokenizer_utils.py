from abc import ABC, abstractmethod
from typing import List

import sentencepiece as spm
from tokenizers import Tokenizer


class BaseTokenizerWrapper(ABC):
    """Abstract base class for unified tokenizer interface."""
    
    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        pass
    
    @abstractmethod
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        pass
    
    @abstractmethod
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        pass
    
    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """Return list of token strings (for analysis)."""
        pass


class BielikTokenizerWrapper(BaseTokenizerWrapper):
    """Wrapper for HuggingFace Bielik tokenizer."""
    
    def __init__(self, tokenizer_path: str):
        self.tokenizer = Tokenizer.from_file(f"{tokenizer_path}/tokenizer.json")
    
    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text).ids
    
    def decode(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids)
    
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()
    
    def tokenize(self, text: str) -> List[str]:
        return self.tokenizer.encode(text).tokens


class WhitespaceTokenizerWrapper(BaseTokenizerWrapper):
    """Wrapper for whitespace tokenizer."""
    
    def __init__(self, tokenizer_path: str):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
    
    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text).ids
    
    def decode(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids)
    
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()
    
    def tokenize(self, text: str) -> List[str]:
        return self.tokenizer.encode(text).tokens


class SentencePieceTokenizerWrapper(BaseTokenizerWrapper):
    """Wrapper for SentencePiece tokenizer."""
    
    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
    
    def encode(self, text: str) -> List[int]:
        return self.sp.encode(text, out_type=int)
    
    def decode(self, token_ids: List[int]) -> str:
        return self.sp.decode(token_ids)
    
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()
    
    def tokenize(self, text: str) -> List[str]:
        return self.sp.encode(text, out_type=str)


def load_tokenizer(tokenizer_type: str, config) -> BaseTokenizerWrapper:
    """Factory function to load the appropriate tokenizer."""
    
    if tokenizer_type == "bielik":
        return BielikTokenizerWrapper(config.bielik_tokenizer_path)
    elif tokenizer_type == "whitespace":
        return WhitespaceTokenizerWrapper(config.whitespace_tokenizer_path)
    elif tokenizer_type == "sentencepiece":
        return SentencePieceTokenizerWrapper(config.sentencepiece_model_path)
    else:
        raise ValueError(
            f"Unknown tokenizer type: {tokenizer_type}. "
            f"Must be one of: bielik, whitespace, sentencepiece"
        )
