import os
import math
import torch
import pickle
import random
import string
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence, Tuple, Union
from torch.utils.data import Dataset
from constant import proteinseq_toks

RawMSA = Sequence[Tuple[str, str]]

class Alphabet(object):
    def __init__(
        self,
        standard_toks: Sequence[str],
        prepend_toks: Sequence[str] = ("<null_0>", "<pad>", "<eos>", "<unk>"),
        append_toks: Sequence[str] = ("<cls>", "<mask>", "<sep>"),
        tokenizer: Sequence[str] = ("rna", "rna-3mer", "rna-3-mixmer"),
        prepend_bos: bool = True,
        append_eos: bool = False,
        use_msa: bool = False,
        k_mer: int = 1,
    ):
        self.standard_toks = list(standard_toks)
        self.prepend_toks = list(prepend_toks)
        self.append_toks = list(append_toks)
        self.tokenizer = tokenizer
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos
        self.use_msa = use_msa
        self.k_mer = k_mer

        self.all_toks = list(self.prepend_toks)
        self.all_toks.extend(self.standard_toks)
        for i in range((8 - (len(self.all_toks) % 8)) % 8):
            self.all_toks.append(f"<null_{i  + 1}>")
        self.all_toks.extend(self.append_toks)

        self.tok_to_idx = {tok: i for i, tok in enumerate(self.all_toks)}

        self.unk_idx = self.tok_to_idx["<unk>"]
        self.padding_idx = self.get_idx("<pad>")
        self.cls_idx = self.get_idx("<cls>")
        self.mask_idx = self.get_idx("<mask>")
        self.eos_idx = self.get_idx("<eos>")
        
        self.words = ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O', '-']

    def __len__(self):
        return len(self.all_toks)

    def get_idx(self, tok):
        return self.tok_to_idx.get(tok, self.unk_idx)

    def get_tok(self, idx):
        return self.all_toks[idx]
    
    def decode(self, input: torch.Tensor, skip_special_tokens: bool):
        decoder_token = [self.get_tok(token) for token in input]
        if skip_special_tokens:
            output_seq = ''.join(char for char in decoder_token if char[0] in self.words)
        else:
            output_seq = ''.join(char for char in decoder_token)
        return output_seq

    def to_dict(self):
        return {"toks": self.toks}

    def save_pretrained(self, output_dir):
        pass

    @classmethod
    def from_architecture(cls, name: str, theme: str = "protein") -> "Alphabet":
        tokenizer = theme
        seq_toks = proteinseq_toks
        k_mer = 1

        if name in ("ESM-1", "protein_bert_base"):
            standard_toks = seq_toks["toks"]
            prepend_toks: Tuple[str, ...] = ("<null_0>", "<pad>", "<eos>", "<unk>")
            append_toks: Tuple[str, ...] = ("<cls>", "<mask>", "<sep>")
            prepend_bos = True
            append_eos = False
            use_msa = False
        elif name in ("ESM-1b", "roberta_large"):
            standard_toks = seq_toks["toks"]
            prepend_toks = ("<cls>", "<pad>", "<eos>", "<unk>")
            append_toks = ("<mask>",)
            prepend_bos = True
            append_eos = True
            use_msa = False
        elif name in ("MSA Transformer", "msa_transformer"):
            standard_toks = seq_toks["toks"]
            prepend_toks = ("<cls>", "<pad>", "<eos>", "<unk>")
            append_toks = ("<mask>",)
            prepend_bos = True
            append_eos = False
            use_msa = True
        elif "invariant_gvp" in name.lower():
            standard_toks = seq_toks["toks"]
            prepend_toks = ("<null_0>", "<pad>", "<eos>", "<unk>")
            append_toks = ("<mask>", "<cath>", "<af2>")
            prepend_bos = True
            append_eos = False
            use_msa = False
        else:
            raise ValueError("Unknown architecture selected")
        return cls(standard_toks, prepend_toks, append_toks, tokenizer, prepend_bos, append_eos, use_msa, k_mer)

class BatchConverter(object):
    """Callable to convert an unprocessed (labels + strings) batch to a
    processed (labels + tensor) batch.
    """

    def __init__(self, alphabet):
        self.alphabet = alphabet
        self.k_mer = self.alphabet.k_mer

    def __call__(self, raw_batch: Sequence[Tuple[str, str]]):
        # RoBERTa uses an eos token, while ESM-1 does not.
        batch_size = len(raw_batch)
        max_len = max(len(seq_str) // self.k_mer for seq_str in raw_batch)
        
        tokens = torch.empty(
            (
                batch_size,
                max_len
                + int(self.alphabet.prepend_bos)
                + int(self.alphabet.append_eos),
            ),
            dtype=torch.int64,
        )
        tokens.fill_(self.alphabet.padding_idx)

        for i, seq_str in enumerate(raw_batch):
            if self.alphabet.prepend_bos:
                tokens[i, 0] = self.alphabet.cls_idx
            seq = torch.tensor(
                #[self.alphabet.get_idx(s) for s in seq_str], dtype=torch.int64
                [self.alphabet.get_idx(seq_str[i:i + self.k_mer]) for i in range(0, len(seq_str), self.k_mer)], dtype=torch.int64
            )

            if self.alphabet.tokenizer == "rna-3-mixmer":
                tokens[
                    i,
                    int(self.alphabet.prepend_bos) : math.ceil(len(seq_str) / self.k_mer)
                                                    + int(self.alphabet.prepend_bos),
                ] = seq
            else:
                tokens[
                    i,
                    int(self.alphabet.prepend_bos) : len(seq_str) // self.k_mer
                                                    + int(self.alphabet.prepend_bos),
                ] = seq[:-1] if len(seq_str) % self.k_mer != 0 else seq # discard or not
            if self.alphabet.append_eos: # False
                tokens[
                    i, len(seq_str) // self.k_mer + int(self.alphabet.prepend_bos)
                ] = self.alphabet.eos_idx

        return tokens
    
    def esm_convert(self, raw_batch: torch.Tensor):
        batch_size = raw_batch.shape[0]
        max_len = raw_batch.shape[1] // self.k_mer
        
        tokens = torch.empty(
            (
                batch_size,
                max_len,
            ),
        )
        tokens.fill_(self.alphabet.padding_idx)
        tokens[:, :max_len, ] = raw_batch

        return tokens
    
    def seq_convert(self, num_alignments: int, raw_batch: torch.Tensor):
        raw_batch = raw_batch.repeat(num_alignments, 1, 1)
        
        batch_size = raw_batch.shape[0]
        max_len = raw_batch.shape[1] // self.k_mer
        
        tokens = torch.empty(
            (
                batch_size,
                max_len,
            ),
        )
        tokens.fill_(self.alphabet.padding_idx)
        tokens[:, :max_len, ] = torch.rand(raw_batch.shape[:2])

        return tokens


class MSABatchConverter(BatchConverter):
    def msa_batch_convert(self, inputs: Union[Sequence[RawMSA], RawMSA]):
        if len(inputs[0][0]) == 1:
            # Input is a single MSA
            raw_batch: Sequence[RawMSA] = [inputs]  # type: ignore
        else:
            raw_batch = inputs  # type: ignore

        batch_size = len(raw_batch) # 1
        max_alignments = max(len(msa) for msa in raw_batch)
        max_seqlen = max(len(seq) // self.k_mer for msa in raw_batch for seq in msa) # 15

        # if self.alphabet.tokenizer == "rna-3-mixmer":
        #     max_seqlen = max(math.ceil(len(seq) / self.k_mer) for msa in raw_batch for seq in msa)
        # else:
        #     max_seqlen = max(len(seq) // self.k_mer for msa in raw_batch for seq in msa)
        
        tokens = torch.empty(
            (
                batch_size,
                max_alignments,
                max_seqlen
                + int(self.alphabet.prepend_bos)
                + int(self.alphabet.append_eos),
            ),
            dtype=torch.int64,
        )
        tokens.fill_(self.alphabet.padding_idx)

        for i, msa in enumerate(raw_batch):
            msa_seqlens = set(len(seq) for seq in msa)
            if not len(msa_seqlens) == 1:
                raise RuntimeError(
                    "Received unaligned sequences for input to MSA, all sequence "
                    "lengths must be equal."
                )
            msa_tokens = super().__call__(msa)
            tokens[i, : msa_tokens.size(0), : msa_tokens.size(1)] = msa_tokens

        return tokens

    def esm_batch_convert(self, inputs: Union[Sequence[RawMSA], RawMSA], labels: torch.Tensor):
        if len(inputs[0][0]) != 1280:
            # Input is a single ESM Embedding
            raw_batch: Sequence[RawMSA] = [inputs]  # type: ignore
        else:
            raw_batch = inputs  # type: ignore
        
        batch_size = len(raw_batch)
        num_alignments = labels.shape[1]
        max_seqlen = labels.shape[-1] - int(self.alphabet.prepend_bos) - int(self.alphabet.append_eos)

        tokens = torch.empty(
            (
                batch_size,
                num_alignments,
                max_seqlen
                + int(self.alphabet.prepend_bos)
                + int(self.alphabet.append_eos),
            ),
        )
        tokens.fill_(self.alphabet.padding_idx)
        
        esm_embeddings = torch.empty(
            (
                batch_size,
                max_seqlen,
                1280
            )
        )
        esm_embeddings.fill_(self.alphabet.padding_idx)
        
        for i, esm in enumerate(raw_batch):
            esm_tokens = super().esm_convert(esm)
            seq_tokens = super().seq_convert(num_alignments, esm)
            tokens[i, : seq_tokens.size(0), : seq_tokens.size(1)] = seq_tokens
            esm_embeddings[i, : esm_tokens.size(0), : esm_tokens.size(1)] = esm_tokens
        
        return esm_embeddings, tokens
    
    def infer_batch_convert(self, inputs: Union[str, torch.Tensor], num_alignments: int):
        seq = inputs[0]
        esm = inputs[1]
        batch_size = 1
        max_seqlen = len(seq)

        tokens = torch.empty(
            (
                batch_size,
                num_alignments,
                max_seqlen
                + int(self.alphabet.prepend_bos)
                + int(self.alphabet.append_eos),
            ),
        )
        tokens.fill_(self.alphabet.padding_idx)
        
        esm_embeddings = torch.empty(
            (
                batch_size,
                max_seqlen,
                1280
            )
        )
        esm_embeddings.fill_(self.alphabet.padding_idx)
        
        esm_tokens = super().esm_convert(esm)
        seq_tokens = super().seq_convert(num_alignments, esm)
        tokens[0, : seq_tokens.size(0), : seq_tokens.size(1)] = seq_tokens
        esm_embeddings[0, : esm_tokens.size(0), : esm_tokens.size(1)] = esm_tokens
        
        return esm_embeddings, tokens

    def __call__(self, batch):
        labels = self.msa_batch_convert([example["msa"] for example in batch])
        esm_emb, input_ids = self.esm_batch_convert([example["emb"] for example in batch], labels)
        labels[labels==self.alphabet.padding_idx]=-100
        attention_mask = input_ids.ne(self.alphabet.padding_idx).type_as(input_ids)
        decoder_attention_mask = labels.ne(self.alphabet.padding_idx).type_as(input_ids)
        return {'input_ids':input_ids, 'labels':labels, "attention_mask":attention_mask, "decoder_attention_mask":decoder_attention_mask, "esm": esm_emb}

class MSADataSet(Dataset):
    def __init__(self, data_args, num_alignments, threshold):
        super().__init__()
        self.data_args = data_args
        self.data_path = self.data_args.train_file

        if os.path.isdir(self.data_path):
            self.data = []
            
            datas = [os.path.join(self.data_path, data) for data in os.listdir(self.data_path)]
        
            with ThreadPoolExecutor() as executor:
                results = list(tqdm(executor.map(self.read_pickle, datas), total=len(datas)))
            
            for result in results:
                if result is not None and len(result['seq']) <= threshold:
                    result['msa'] = random.sample(result['msa'], num_alignments)
                    self.data.append(result) # name msa emb seq
        else:
            with open(self.data_path, "rb") as f:
                self.data = pickle.load(f)
        
    def read_pickle(self, path):
        with open(path, "rb") as f:
            try:
                protein_data = pickle.load(f, encoding='bytes')
                return protein_data
            except Exception as e:
                print(f"Error loading {path}: {e}")
                return None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

class MSAInferenceDataSet(Dataset):
    def __init__(self, args):
        super().__init__()
        
        self.data = []
        for filename in os.listdir(args.data_path):
            data_path = os.path.join(args.data_path, filename)
            
            with open(data_path, "rb") as f:
                data = pickle.load(f)
        
            self.data.append(data) # name, seq, emb

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]