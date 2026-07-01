import numpy as np
import pandas as pd

class GenomicDataLoader:
    def __init__(self) -> None:
        self.nucleotide_map = {
            'A': [1, 0, 0, 0], 
            'C': [0, 1, 0, 0], 
            'G': [0, 0, 1, 0], 
            'T': [0, 0, 0, 1], 
            'N': [0, 0, 0, 0]
        }

    def one_hot_encode(self, seq: str) -> np.ndarray:
        clean_seq = seq.upper().replace('U', 'T')
        encoded = []
        for char in clean_seq:
            encoded.extend(self.nucleotide_map.get(char, [0, 0, 0, 0]))
        return np.array(encoded, dtype=np.float32)

    def generate_synthetic_benchmark(self, n_samples: int = 2000) -> pd.DataFrame:
        np.random.seed(42)
        bases = ['A', 'C', 'G', 'T']
        data = []
        for _ in range(n_samples):
            target = ''.join(np.random.choice(bases, 20))
            mismatches = int(np.random.randint(0, 5))
            grna_list = list(target)
            if mismatches > 0:
                indices = np.random.choice(20, mismatches, replace=False)
                for idx in indices:
                    current_base = grna_list[idx]
                    mutations = [b for b in bases if b != current_base]
                    grna_list[idx] = str(np.random.choice(mutations))
            grna = ''.join(grna_list)
            is_off_target = 1 if mismatches <= 2 else 0
            if np.random.rand() < 0.05:  
                is_off_target = 1 - is_off_target
            data.append({"grna": grna, "target": target, "label": is_off_target})
        return pd.DataFrame(data)

    def prepare_features(self, df: pd.DataFrame):
        X = []
        for _, row in df.iterrows():
            g_enc = self.one_hot_encode(str(row['grna']))
            t_enc = self.one_hot_encode(str(row['target']))
            mismatch_vector = np.abs(g_enc - t_enc)
            X.append(np.concatenate([g_enc, t_enc, mismatch_vector]))
        return np.array(X, dtype=np.float32), np.array(df['label'].values, dtype=np.int32)