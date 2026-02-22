import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import pandas as pd

def load_and_pad_embeddings(pt_file_path):
    """
    GAT'tan gelen pt dosyasını yükler ve 0. indekse PAD vektörü ekler.
    """
    # pt dosyasını yükle
    data = torch.load(pt_file_path)
    embeddings = data['embeddings']
    
    book_embs = embeddings['book']   
    movie_embs = embeddings['movie'] 
    user_embs = embeddings['user']   
    embed_dim = book_embs.shape[1]   
    
    # Her biri için en başa (0. indeks) sıfır vektörü ekle
    pad_vector = torch.zeros(1, embed_dim)
    
    padded_book_embs = torch.cat([pad_vector, book_embs], dim=0)
    padded_movie_embs = torch.cat([pad_vector, movie_embs], dim=0)
    # User için padding'e genelde gerek olmaz (user'lar dizi/sequence oluşturmaz), 
    # ama indeks uyumluluğu için yapabiliriz.
    padded_user_embs = torch.cat([pad_vector, user_embs], dim=0)
    
    return padded_user_embs, padded_book_embs, padded_movie_embs


# RECENT-N SLIDING WINDOW
class CrossDomainDataset(Dataset):
    def __init__(self, book_inter_path, movie_inter_path, 
                 book_mapping_path, movie_mapping_path, user_mapping_path, 
                 max_seq_len=10, mode='train', 
                 train_movie_inter_path=None):
        self.max_seq_len = max_seq_len
        self.mode = mode
        
        with open(book_mapping_path, 'r') as f:
            self.book_mapping = json.load(f)
        with open(movie_mapping_path, 'r') as f:
            self.movie_mapping = json.load(f)
        with open(user_mapping_path, 'r') as f:
            self.user_mapping = json.load(f)
            
        # Kitap geçmişi (Source Domain) - Statik kalabilir
        self.book_df = pd.read_csv(book_inter_path, sep='\t')
        self.user_book_history = self._build_history(self.book_df, self.book_mapping)
        
        # Movie geçmişi ve Örneklerin Hazırlanması
        self.samples = [] # Tüm (User, History, Target) çiftlerini burada tutacağız
        
        if self.mode == 'train':
            self.movie_df = pd.read_csv(movie_inter_path, sep='\t')
            # Kullanıcı bazlı gruplayıp her kullanıcı için pencere oluşturuyoruz
            grouped = self.movie_df.sort_values('timestamp:float').groupby('user_id:token')
            
            for user_str, group in grouped:
                if user_str not in self.user_mapping: continue
                user_id = self.user_mapping[user_str]
                
                # Kullanıcının tüm film geçmişini al (+1 ekleyerek)
                full_movie_history = [self.movie_mapping[tid] + 1 for tid in group['item_id:token'].values if tid in self.movie_mapping]
                
                # RECENT-N SLIDING WINDOW MANTIĞI:
                # Örn: [1, 2, 3, 4, 5, 6] izlemişse ve N=3 ise:
                # 1. [1, 2, 3] -> Hedef: 4
                # 2. [1, 2, 3, 4] -> Hedef: 5
                # 3. [1, 2, 3, 4, 5] -> Hedef: 6
                N = 3
                for i in range(max(1, len(full_movie_history) - N), len(full_movie_history)):
                    history = full_movie_history[:i]
                    target = full_movie_history[i]
                    self.samples.append({
                        'user_id': user_id,
                        'movie_history': history,
                        'target_movie_id': target
                    })
            print(f"Eğitim için Sliding Window ile üretilen toplam örnek sayısı: {len(self.samples)}")
            
        else:
            # Valid/Test Modu: Sadece tek bir hedef (dosyadan gelen)
            assert train_movie_inter_path is not None
            train_movie_df = pd.read_csv(train_movie_inter_path, sep='\t')
            self.user_movie_history_train = self._build_history(train_movie_df, self.movie_mapping)
            
            target_df = pd.read_csv(movie_inter_path, sep='\t')
            for _, row in target_df.iterrows():
                u_str, i_str = row['user_id:token'], row['item_id:token']
                if u_str in self.user_mapping and i_str in self.movie_mapping:
                    user_id = self.user_mapping[u_str]
                    self.samples.append({
                        'user_id': user_id,
                        'movie_history': self.user_movie_history_train.get(user_id, []),
                        'target_movie_id': self.movie_mapping[i_str] + 1
                    })
            print(f"Test/Validasyon için toplam örnek sayısı: {len(self.samples)}")

    def _build_history(self, df, item_mapping):
        history = {}
        df_sorted = df.sort_values(by=['user_id:token', 'timestamp:float'])
        grouped = df_sorted.groupby('user_id:token')
        for user_id_str, group in grouped:
            if user_id_str not in self.user_mapping: continue
            items = [item_mapping[tid] + 1 for tid in group['item_id:token'].values if tid in item_mapping]
            history[self.user_mapping[user_id_str]] = items
        return history

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        user_id = sample['user_id']
        target_movie_id = sample['target_movie_id']
        
        # Geçmişleri al ve maskele
        book_seq = self.user_book_history.get(user_id, []).copy()[-self.max_seq_len:]
        movie_seq = sample['movie_history'][-self.max_seq_len:]
        
        # Padding
        book_seq = book_seq + [0] * (self.max_seq_len - len(book_seq))
        movie_seq = movie_seq + [0] * (self.max_seq_len - len(movie_seq))
        
        # Maskeler
        book_mask = [True if x == 0 else False for x in book_seq]
        movie_mask = [True if x == 0 else False for x in movie_seq]
        
        # NaN Yaması
        if all(book_mask): book_mask[0] = False
        if all(movie_mask): movie_mask[0] = False
        
        return {
            'user_id': torch.tensor(user_id, dtype=torch.long),
            'target_movie_id': torch.tensor(target_movie_id, dtype=torch.long),
            'book_seq': torch.tensor(book_seq, dtype=torch.long),
            'movie_seq': torch.tensor(movie_seq, dtype=torch.long),
            'book_mask': torch.tensor(book_mask, dtype=torch.bool),
            'movie_mask': torch.tensor(movie_mask, dtype=torch.bool)
        }