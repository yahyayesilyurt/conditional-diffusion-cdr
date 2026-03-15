import torch
from torch.utils.data import Dataset
import json
import pandas as pd


def load_and_pad_embeddings(pt_file_path):
    """
    Loads GNN embeddings and prepends a zero PAD vector at index 0.

    Index conventions after padding:
      index 0    → zero vector (padding token)
      index 1..N → actual embeddings

    Downstream usage:
      user_ids        : 0-indexed in dataset → E2EWrapper applies +1
      book/movie seq  : already +1 in dataset → used directly
    """
    data       = torch.load(pt_file_path)
    embeddings = data['embeddings']

    book_embs  = embeddings['book']
    movie_embs = embeddings['movie']
    user_embs  = embeddings['user']
    embed_dim  = book_embs.shape[1]

    print(f"embed_dim: {embed_dim}")
    print(f"Users: {user_embs.shape[0]} | Books: {book_embs.shape[0]} | Movies: {movie_embs.shape[0]}")

    pad = torch.zeros(1, embed_dim)
    return (
        torch.cat([pad, user_embs],  dim=0),   # padded_user_embs
        torch.cat([pad, book_embs],  dim=0),   # padded_book_embs
        torch.cat([pad, movie_embs], dim=0),   # padded_movie_embs
    )


class CrossDomainDataset(Dataset):
    """
    Full sliding window dataset for cross-domain sequential recommendation.

    Source domain : Book  (user's reading history → context signal)
    Target domain : Movie (predict next movie interaction)

    Index conventions:
      user_id         : 0-indexed (raw mapping value)
      book_seq        : 1-indexed (0 = padding)
      movie_seq       : 1-indexed (0 = padding)
      target_movie_id : 1-indexed

    Book history is time-bounded: only books read BEFORE the movie
    interaction's timestamp are included, preventing data leakage.

    All-padding fallback is handled in E2EWrapper (not here).
    """

    def __init__(
        self,
        book_inter_path,
        movie_inter_path,
        book_mapping_path,
        movie_mapping_path,
        user_mapping_path,
        max_seq_len=10,
        mode='train',
        train_movie_inter_path=None
    ):
        assert mode in ('train', 'valid', 'test'), \
            f"mode must be 'train', 'valid', or 'test', got '{mode}'"
        if mode != 'train':
            assert train_movie_inter_path is not None, \
                "train_movie_inter_path required for valid/test mode"

        self.max_seq_len = max_seq_len
        self.mode        = mode

        # --- Load mappings ---
        with open(book_mapping_path,  'r') as f: self.book_mapping  = json.load(f)
        with open(movie_mapping_path, 'r') as f: self.movie_mapping = json.load(f)
        with open(user_mapping_path,  'r') as f: self.user_mapping  = json.load(f)

        # --- Book history with timestamps (leakage prevention) ---
        book_df = pd.read_csv(book_inter_path, sep='\t')
        self.user_book_history = self._build_history_with_ts(book_df, self.book_mapping)

        # --- Build samples ---
        self.samples = []

        if mode == 'train':
            self._build_train_samples(movie_inter_path)
        else:
            self._build_eval_samples(movie_inter_path, train_movie_inter_path)

    # ------------------------------------------------------------------
    # History builder
    # ------------------------------------------------------------------

    def _build_history_with_ts(self, df, item_mapping):
        """
        Returns {user_id: [(item_idx, timestamp), ...]} sorted by timestamp.
        item_idx is 1-indexed (0 = padding).
        """
        history   = {}
        df_sorted = df.sort_values(by=['user_id:token', 'timestamp:float'])

        for user_str, group in df_sorted.groupby('user_id:token'):
            user_str = str(user_str)
            if user_str not in self.user_mapping:
                continue
            items = [
                (item_mapping[str(tid)] + 1, float(ts))
                for tid, ts in zip(
                    group['item_id:token'].values,
                    group['timestamp:float'].values
                )
                if str(tid) in item_mapping
            ]
            history[self.user_mapping[user_str]] = items
        return history

    # ------------------------------------------------------------------
    # Sample builders
    # ------------------------------------------------------------------

    def _build_train_samples(self, movie_inter_path):
        movie_df = pd.read_csv(movie_inter_path, sep='\t')
        grouped  = (
            movie_df.sort_values('timestamp:float')
                    .groupby('user_id:token')
        )

        for user_str, group in grouped:
            user_str = str(user_str)
            if user_str not in self.user_mapping:
                continue
            user_id = self.user_mapping[user_str]

            rows = [
                (self.movie_mapping[str(tid)] + 1, float(ts))
                for tid, ts in zip(
                    group['item_id:token'].values,
                    group['timestamp:float'].values
                )
                if str(tid) in self.movie_mapping
            ]

            for i in range(1, len(rows)):
                target_id, cutoff_ts = rows[i]
                movie_history = [mid for mid, _ in rows[:i]]
                self.samples.append({
                    'user_id'        : user_id,
                    'movie_history'  : movie_history,
                    'target_movie_id': target_id,
                    'cutoff_ts'      : cutoff_ts,
                })

        print(f"[Train] {len(self.samples)} sliding-window samples generated.")

    def _build_eval_samples(self, movie_inter_path, train_inter_path):
        train_df  = pd.read_csv(train_inter_path, sep='\t')
        target_df = pd.read_csv(movie_inter_path,  sep='\t')

        # Train movie history with timestamps (for context)
        train_movie_history = {}
        for user_str, group in (
            train_df.sort_values('timestamp:float').groupby('user_id:token')
        ):
            user_str = str(user_str)
            if user_str not in self.user_mapping:
                continue
            uid   = self.user_mapping[user_str]
            items = [
                (self.movie_mapping[str(tid)] + 1, float(ts))
                for tid, ts in zip(
                    group['item_id:token'].values,
                    group['timestamp:float'].values
                )
                if str(tid) in self.movie_mapping
            ]
            train_movie_history[uid] = items

        for _, row in target_df.iterrows():
            u_str = str(row['user_id:token'])
            i_str = str(row['item_id:token'])
            if u_str not in self.user_mapping or i_str not in self.movie_mapping:
                continue

            user_id   = self.user_mapping[u_str]
            target_id = self.movie_mapping[i_str] + 1
            cutoff_ts = float(row['timestamp:float'])
            history   = train_movie_history.get(user_id, [])

            self.samples.append({
                'user_id'        : user_id,
                'movie_history'  : [mid for mid, _ in history],
                'target_movie_id': target_id,
                'cutoff_ts'      : cutoff_ts,
            })

        print(f"[{self.mode}] {len(self.samples)} eval samples generated.")

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample  = self.samples[idx]
        user_id = sample['user_id']
        cutoff  = sample['cutoff_ts']

        # Book seq: only books before cutoff timestamp (data leakage prevention)
        all_books = self.user_book_history.get(user_id, [])
        book_seq  = [
            bid for bid, ts in all_books if ts < cutoff
        ][-self.max_seq_len:]

        movie_seq = sample['movie_history'][-self.max_seq_len:]

        # Padding
        book_seq  = book_seq  + [0] * (self.max_seq_len - len(book_seq))
        movie_seq = movie_seq + [0] * (self.max_seq_len - len(movie_seq))

        # Masks (True = padding)
        # All-padding case is handled with fallback in E2EWrapper
        book_mask  = [x == 0 for x in book_seq]
        movie_mask = [x == 0 for x in movie_seq]

        return {
            'user_id'        : torch.tensor(user_id,                   dtype=torch.long),
            'target_movie_id': torch.tensor(sample['target_movie_id'], dtype=torch.long),
            'book_seq'       : torch.tensor(book_seq,                  dtype=torch.long),
            'movie_seq'      : torch.tensor(movie_seq,                 dtype=torch.long),
            'book_mask'      : torch.tensor(book_mask,                 dtype=torch.bool),
            'movie_mask'     : torch.tensor(movie_mask,                dtype=torch.bool),
        }