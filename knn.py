import numpy as np
import faiss
import torch
import scipy

from utils import to_device
from tqdm import tqdm

class KNNHelper:
    def __init__(self, model, predictor, k):
        last_embs, gt_labels = self._get_knn_embedding_score(model)
        self.last_embs = last_embs
        self.gt_labels = gt_labels
        self.k = k

        # self.faiss_index = faiss.IndexFlatL2(last_embs.shape[1])
        self.faiss_index = faiss.IndexFlatIP(last_embs.shape[1])
        self.faiss_index.add(last_embs.astype(np.float32))
    
    def _get_knn_embedding_score(self, model):
        print("generating embedding for knn...")
        model.eval()
        last_embs, labels = [], []
        for collate_batch in tqdm(model.static_train_loader):
            collate_batch = to_device(collate_batch, model.device)
            with torch.no_grad():
                output = model(collate_batch)
            numpy_last_emb = output["last_emb"].cpu().numpy()
            last_embs.append(numpy_last_emb)
            labels.append(collate_batch["y"].cpu().numpy())
        last_embs = np.vstack(last_embs)
        labels = np.hstack(labels)
        return last_embs, labels
    
    def predict(self, pred_emb):
        dists, neighbors = self.faiss_index.search(pred_emb, k=self.k)
        neighbor_labels = self.gt_labels[neighbors]
        preds, _ = scipy.stats.mode(neighbor_labels, axis=1)
        return preds.flatten()
