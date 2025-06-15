import torch
import torch.nn as nn

class CRF(nn.Module):
    """
    Реализация Conditional Random Field (CRF) (были проблемы с импортом)
    """
    def __init__(self, num_tags: int, batch_first: bool = True):
        if num_tags <= 0:
            raise ValueError(f"invalid number of tags: {num_tags}")
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)

    def forward(self, emissions, tags=None, mask=None, reduction: str = 'mean'):
        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            if tags is not None:
                tags = tags.transpose(0, 1)
            if mask is not None:
                mask = mask.transpose(0, 1)
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)
        log_likelihood = self._compute_log_likelihood(emissions, tags, mask)
        if reduction == 'sum':
            return -log_likelihood.sum()
        if reduction == 'mean':
            return -log_likelihood.mean()
        return -log_likelihood

    def decode(self, emissions, mask=None):
        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            if mask is not None:
                mask = mask.transpose(0, 1)
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.uint8, device=emissions.device)
        return self._viterbi_decode(emissions, mask)

    def _compute_log_likelihood(self, emissions, tags, mask):
        seq_length, batch_size, _ = emissions.shape
        log_alpha = self._forward_pass(emissions, mask)
        gold_score = self._score_sequence(emissions, tags, mask)    
        return gold_score - log_alpha

    def _forward_pass(self, emissions, mask):
        seq_length, batch_size, _ = emissions.shape
        log_alpha = self.start_transitions + emissions[0]
        for i in range(1, seq_length):
            emit_scores = emissions[i].unsqueeze(1)
            trans_scores = self.transitions.unsqueeze(0)
            alpha_t = log_alpha.unsqueeze(2)
            scores = trans_scores + alpha_t + emit_scores
            log_alpha_next = torch.logsumexp(scores, dim=1)
            mask_t = mask[i].unsqueeze(1).float()
            log_alpha = mask_t * log_alpha_next + (1 - mask_t) * log_alpha
        log_alpha += self.end_transitions
        return torch.logsumexp(log_alpha, dim=1)


    def _score_sequence(self, emissions, tags, mask):
        batch_size = emissions.size(1)
        score = self.start_transitions[tags[0]]
        score += (emissions.gather(2, tags.unsqueeze(2)).squeeze(2) * mask.float()).sum(0)
        for i in range(emissions.size(0) - 1):
            score += self.transitions[tags[i], tags[i+1]] * mask[i+1].float()
        seq_ends = mask.long().sum(dim=0) - 1
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        score += self.end_transitions[last_tags]
        return score
    
    def _viterbi_decode(self, emissions, mask):
        seq_length, batch_size, _ = emissions.shape
        log_delta = self.start_transitions + emissions[0]
        backpointers = []
        for i in range(1, seq_length):
            delta_t = log_delta.unsqueeze(2)
            trans_scores = self.transitions.unsqueeze(0)
            scores = delta_t + trans_scores
            log_delta_next, backpointers_t = torch.max(scores, dim=1)
            log_delta_next += emissions[i]
            mask_t = mask[i].unsqueeze(1).float()
            log_delta = mask_t * log_delta_next + (1 - mask_t) * log_delta
            backpointers.append(backpointers_t)

        log_delta += self.end_transitions
        best_last_tag = torch.argmax(log_delta, dim=1)
        best_path = [best_last_tag]
        for backpointers_t in reversed(backpointers):
            best_last_tag = backpointers_t.gather(1, best_last_tag.unsqueeze(1)).squeeze(1)
            best_path.insert(0, best_last_tag)    
        return torch.stack(best_path).transpose(0, 1).tolist()


class IntroDetectionTransformer(nn.Module):
    def __init__(self, d_model=768, n_heads=8, n_layers=4, num_labels=2, class_weights=None):
        super().__init__()
        self.pos_encoding = nn.Parameter(torch.zeros(1, 60, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Linear(d_model, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
    
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None

    def forward(self, embeddings, labels=None, mask=None):
        x = embeddings + self.pos_encoding
        x = self.transformer(x) 
        logits = self.classifier(x)

        if labels is not None:
            if self.class_weights is not None:
                logits[:, :, 1] = logits[:, :, 1] * self.class_weights[1]
            
            return self.crf(logits, labels.long(), mask=mask)
        else:
            return self.crf.decode(logits, mask=mask)