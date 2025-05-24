import torch
import pandas as pd


class MetaPatternPool():
    def __init__(self, wave_class, wave_len, threshold, device):
        self.seasonal_pool = torch.empty(wave_class, wave_len, requires_grad=False).to(device)
        self.wave_len = wave_len
        self.threshold = threshold
        self.count = 0
        self.update_count = 0  

        self.s_init = False
        self.t_init = False

        self.step = 0  # 当前步数
        self.pool_update_step = torch.zeros(wave_class, dtype=torch.long)  # 上次更新步数

    def build_pool_seasonal(self, series):
        processed_indices = set()

        patch_series = series.reshape(-1, self.wave_len)
        product_matrix = patch_series.unsqueeze(1) * patch_series.unsqueeze(0)
        product_sum = product_matrix.sum(dim=2)
        product_matrix.view(product_matrix.size(0), -1)[:, ::product_matrix.size(-1) + 1] = 0
        new_seasonal_pool = self.seasonal_pool.clone()
        origin_seasonal_pool = self.seasonal_pool.clone()


        for i in range(product_sum.size(0)):
            if i in processed_indices:
                continue
            if self.count >= self.seasonal_pool.size(0):
                break

            max_value, max_index = product_sum[i].max(dim=0)

            # If the maximum value of a certain column is significantly different from
            # other series in the novel, save it as a new pattern
            if max_value.item() <= self.threshold:
                new_seasonal_pool[self.count] = patch_series[i]
                self.count += 1
            # If there is a large value, it indicates that there is a similar pattern,
            # and similar merging should be performed
            else:
                max_sequence_index = max_index.item()
                other_max_indices = (product_sum[i] > self.threshold).nonzero().squeeze(1)

                processed_indices.update(other_max_indices.tolist())

                if torch.all(other_max_indices == max_sequence_index):
                    weights = product_sum[i, other_max_indices] / product_sum[i, other_max_indices].sum()
                    mean_sequence = (patch_series[other_max_indices].T @ weights.unsqueeze(1)).squeeze(1)
                    new_seasonal_pool[self.count] = mean_sequence
                    self.count += 1

        self.seasonal_pool = new_seasonal_pool
        self.s_init = True
        mpp = self.seasonal_pool.clone()
        
        return mpp


    def update_pool(self, series, alpha=0.1):
        patch_series = series.reshape(-1, self.wave_len)
        product_matrix = patch_series.unsqueeze(1) * self.seasonal_pool.unsqueeze(0)
        product_sum = product_matrix.sum(dim=2)

        new_seasonal_pool = self.seasonal_pool.clone()
        original_seasonal_pool = self.seasonal_pool.clone()  

        max_values, max_indices = product_sum.max(dim=1)
        update_mask = max_values > self.threshold

        updated_indices = []  


        if update_mask.any():
            updated_indices.extend(max_indices[update_mask].tolist())
            new_seasonal_pool[max_indices[update_mask]] = (
                    alpha * patch_series[update_mask] + (1 - alpha) * new_seasonal_pool[max_indices[update_mask]]
            )

        no_update_indices = torch.where(~update_mask)[0]

        for idx in no_update_indices:
            if (new_seasonal_pool == 0).all(dim=1).any():
                zero_row_index = (new_seasonal_pool == 0).all(dim=1).nonzero(as_tuple=True)[0][0]
                new_seasonal_pool[zero_row_index] = patch_series[idx]
                updated_indices.append(zero_row_index.item())
            else:
                max_sim_index = product_sum[idx].argmax()
                new_seasonal_pool[max_sim_index] = (
                        alpha * patch_series[idx] + (1 - alpha) * new_seasonal_pool[max_sim_index]
                )
                updated_indices.append(max_sim_index.item())

        self.seasonal_pool = new_seasonal_pool
        
        return new_seasonal_pool.clone()




