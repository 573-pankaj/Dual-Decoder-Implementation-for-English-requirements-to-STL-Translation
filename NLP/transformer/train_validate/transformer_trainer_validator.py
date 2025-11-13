from transformer import transformer_hyperparas
from public import paths
import random
import numpy as np
import torch
from d2l import torch as d2l
import pickle
import torch.serialization

# 👇 import the model class you trust (your implementation)
from transformer.model.transformer_encoder import TransformerEncoder


class TransformerTrainerValidator:
    def __init__(self, seed):
        self.read_file(seed)

        # Adam with transformer schedule (lr set each step by lrate_compute)
        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=0.0, betas=(0.9, 0.98), eps=1e-9
        )

        # If resuming, restore optimizer + RNG state
        if self.step > 1:
            self.optimizer.load_state_dict(self.checkpoint_dict['optimizer_state_dict'])
            random.setstate(self.checkpoint_dict['python_random_state'])
            np.random.set_state(self.checkpoint_dict['numpy_random_state'])
            torch.set_rng_state(self.checkpoint_dict['torch_random_state'])

        # Loss: masked CE (expects [B,T,V], [B,T], valid_lens)
        self.loss = d2l.MaskedSoftmaxCELoss()

        self.net.train()
        self.train(seed)

    def read_file(self, seed):
        # Make sure pickle can load your model classes
        torch.serialization.add_safe_globals([TransformerEncoder])

        # Load checkpoint (PyTorch 2.6+ safe)
        self.checkpoint_dict = torch.load(
            paths.transformer_record_path + str(seed) + '/checkpoint_dict',
            weights_only=False
        )

        # Read model state
        self.device = self.checkpoint_dict['device']
        self.net = self.checkpoint_dict['net_model']
        self.net.load_state_dict(self.checkpoint_dict['net_state_dict'])
        self.net.to(self.device)

        self.max_epochs = self.checkpoint_dict['max_epochs']
        self.epochs_finished = self.checkpoint_dict['epochs_finished']
        self.step = self.checkpoint_dict['step']

        # Load iterators (pickled)
        with open(paths.transformer_record_path + str(seed) + '/data_iter_dict', 'rb') as f:
            data_iter_dict = pickle.load(f)
        self.train_data_iter = data_iter_dict['train']
        self.dev_data_iter = data_iter_dict['dev']

        # Load tokenizer / vocab
        with open(paths.preprocess_info_dict_path, 'rb') as f:
            preprocess_info_dict = pickle.load(f)
        self.tgt_vocab = preprocess_info_dict['stl_tokenizer']

    def lrate_compute(self):
        # Transformer schedule
        return (transformer_hyperparas.factor
                * transformer_hyperparas.num_hiddens ** (-0.5)
                * min(self.step ** (-0.5),
                      self.step * transformer_hyperparas.warmup_steps ** (-1.5)))

    def train(self, seed):
        while self.epochs_finished < self.max_epochs:
            timer = d2l.Timer()
            # Accumulator: sum_loss_over_tokens, sum_acc_over_samples, sum_batch_size, sum_valid_tokens
            metric = d2l.Accumulator(4)
            batch_index = 1

            for batch in self.train_data_iter:
                self.optimizer.zero_grad()

                X, X_valid_len, Y, Y_valid_len = [x.to(self.device) for x in batch]

                # Teacher forcing decoder input (<bos> + shifted Y)
                bos = torch.tensor(self.tgt_vocab.encode('<bos>').ids * Y.shape[0],
                                   device=self.device).reshape(-1, 1)
                dec_input = torch.cat([bos, Y[:, :-1]], 1)

                # ---------------- Dual-decoder forward ----------------
                # Expect decoder forward to return: fused logits, syn list, sem list, state
                Y_hat_fused, syn_outputs, sem_outputs, _ = self.net(
                    X, dec_input, X_valid_len, Y_valid_len
                )

                # ---------- Losses ----------
                # fused loss (always on)
                loss_fused = self.loss(Y_hat_fused, Y, Y_valid_len)

                # syntax auxiliary loss (sum over layers)
                loss_syn = 0
                for syn in syn_outputs:
                    syn_logits = self.net.decoder.dense(syn)
                    loss_syn += self.loss(syn_logits, Y, Y_valid_len)

                # semantic auxiliary loss (sum over layers)
                loss_sem = 0
                for sem in sem_outputs:
                    sem_logits = self.net.decoder.dense(sem)
                    loss_sem += self.loss(sem_logits, Y, Y_valid_len)

                # weights for aux losses
                alpha = 0.6  # semantic
                beta = 0.4   # syntax

                # total loss (average per sample)
                l = loss_fused + alpha * loss_sem + beta * loss_syn
                l = l.sum() / Y.shape[0]

                # Backprop once
                l.backward()
                # Optional: gradient clipping
                # d2l.grad_clipping(self.net, 1.0)

                # LR schedule step
                lrate = self.lrate_compute()
                for p in self.optimizer.param_groups:
                    p['lr'] = lrate
                self.optimizer.step()

                # -------------- Metrics --------------
                with torch.no_grad():
                    num_tokens = Y_valid_len.sum()
                    acc_sum = self.accuracy_compute(Y_hat_fused, Y)

                    # For epoch averages: sum total loss over samples in batch
                    metric.add(l.item() * Y.shape[0], acc_sum, Y.shape[0], num_tokens)

                    batch_log = (
                        f"Train, Epoch: {self.epochs_finished+1}, Batch: {batch_index}, "
                        f"Step num: {self.step}, Learning rate: {lrate:.8f}, "
                        f"Avg batch loss: {l.item():.4f}, Avg batch acc: {acc_sum/Y.shape[0]:.4f}"
                    )
                    print(batch_log)
                    self.log_write(batch_log, seed)

                    batch_index += 1
                    self.step += 1

            # ---------- End of epoch: compute averages ----------
            time_taken = timer.stop()
            train_loss = metric[0] / metric[2]         # average loss per sample
            train_acc = metric[1] / metric[2]          # average acc per sample
            train_speed = metric[3] / time_taken       # tokens / sec

            epoch_log = (f"Train, Epoch: {self.epochs_finished+1}, "
                         f"Avg epoch loss: {train_loss:.4f}, Avg epoch acc: {train_acc:.4f}, "
                         f"Overall time: {time_taken:.1f} s, Speed: {train_speed:.1f} tokens/s")
            epoch_log += f" on {self.device}\n"
            print(epoch_log)
            self.log_write(epoch_log, seed)

            # Validate
            validate_loss, validate_acc = self.validate(seed)

            # Bump epoch count and save
            self.epochs_finished += 1
            self.checkpoint_write_read(train_loss, train_acc, validate_loss, validate_acc, seed)

    def validate(self, seed):
        with torch.no_grad():
            timer = d2l.Timer()
            metric = d2l.Accumulator(4)
            batch_index = 1

            for batch in self.dev_data_iter:
                X, X_valid_len, Y, Y_valid_len = [x.to(self.device) for x in batch]

                bos = torch.tensor(self.tgt_vocab.encode('<bos>').ids * Y.shape[0],
                                   device=self.device).reshape(-1, 1)
                dec_input = torch.cat([bos, Y[:, :-1]], 1)

                # Validation: use fused output only
                Y_hat_fused, _, _, _ = self.net(X, dec_input, X_valid_len, Y_valid_len)
                l = self.loss(Y_hat_fused, Y, Y_valid_len)
                l = l.sum() / Y.shape[0]

                num_tokens = Y_valid_len.sum()
                acc_sum = self.accuracy_compute(Y_hat_fused, Y)

                metric.add(l.item() * Y.shape[0], acc_sum, Y.shape[0], num_tokens)

                batch_log = (f"Validate, Epoch: {self.epochs_finished+1}, Batch: {batch_index}, "
                             f"Avg batch loss: {l.item():.4f}, Avg batch acc: {acc_sum/Y.shape[0]:.4f}")
                print(batch_log)
                self.log_write(batch_log, seed)

                batch_index += 1

            time_taken = timer.stop()
            validate_loss = metric[0] / metric[2]
            validate_acc = metric[1] / metric[2]
            validate_speed = metric[3] / time_taken

            epoch_log = (f"Validate, Epoch: {self.epochs_finished+1}, "
                         f"Avg epoch loss: {validate_loss:.4f}, Avg epoch acc: {validate_acc:.4f}, "
                         f"Overall time: {time_taken:.1f} s, Speed: {validate_speed:.1f} tokens/s")
            epoch_log += f" on {self.device}\n"
            print(epoch_log)
            self.log_write(epoch_log, seed)

        return validate_loss, validate_acc

    def accuracy_compute(self, Y_hat, Y):
        """
        Y_hat: [B, T, V] logits
        Y:     [B, T] target token ids
        Returns sum of per-sample accuracies for the batch.
        """
        acc_sum = 0.0
        for i in range(Y.shape[0]):
            prediction = Y_hat[i]     # (T, V)
            reference_index = Y[i]    # (T,)
            prediction_index = prediction.argmax(dim=1)

            prediction_list = prediction_index.detach().cpu().numpy().tolist()
            reference_list = reference_index.detach().cpu().numpy().tolist()

            eos_id = self.tgt_vocab.encode('<eos>').ids[0]
            # clip at first EOS in reference
            if eos_id in reference_list:
                id_last = reference_list.index(eos_id)
                del reference_list[id_last:]
                del prediction_list[id_last:]

            # sanity check
            if len(reference_list) != len(prediction_list):
                print('Warning: reference/prediction length mismatch after EOS trim.')

            prediction_tensor_new = torch.tensor(prediction_list)
            reference_tensor_new = torch.tensor(reference_list)
            correct_list, correct_num = self.correct_num_compute(prediction_tensor_new, reference_tensor_new)
            total_number = max(1, prediction_tensor_new.shape[0])  # guard
            sample_acc = correct_num / total_number
            acc_sum += sample_acc

            # verbose print for last sample in batch
            if i == Y.shape[0] - 1:
                print()
                print('Last sample in batch:')
                print('prediction id:')
                print(prediction_list)
                print('reference id:')
                print(reference_list)
                print('correctness:')
                print(correct_list)

                prediction_str = self.tgt_vocab.decode(prediction_list, skip_special_tokens=False)
                reference_str = self.tgt_vocab.decode(reference_list, skip_special_tokens=False)
                print('prediction:', prediction_str)
                print('reference: ', reference_str)
                print('acc:', sample_acc)

        return acc_sum

    @staticmethod
    def correct_num_compute(y_hat, y):
        cmp = y_hat.type(y.dtype) == y
        return cmp.type(y.dtype).cpu().numpy().tolist(), float(cmp.type(y.dtype).sum())

    @staticmethod
    def log_write(log, seed):
        with open(paths.transformer_record_path + str(seed) + '/log.txt', 'a') as f:
            f.write(log + '\n')

    def checkpoint_write_read(self, train_loss, train_acc, validate_loss, validate_acc, seed):
        # Save checkpoint dict
        self.checkpoint_dict['net_state_dict'] = self.net.state_dict()
        self.checkpoint_dict['optimizer_state_dict'] = self.optimizer.state_dict()
        self.checkpoint_dict['epochs_finished'] = self.epochs_finished
        self.checkpoint_dict['step'] = self.step  # next step index
        self.checkpoint_dict['train_loss_list'].append(train_loss)
        self.checkpoint_dict['train_acc_list'].append(train_acc)
        self.checkpoint_dict['validate_loss_list'].append(validate_loss)
        self.checkpoint_dict['validate_acc_list'].append(validate_acc)

        # Save RNG states
        self.checkpoint_dict['python_random_state'] = random.getstate()
        self.checkpoint_dict['numpy_random_state'] = np.random.get_state()
        self.checkpoint_dict['torch_random_state'] = torch.get_rng_state()

        base = paths.transformer_record_path + str(seed)
        torch.save(self.checkpoint_dict, base + '/checkpoint_dict')
        torch.save(self.checkpoint_dict['net_state_dict'], base + '/net_state_dict')

        info_dict = {
            'step': self.checkpoint_dict['step'] - 1,  # current step index
            'train_loss_list': self.checkpoint_dict['train_loss_list'],
            'train_acc_list': self.checkpoint_dict['train_acc_list'],
            'validate_loss_list': self.checkpoint_dict['validate_loss_list'],
            'validate_acc_list': self.checkpoint_dict['validate_acc_list'],
        }
        torch.save(info_dict, base + '/info_dict')

        # Reload checkpoint to confirm integrity (optional)
        self.checkpoint_dict = torch.load(base + '/checkpoint_dict', weights_only=False)
        self.net.load_state_dict(self.checkpoint_dict['net_state_dict'])
        self.optimizer.load_state_dict(self.checkpoint_dict['optimizer_state_dict'])

        # Restore RNG
        random.setstate(self.checkpoint_dict['python_random_state'])
        np.random.set_state(self.checkpoint_dict['numpy_random_state'])
        torch.set_rng_state(self.checkpoint_dict['torch_random_state'])
