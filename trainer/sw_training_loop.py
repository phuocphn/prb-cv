# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from contextlib import contextmanager, suppress
from copy import copy, deepcopy
from typing import Optional

import numpy as np
import torch

from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.core.step_result import Result
from pytorch_lightning.plugins import ParallelPlugin
from pytorch_lightning.trainer.states import TrainerState
from pytorch_lightning.trainer.supporters import Accumulator, TensorRunningAccum
from pytorch_lightning.utilities import _TPU_AVAILABLE, AMPType, DeviceType, parsing
from pytorch_lightning.utilities.distributed import rank_zero_info
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.memory import recursive_detach
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.parsing import AttributeDict
from pytorch_lightning.utilities.warnings import WarningCache


from pytorch_lightning.trainer.training_loop import TrainLoop

class SwitchablePrecisionTrainLoop(TrainLoop):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def run_training_batch(self, batch, batch_idx, dataloader_idx):
        # track grad norms
        grad_norm_dic = {}

        # bookkeeping
        self.trainer.hiddens = None

        # track all outputs across time and num of optimizers
        batch_outputs = [[] for _ in range(len(self.get_optimizers_iterable()))]

        if batch is None:
            return AttributeDict(signal=0, grad_norm_dic=grad_norm_dic)

        # hook
        response = self.trainer.call_hook("on_batch_start")
        if response == -1:
            return AttributeDict(signal=-1, grad_norm_dic=grad_norm_dic)

        # hook
        response = self.trainer.call_hook("on_train_batch_start", batch, batch_idx, dataloader_idx)
        if response == -1:
            return AttributeDict(signal=-1, grad_norm_dic=grad_norm_dic)

        # lightning module hook
        splits = self.tbptt_split_batch(batch)

        for split_idx, split_batch in enumerate(splits):

            # create an iterable for optimizers and loop over them
            for opt_idx, optimizer in self.prepare_optimizers():

                # toggle model params + set info to logger_connector
                self.run_train_split_start(split_idx, split_batch, opt_idx, optimizer)

                if self.should_accumulate():
                    # For gradient accumulation

                    # -------------------
                    # calculate loss (train step + train step end)
                    # -------------------

                    # automatic_optimization=True: perform dpp sync only when performing optimizer_step
                    # automatic_optimization=False: don't block synchronization here
                    with self.block_ddp_sync_behaviour():
                        self.training_step_and_backward(
                            split_batch, batch_idx, opt_idx, optimizer, self.trainer.hiddens
                        )

                    batch_outputs = self._process_closure_result(
                        batch_outputs=batch_outputs,
                        opt_idx=opt_idx,
                        clear_curr_step_result=False
                    )

                # ------------------------------
                # BACKWARD PASS
                # ------------------------------
                # gradient update with accumulated gradients

                else:
                    if self.automatic_optimization:

                        def train_step_and_backward_closure():
                            result = self.training_step_and_backward(
                                split_batch, batch_idx, opt_idx, optimizer, self.trainer.hiddens
                            )
                            return None if result is None else result.loss

                        # optimizer step
                        self.optimizer_step(optimizer, opt_idx, batch_idx, train_step_and_backward_closure)

                    else:
                        self._curr_step_result = self.training_step(
                            split_batch, batch_idx, opt_idx, self.trainer.hiddens
                        )

                    if self._curr_step_result is None:
                        # user decided to skip optimization
                        # make sure to zero grad.
                        continue

                    batch_outputs = self._process_closure_result(
                        batch_outputs=batch_outputs,
                        opt_idx=opt_idx,
                        clear_curr_step_result=True
                    )

                    # todo: Properly aggregate grad_norm accros opt_idx and split_idx
                    grad_norm_dic = self._cur_grad_norm_dict
                    self._cur_grad_norm_dict = None

                    # update running loss + reset accumulated loss
                    self.update_running_loss()

        result = AttributeDict(
            signal=0,
            grad_norm_dic=grad_norm_dic,
            training_step_output_for_epoch_end=batch_outputs,
        )
        return result


    def _process_closure_result(self, batch_outputs: list, opt_idx: int, clear_curr_step_result: bool=False) -> list:
        opt_closure_result = self._curr_step_result

        if opt_closure_result is not None:

            # cache metrics
            self.trainer.logger_connector.cache_training_step_metrics(opt_closure_result)

            # track hiddens
            self.trainer.hiddens = self.process_hiddens(opt_closure_result)

            # check if loss or model weights are nan
            if self.trainer.terminate_on_nan:
                self.trainer.detect_nan_tensors(opt_closure_result.loss)

            # track all the outputs across all steps
            batch_opt_idx = opt_idx if len(batch_outputs) > 1 else 0
            batch_outputs[batch_opt_idx].append(opt_closure_result.training_step_output_for_epoch_end)

            if self.automatic_optimization:
                # track total loss for logging (avoid mem leaks)
                self.accumulated_loss.append(opt_closure_result.loss)

        if clear_curr_step_result:
            self._curr_step_result = None

        return batch_outputs

    def training_step_and_backward(self, split_batch, batch_idx, opt_idx, optimizer, hiddens):
        """
        wrap the forward step in a closure so second order methods work
        """
        with self.trainer.profiler.profile("training_step_and_backward"):
            # lightning module hook
            for bw in [8,6,5,4]:
                self.lightning_module.model.switch_precision(bit=bw)
                result = self.training_step(split_batch, batch_idx, opt_idx, hiddens)
                if self._curr_step_result == None:
                    self._curr_step_result = result
                else:
                    self._curr_step_result = result + self._curr_step_result


                if not self._skip_backward and self.automatic_optimization:
                    #is_first_batch_to_accumulate = batch_idx % self.trainer.accumulate_grad_batches == 0
                    #if is_first_batch_to_accumulate:
                    if bw == 8:
                        self.on_before_zero_grad(optimizer)
                        self.optimizer_zero_grad(batch_idx, optimizer, opt_idx)

                    # backward pass
                    if result is not None:
                        with self.trainer.profiler.profile("backward"):
                            self.backward(result, optimizer, opt_idx)

                        # hook - call this hook only
                        # when gradients have finished to accumulate
                        if not self.should_accumulate():
                            self.on_after_backward(result.training_step_output, batch_idx, result.loss)

                        # check if loss or model weights are nan
                        if self.trainer.terminate_on_nan:
                            self.trainer.detect_nan_tensors(result.loss)

                    else:
                        self.warning_cache.warn("training_step returned None if it was on purpose, ignore this warning...")

                    if len(self.trainer.optimizers) > 1:
                        # revert back to previous state
                        self.trainer.lightning_module.untoggle_optimizer(opt_idx)

        return result

 