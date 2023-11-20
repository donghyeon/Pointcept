import os
import shutil
from collections import OrderedDict

import numpy as np
import torch
import torch.distributed as dist
import pointops

import pointcept.utils.comm as comm
from pointcept.utils.comm import is_main_process
from pointcept.utils.misc import intersection_and_union_gpu

from .default import HookBase
from .builder import HOOKS


@HOOKS.register_module()
class SemSegEvaluatorPerSteps(HookBase):
    def __init__(self, eval_steps) -> None:
        self.eval_steps = eval_steps
    
    def after_step(self):
        if self.trainer.cfg.evaluate:
            if self.trainer.comm_info["step"] % self.eval_steps == 0:
                self.eval()
                self.trainer.model.train()  # Change model to training mode after evaluation

    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()
        for i, input_dict in enumerate(self.trainer.val_loader):
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                output_dict = self.trainer.model(input_dict)
            output = output_dict["seg_logits"]
            loss = output_dict["loss"]
            pred = output.max(1)[1]
            segment = input_dict["segment"]
            if "origin_coord" in input_dict.keys():
                idx, _ = pointops.knn_query(
                    1,
                    input_dict["coord"].float(),
                    input_dict["offset"].int(),
                    input_dict["origin_coord"].float(),
                    input_dict["origin_offset"].int(),
                )
                pred = pred[idx.flatten().long()]
                segment = input_dict["origin_segment"]
            intersection, union, target = intersection_and_union_gpu(
                pred,
                segment,
                self.trainer.cfg.data.num_classes,
                self.trainer.cfg.data.ignore_index,
            )
            if comm.get_world_size() > 1:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(
                    target
                )
            intersection, union, target = (
                intersection.cpu().numpy(),
                union.cpu().numpy(),
                target.cpu().numpy(),
            )
            # Here there is no need to sync since sync happened in dist.all_reduce
            self.trainer.storage.put_scalar("val_intersection", intersection)
            self.trainer.storage.put_scalar("val_union", union)
            self.trainer.storage.put_scalar("val_target", target)
            self.trainer.storage.put_scalar("val_loss", loss.item())
            info = "Test: [{iter}/{max_iter}] ".format(
                iter=i + 1, max_iter=len(self.trainer.val_loader)
            )
            if "origin_coord" in input_dict.keys():
                info = "Interp. " + info
            self.trainer.logger.info(
                info
                + "Loss {loss:.4f} ".format(
                    iter=i + 1, max_iter=len(self.trainer.val_loader), loss=loss.item()
                )
            )
            
            if self.trainer.cfg.empty_cache:
                torch.cuda.empty_cache()

        loss_avg = self.trainer.storage.history("val_loss").avg
        intersection = self.trainer.storage.history("val_intersection").total
        union = self.trainer.storage.history("val_union").total
        target = self.trainer.storage.history("val_target").total
        iou_class = intersection / (union + 1e-10)
        acc_class = intersection / (target + 1e-10)
        m_iou = np.mean(iou_class)
        m_acc = np.mean(acc_class)
        all_acc = sum(intersection) / (sum(target) + 1e-10)
        self.trainer.logger.info(
            "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
                m_iou, m_acc, all_acc
            )
        )
        for i in range(self.trainer.cfg.data.num_classes):
            self.trainer.logger.info(
                "Class_{idx}-{name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                    idx=i,
                    name=self.trainer.cfg.data.names[i],
                    iou=iou_class[i],
                    accuracy=acc_class[i],
                )
            )

        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("val/loss", loss_avg, self.trainer.comm_info["step"])
            self.trainer.writer.add_scalar("val/mIoU", m_iou, self.trainer.comm_info["step"])
            self.trainer.writer.add_scalar("val/mAcc", m_acc, self.trainer.comm_info["step"])
            self.trainer.writer.add_scalar("val/allAcc", all_acc, self.trainer.comm_info["step"])
        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        self.trainer.comm_info["current_metric_value"] = m_iou  # save for saver
        self.trainer.comm_info["current_metric_name"] = "mIoU"  # save for saver

    def after_train(self):
        self.trainer.logger.info(
            "Best {}: {:.4f}".format("mIoU", self.trainer.best_metric_value)
        )


@HOOKS.register_module()
class CheckpointSaverPerSteps(HookBase):
    def __init__(self, save_steps):
        self.save_steps = save_steps

    def after_step(self):
        if self.trainer.comm_info["step"] % self.save_steps == 0:
            if is_main_process():
                is_best = False
                if self.trainer.cfg.evaluate:
                    current_metric_value = self.trainer.comm_info["current_metric_value"]
                    current_metric_name = self.trainer.comm_info["current_metric_name"]
                    if current_metric_value > self.trainer.best_metric_value:
                        self.trainer.best_metric_value = current_metric_value
                        is_best = True
                        self.trainer.logger.info(
                            "Best validation {} updated to: {:.4f}".format(
                                current_metric_name, current_metric_value
                            )
                        )
                    self.trainer.logger.info(
                        "Currently Best {}: {:.4f}".format(
                            current_metric_name, self.trainer.best_metric_value
                        )
                    )

                filename = os.path.join(
                    self.trainer.cfg.save_path, "model", "model_last.pth"
                )
                self.trainer.logger.info("Saving checkpoint to: " + filename)
                torch.save(
                    {
                        "step": self.trainer.comm_info["step"],
                        "epoch": (
                            self.trainer.epoch + 1
                            if self.trainer.comm_info["step"] % len(self.trainer.train_loader) == 0
                            else self.trainer.epoch
                        ),
                        "state_dict": self.trainer.model.state_dict(),
                        "optimizer": self.trainer.optimizer.state_dict(),
                        "scheduler": self.trainer.scheduler.state_dict(),
                        "scaler": self.trainer.scaler.state_dict()
                        if self.trainer.cfg.enable_amp
                        else None,
                        "best_metric_value": self.trainer.best_metric_value,
                    },
                    filename + ".tmp",
                )
                os.replace(filename + ".tmp", filename)
                if is_best:
                    shutil.copyfile(
                        filename,
                        os.path.join(self.trainer.cfg.save_path, "model", "model_best.pth"),
                    )
                
                shutil.copyfile(
                    filename,
                    os.path.join(
                        self.trainer.cfg.save_path,
                        "model",
                        f"step_{self.trainer.comm_info['step']}.pth",
                    ),
                )


@HOOKS.register_module()
class CheckpointLoaderWithStep(HookBase):
    def __init__(self, keywords="", replacement=None, strict=False):
        self.keywords = keywords
        self.replacement = replacement if replacement is not None else keywords
        self.strict = strict
        self.start_step = 0

    def before_train(self):
        self.trainer.logger.info("=> Loading checkpoint & weight ...")
        if self.trainer.cfg.weight and os.path.isfile(self.trainer.cfg.weight):
            self.trainer.logger.info(f"Loading weight at: {self.trainer.cfg.weight}")
            checkpoint = torch.load(
                self.trainer.cfg.weight,
                map_location=lambda storage, loc: storage.cuda(),
            )
            self.trainer.logger.info(
                f"Loading layer weights with keyword: {self.keywords}, "
                f"replace keyword with: {self.replacement}"
            )
            weight = OrderedDict(
                [
                    (key.replace(self.keywords, self.replacement), value)
                    for key, value in checkpoint["state_dict"].items()
                    if self.keywords in key
                ]
            )
            load_state_info = self.trainer.model.load_state_dict(
                weight, strict=self.strict
            )
            self.trainer.logger.info(f"Missing keys: {load_state_info[0]}")
            if self.trainer.cfg.resume:
                self.trainer.logger.info(
                    f"Resuming train at step: {checkpoint['step']}"
                )
                self.start_step = checkpoint["step"]
                self.trainer.start_epoch = checkpoint["epoch"]
                self.trainer.best_metric_value = checkpoint["best_metric_value"]
                self.trainer.optimizer.load_state_dict(checkpoint["optimizer"])
                self.trainer.scheduler.load_state_dict(checkpoint["scheduler"])
                if self.trainer.cfg.enable_amp:
                    self.trainer.scaler.load_state_dict(checkpoint["scaler"])
        else:
            self.trainer.logger.info(f"No weight found at: {self.trainer.cfg.weight}")
        
        self.trainer.comm_info["step"] = self.trainer.start_epoch * len(self.trainer.train_loader)
    
    def before_epoch(self):
        # Skip samples to set self.trainer.comm_info["step"] to start_step
        if self.trainer.comm_info["step"] < self.start_step:
            it = iter(self.trainer.data_iterator)
            self.trainer.data_iterator = it

            num_skip_steps = self.start_step - self.trainer.comm_info["step"]
            for i in range(num_skip_steps):
                self.trainer.comm_info["step"] += 1

                (self.trainer.comm_info["iter"],
                 self.trainer.comm_info["input_dict"]) = next(it)
                
                info = "Skip: [{epoch}/{max_epoch}][{iter}/{max_iter}] ".format(
                    epoch=self.trainer.epoch + 1,
                    max_epoch=self.trainer.max_epoch,
                    iter=self.trainer.comm_info["iter"] + 1,
                    max_iter=len(self.trainer.train_loader),
                )
                self.trainer.comm_info["iter_info"] += info
                self.trainer.logger.info(self.trainer.comm_info["iter_info"])
                self.trainer.comm_info["iter_info"] = ""  # reset iter info


@HOOKS.register_module()
class InformationWriterWithStep(HookBase):
    def __init__(self):
        self.model_output_keys = []

    def before_train(self):
        self.trainer.comm_info["iter_info"] = ""
        self.trainer.comm_info["step"] = self.trainer.start_epoch * len(self.trainer.train_loader)

    def before_step(self):
        self.trainer.comm_info["step"] += 1
        # MSC pretrain do not have offset information. Comment the code for support MSC
        # info = "Train: [{epoch}/{max_epoch}][{iter}/{max_iter}] " \
        #        "Scan {batch_size} ({points_num}) ".format(
        #     epoch=self.trainer.epoch + 1, max_epoch=self.trainer.max_epoch,
        #     iter=self.trainer.comm_info["iter"], max_iter=len(self.trainer.train_loader),
        #     batch_size=len(self.trainer.comm_info["input_dict"]["offset"]),
        #     points_num=self.trainer.comm_info["input_dict"]["offset"][-1]
        # )
        info = "Train: [{epoch}/{max_epoch}][{iter}/{max_iter}] ".format(
            epoch=self.trainer.epoch + 1,
            max_epoch=self.trainer.max_epoch,
            iter=self.trainer.comm_info["iter"] + 1,
            max_iter=len(self.trainer.train_loader),
        )
        self.trainer.comm_info["iter_info"] += info

    def after_step(self):
        if "model_output_dict" in self.trainer.comm_info.keys():
            model_output_dict = self.trainer.comm_info["model_output_dict"]
            self.model_output_keys = model_output_dict.keys()
            for key in self.model_output_keys:
                self.trainer.storage.put_scalar(key, model_output_dict[key].item())

        for key in self.model_output_keys:
            self.trainer.comm_info["iter_info"] += "{key}: {value:.4f} ".format(
                key=key, value=self.trainer.storage.history(key).val
            )
        lr = self.trainer.optimizer.state_dict()["param_groups"][0]["lr"]
        self.trainer.comm_info["iter_info"] += "Lr: {lr:.5f}".format(lr=lr)
        self.trainer.logger.info(self.trainer.comm_info["iter_info"])
        self.trainer.comm_info["iter_info"] = ""  # reset iter info
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("lr", lr, self.trainer.comm_info["step"])
            for key in self.model_output_keys:
                self.trainer.writer.add_scalar(
                    "train_batch/" + key,
                    self.trainer.storage.history(key).val,
                    self.trainer.comm_info["step"],
                )

    def after_epoch(self):
        epoch_info = "Train result: "
        for key in self.model_output_keys:
            epoch_info += "{key}: {value:.4f} ".format(
                key=key, value=self.trainer.storage.history(key).avg
            )
        self.trainer.logger.info(epoch_info)
        if self.trainer.writer is not None:
            for key in self.model_output_keys:
                self.trainer.writer.add_scalar(
                    "train/" + key,
                    self.trainer.storage.history(key).avg,
                    self.trainer.epoch + 1,
                )
