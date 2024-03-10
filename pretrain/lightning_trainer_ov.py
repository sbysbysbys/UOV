import os
import re
import torch
import numpy as np
from torch.nn import functional as F
import torch.optim as optim
import MinkowskiEngine as ME
import pytorch_lightning as pl
from pretrain.criterion import NCELoss
from pretrain.nuscenes_tools import CategoryProcessor
from pytorch_lightning.utilities import rank_zero_only


class LightningPretrain(pl.LightningModule):
    def __init__(self, model_points, config):
        super().__init__()
        self.model_points = model_points
        # self.model_images = model_images
        self._config = config
        self.losses = config["losses"]
        self.train_losses = []
        self.val_losses = []
        self.num_matches = config["num_matches"]
        self.batch_size = config["batch_size"]
        self.num_epochs = config["num_epochs"]
        self.superpixel_size = config["superpixel_size"]
        self.text_embedding_path = config["text_embedding_path"]
        text_embedding = torch.load(self.text_embedding_path)
        self.text_embedding = F.normalize(text_embedding, dim=-1)
        text_number = config["text_number"]
        if self.text_embedding.shape[0] != text_number:
            self.text_embedding = self.text_embedding[:-1]      
        self.text_simm = torch.mm(self.text_embedding, self.text_embedding.transpose(1, 0))
        category_processor = CategoryProcessor()
        self.text_same_class = category_processor.text_same_class()
        self.alpha_i = config["alpha_i"]
        self.alpha_t = config["alpha_t"]

        self.epoch = 0
        if config["resume_path"] is not None:
            self.epoch = int(
                re.search(r"(?<=epoch=)[0-9]+", config["resume_path"])[0]
            )
        self.criterion = NCELoss(temperature=config["NCE_temperature"])
        self.working_dir = os.path.join(config["working_dir"], config["datetime"])
        if os.environ.get("LOCAL_RANK", 0) == 0:
            os.makedirs(self.working_dir, exist_ok=True)

    def configure_optimizers(self):
        optimizer = optim.SGD(
            # list(self.model_points.parameters()) + list(self.model_images.parameters()),
            list(self.model_points.parameters()),
            lr=self._config["lr"],
            momentum=self._config["sgd_momentum"],
            dampening=self._config["sgd_dampening"],
            weight_decay=self._config["weight_decay"],
        )
        # optimizer = optim.AdamW(
        #     list(self.model_points.parameters()) + list(self.model_images.parameters()),
        #     lr=self._config["lr"],
        #     weight_decay=self._config["weight_decay"]
        # )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.num_epochs)
        return [optimizer], [scheduler]

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)

    def training_step(self, batch, batch_idx):
        sparse_input = ME.SparseTensor(batch["sinput_F"], batch["sinput_C"])
        output_points = self.model_points(sparse_input).F
        # self.model_images.eval()
        # self.model_images.decoder.train()
        # output_images = self.model_images(batch["input_I"])

        del batch["sinput_F"]
        del batch["sinput_C"]
        del sparse_input
        # each loss is applied independtly on each GPU
        losses = [
            # getattr(self, loss)(batch, output_points, output_images)
            getattr(self, loss)(batch, output_points)
            for loss in self.losses
        ]
        loss = torch.mean(torch.stack(losses))

        torch.cuda.empty_cache()
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size
        )
        self.train_losses.append(loss.detach().cpu())
        return loss

    def loss(self, batch, output_points, output_images):
        pairing_points = batch["pairing_points"]
        pairing_images = batch["pairing_images"]
        idx = np.random.choice(pairing_points.shape[0], self.num_matches, replace=False)
        k = output_points[pairing_points[idx]]
        m = tuple(pairing_images[idx].T.long())
        q = output_images.permute(0, 2, 3, 1)[m]
        return self.criterion(k, q)

    # def loss_superpixels_average(self, batch, output_points, output_images):
    def loss_superpixels_average(self, batch, output_points):
        # compute a superpoints to superpixels loss using superpixels
        torch.cuda.empty_cache()  # This method is extremely memory intensive
        superpixels = batch["superpixels"]
        pairing_images = batch["pairing_images"]
        pairing_points = batch["pairing_points"]
        images = batch["input_I"]
        # 计算出每个超像素在批次中的全局索引
        superpixels = (
            torch.arange(
                0,
                images.shape[0] * self.superpixel_size,
                self.superpixel_size,
                device=self.device,
            )[:, None, None] + superpixels
        )
        m = tuple(pairing_images.cpu().T.long())

        # 提取图像和点云的配对索引
        superpixels_I = superpixels.flatten()
        idx_P = torch.arange(pairing_points.shape[0], device=superpixels.device)
        total_pixels = superpixels_I.shape[0]
        idx_I = torch.arange(total_pixels, device=superpixels.device)

        with torch.no_grad():
            one_hot_P = torch.sparse_coo_tensor(
                torch.stack((
                    superpixels[m], idx_P
                ), dim=0),
                torch.ones(pairing_points.shape[0], device=superpixels.device),
                (superpixels.shape[0] * self.superpixel_size, pairing_points.shape[0])
            )

            one_hot_I = torch.sparse_coo_tensor(
                torch.stack((
                    superpixels_I, idx_I
                ), dim=0),
                torch.ones(total_pixels, device=superpixels.device),
                (superpixels.shape[0] * self.superpixel_size, total_pixels)
            )
        
        # 稀疏张量乘法
        # 将对应点分类计算平均特征，相当于平均池
        # 不过不想要idx = 0的，需要剔除
        k = one_hot_P @ output_points[pairing_points]
        k = k / (torch.sparse.sum(one_hot_P, 1).to_dense()[:, None] + 1e-6)
        # q = one_hot_I @ output_images.permute(0, 2, 3, 1).flatten(0, 2)
        # q = q / (torch.sparse.sum(one_hot_I, 1).to_dense()[:, None] + 1e-6)

        idx_F = torch.arange(k.shape[0], device=superpixels.device)
        mask = torch.where(torch.logical_and(k[:, 0] != 0, idx_F[:]%self.superpixel_size != 0))
        # mask = torch.where(k[:, 0] != 0)
        k = k[mask]
        # q = q[mask]
        # return self.criterion(k, q)

        features = batch["features"]
        features_ids = features[:, 0] * self.superpixel_size + features[:, 1] + 1
        image_features = features[:, 3:]
        feature_mask = torch.where(torch.isin(features_ids, mask[0]))        
        qi = image_features[feature_mask].float()
        # normalize for qi
        qi = F.normalize(qi, dim=-1)

        text_label = features[:, 2].long()
        xids = torch.arange(k.shape[0], device=superpixels.device).long()
        yids = text_label[feature_mask]

        target_or = torch.zeros(
            (xids.shape[0], self.text_embedding.shape[0]), 
            dtype=torch.float, 
            device=superpixels.device
        )
        # target_or = torch.zeros((xids.shape[0], self.text_embedding.shape[0]), dtype=torch.float, device=superpixels.device)
        target_or[xids, yids] = 1
        target_weight = 0.2
        target_power = 1
        cos_simmilarity = self.text_simm.to(superpixels.device)
        cos_simmilarity = cos_simmilarity * (self.text_same_class.to(superpixels.device))
        target = (cos_simmilarity[yids] ** target_power).to(superpixels.device)
        target = target_weight * target_or + (1-target_weight) * target
        # softmax
        # target = F.softmax(target, dim=-1)
        # or 
        row_sums = target.sum(dim=1, keepdim=True)
        target = target/row_sums

        # text_features = self.text_embedding[text_label].to(superpixels.device)
        # qt = text_features[feature_mask]

        # import pdb; pdb.set_trace()
        
        pi_infonce_loss = self.criterion(k, qi) * self.alpha_i 
        pt_infonce_loss = self.criterion(k, self.text_embedding.to(superpixels.device), target = target) * self.alpha_t
        
        return pi_infonce_loss + pt_infonce_loss
        
        

    def training_epoch_end(self, outputs):
        self.epoch += 1
        # if self.epoch == self.num_epochs-1:
        #     self.save()
        # if self.epoch == self.num_epochs:
        #     self.save()
        self.save()
        return super().training_epoch_end(outputs)

    def validation_step(self, batch, batch_idx):
        sparse_input = ME.SparseTensor(batch["sinput_F"], batch["sinput_C"])
        output_points = self.model_points(sparse_input).F
        # self.model_images.eval()
        # output_images = self.model_images(batch["input_I"])

        losses = [
            getattr(self, loss)(batch, output_points)
            for loss in self.losses
        ]
        loss = torch.mean(torch.stack(losses))
        self.val_losses.append(loss.detach().cpu())

        self.log(
            "val_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=self.batch_size
        )
        return loss

    @rank_zero_only
    def save(self):
        path = os.path.join(self.working_dir, str(self.epoch) + "model.pt")
        print(path)
        torch.save(
            {
                "model_points": self.model_points.state_dict(),
                # "model_images": self.model_images.state_dict(),
                "epoch": self.epoch,
                "config": self._config,
            },
            path,
        )
