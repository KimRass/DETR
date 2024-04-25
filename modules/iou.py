import torch


class GIoULoss(object):
    @staticmethod
    def get_area(ltrb):
        """
        args:
            `ltrb`: Tensor of shape (N, 4)
        returns:
            Tensor of shape (N)
        """
        return torch.clip(
            ltrb[..., 2] - ltrb[..., 0], min=0
        ) * torch.clip(ltrb[..., 3] - ltrb[..., 1], min=0)

    @staticmethod
    def get_intersection_area(ltrb1, ltrb2):
        """
        args:
            `ltrb1`: Tensor of shape (N, 4)
            `ltrb2`: Tensor of shape (M, 4)
        returns:
            Tensor of shape (N, M)
        """
        l = torch.maximum(ltrb1[..., 0][..., None], ltrb2[..., 0][None, ...])
        t = torch.maximum(ltrb1[..., 1][..., None], ltrb2[..., 1][None, ...])
        r = torch.minimum(ltrb1[..., 2][..., None], ltrb2[..., 2][None, ...])
        b = torch.minimum(ltrb1[..., 3][..., None], ltrb2[..., 3][None, ...])
        return torch.clip(r - l, min=0) * torch.clip(b - t, min=0)

    @staticmethod
    def get_smallest_enclosing_area(ltrb1, ltrb2):
        l = torch.minimum(ltrb1[:, 0][:, None], ltrb2[:, 0][None, :])
        t = torch.minimum(ltrb1[:, 1][:, None], ltrb2[:, 1][None, :])
        r = torch.maximum(ltrb1[:, 2][:, None], ltrb2[:, 2][None, :])
        b = torch.maximum(ltrb1[:, 3][:, None], ltrb2[:, 3][None, :])
        return torch.clip(r - l, min=0) * torch.clip(b - t, min=0)

    def get_giou(self, ltrb1, ltrb2):
        ltrb1_area = self.get_area(ltrb1)
        ltrb2_area = self.get_area(ltrb2)
        intersec_area = self.get_intersection_area(ltrb1, ltrb2)
        union_area = ltrb1_area[:, None] + ltrb2_area[None, :] - intersec_area
        enclose = self.get_smallest_enclosing_area(ltrb1, ltrb2)
        iou = torch.where(union_area == 0, 0, intersec_area / union_area)
        return torch.where(
            enclose == 0, -1, iou - ((enclose - union_area) / enclose),
        )

    def __call__(self, ltrb1, ltrb2):
        giou = self.get_giou(ltrb1, ltrb2)
        return 1 - giou
