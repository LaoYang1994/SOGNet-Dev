import io
import logging
import os
import numpy as np
from PIL import Image

from detectron2.evaluation import COCOPanopticEvaluator

logger = logging.getLogger(__name__)


class COCOPanopticEvaluatorWith2ChPNG(COCOPanopticEvaluator):
    """
    Evaluate Panoptic Quality metrics on COCO using PanopticAPI.
    It saves panoptic segmentation prediction in `output_dir`

    It contains a synchronize call and has to be called from all workers.
    """

    def __init__(self, dataset_name, output_dir, gen_png=False):
        """
        Args:
            dataset_name (str): name of the dataset
            output_dir (str): output directory to save results for evaluation
        """
        super(COCOPanopticEvaluatorWith2ChPNG, self).__init__(dataset_name, output_dir)
        self.output_dir = output_dir
        self.gen_png = gen_png

    def process(self, inputs, outputs):
        from panopticapi.utils import id2rgb

        for input, output in zip(inputs, outputs):
            panoptic_img, segments_info = output["panoptic_seg"]
            panoptic_img = panoptic_img.cpu().numpy()

            file_name = os.path.basename(input["file_name"])
            file_name_png = os.path.splitext(file_name)[0] + ".png"
            with io.BytesIO() as out:
                Image.fromarray(id2rgb(panoptic_img)).save(out, format="PNG")
                segments_info = [self._convert_category_id(x) for x in segments_info]
                if self.gen_png:
                    png = self.gen_2ch_pngs(panoptic_img, segments_info)
                    Image.fromarray(png).save(
                        os.path.join(self.output_dir, file_name_png))
                self._predictions.append(
                    {
                        "image_id": input["image_id"],
                        "file_name": file_name_png,
                        "png_string": out.getvalue(),
                        "segments_info": segments_info,
                    }
                )

    def gen_2ch_pngs(self, pan_seg, seg_infos):
        png = np.zeros(pan_seg.shape + (3, ), dtype=np.uint8)
        for seg_info in seg_infos:
            seg_id = seg_info['id']
            mask = pan_seg == seg_id
            png[..., 0][mask] = seg_info['category_id']
            if seg_info['isthing']:
                png[..., 1][mask] = seg_info['instance_id']

        return png
