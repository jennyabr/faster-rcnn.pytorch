import time
import pickle
import logging
import numpy as np
import cv2


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def faster_rcnn_visualization(data_manager, cfg, epoch_num):
    pp_preds_path = cfg.get_postprocessed_preds_path(epoch_num)
    with open(pp_preds_path, 'rb') as f:
        bboxs = pickle.load(f)

    start_time = time.time()
    bboxs = bboxs.numpy()  # TODO JA maybe this needed?
    for i in range(data_manager.num_images):
        im = cv2.imread(data_manager.imdb.image_path_at(i))
        im2show = np.copy(im)
        for j in range(1, data_manager.num_classes):
            cls_bboxs = bboxs[i][j]  # TODO JA check this
            for x in range(np.minimum(10, cls_bboxs.shape[0])):
                bbox = tuple(int(np.round(x)) for x in cls_bboxs[x, :4])
                score = cls_bboxs[x, -1]
                if score > 0.3:
                    cv2.rectangle(im, bbox[0:2], bbox[2:4], (0, 204, 0), 2)
                    class_name = data_manager.imdb.classes[j]
                    cv2.putText(im, '%s: %.3f' % (class_name, score), (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                                1.0, (0, 0, 255), thickness=1)

        cv2.imwrite(cfg.get_visualizations_path(epoch_num, i), im2show)

    end_time = time.time()
    logger.info("Visualization time: {:.4f}s".format(end_time - start_time))
