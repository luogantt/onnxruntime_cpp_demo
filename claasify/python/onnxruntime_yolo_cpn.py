import numpy as np
import glob
import onnxruntime
import cv2
import os


def resize(image, _resize_size_hw,lb=False):
    h, w = image.shape[:2]
    if w == _resize_size_hw[1] and h == _resize_size_hw[0]:
        scale = 1.0
        pad = (0, 0, 0, 0)
    else:
        if w / _resize_size_hw[1] >= h / _resize_size_hw[0]:
            scale = _resize_size_hw[1] / w
        else:
            scale = _resize_size_hw[0] / h
        new_w = int(w * scale)
        new_h = int(h * scale)
        if new_w == _resize_size_hw[1] and new_h == _resize_size_hw[0]:
            pad = (0, 0, 0, 0)
        else:
            if lb:
                pad_w = _resize_size_hw[1] - new_w
                pad_h = _resize_size_hw[0] - new_h
                pad = (0, int(pad_h + .5), 0, int(pad_w + .5))
            else:
                pad_w = (_resize_size_hw[1] - new_w) / 2
                pad_h = (_resize_size_hw[0] - new_h) / 2
                pad = (int(pad_h), int(pad_h + .5), int(pad_w), int(pad_w + .5))

    parameter_dict = {}
    parameter_dict['scale'] = scale
    parameter_dict['pad_tblr'] = pad
    parameter_dict['scale_offset_hw'] = (0, 0)

    if parameter_dict['scale'] != 1:
        new_w = int(w * parameter_dict['scale'])
        new_h = int(h * parameter_dict['scale'])
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        scale_offset_w = new_w / w - parameter_dict['scale']
        scale_offset_h = new_h / h - parameter_dict['scale']
        parameter_dict['scale_offset_hw'] = (scale_offset_h, scale_offset_w)

    top = parameter_dict['pad_tblr'][0]
    bottom = parameter_dict['pad_tblr'][1]
    left = parameter_dict['pad_tblr'][2]
    right = parameter_dict['pad_tblr'][3]

    pad_img = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    return pad_img, parameter_dict


def _convert_origshape(boxes, parameter_dict):
    top = parameter_dict['pad_tblr'][0]
    left = parameter_dict['pad_tblr'][2]
    scale = parameter_dict['scale']
    boxes[:, 0:3:2] -= left
    boxes[:, 1:4:2] -= top
    boxes[:, :4] /= scale
    return boxes


def _generate_batch_data(image, bboxes, lables, batch_size=5, size=(320, 320)):
    batch_patch = []
    batch_bbox=[]
    batch_dict=[]
    for i,(bbox,label) in enumerate(zip(bboxes,lables)):
        if np.any(bbox < 0): continue
        if label == 1: continue
        bbox_f = np.array(bbox[:4], np.int32)
        patch = image[bbox_f[1]:bbox_f[3], bbox_f[0]:bbox_f[2]]
        img, cpn_para_dict = resize(patch, size)
        # cv2.namedWindow("img", 0)
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        img1 = np.transpose(img, (2, 0, 1))
        img1 = np.float32(img1)
        batch_patch.append(img1)
        batch_bbox.append(bbox)
        batch_dict.append(cpn_para_dict)
        if len(batch_patch) == batch_size:
            yield [batch_patch, batch_bbox,batch_dict]
            batch_patch = []
            batch_bbox = []
            batch_dict = []
    if len(batch_bbox) > 0:
        yield [batch_patch, batch_bbox,batch_dict]


if __name__ == '__main__':

    root = '/home/cxj/Downloads/data/test'

    img_name_list = glob.glob(root + os.sep + '*')
    img_name_list = list(filter(lambda f: f.find('json') < 0, img_name_list))

    ort_session = onnxruntime.InferenceSession("/home/cxj/Downloads/data/yolo.onnx")
    cpn_session = onnxruntime.InferenceSession("/home/cxj/Desktop/onnx_deploy/onnxruntime_yolo_cpn/weights/cpn_new.onnx")
    num_classes = ['ok', 'ng']

    for img_path in img_name_list:
        print("img_path = ", img_path)
        img = cv2.imread(img_path)
        image_copy = img.copy()
        cpn_img = image_copy / 255
        img, para_dict = resize(img, (448, 448),lb=True)
        img = img / 255
        img = np.float32(img)
        img1 = np.transpose(img, (2, 0, 1))[None]
        ort_inputs = {ort_session.get_inputs()[0].name: img1}
        ort_outs = ort_session.run(None, ort_inputs)
        bboxes = ort_outs[0][:, :5]  # nx5
        lables = ort_outs[0][:, 5:]  # nx1

        # revert to orig img
        bboxes = _convert_origshape(bboxes, para_dict)  # nx4
        index = bboxes[:, 4] > 0.1
        bboxes = bboxes[index]

        g_batch_data = _generate_batch_data(cpn_img, bboxes, lables,batch_size=32, size=(256, 256))

        for batch in g_batch_data:
            batch_patch, batch_bbox, batch_dict=batch
            batch_patch=np.array(batch_patch)

            ort_inputs = {cpn_session.get_inputs()[0].name: batch_patch}
            array_bcwh = cpn_session.run(None, ort_inputs)[0]  # b,c,h,w

            h, w = array_bcwh.shape[2], array_bcwh.shape[3]
            array_bcwh = np.reshape(array_bcwh, [array_bcwh.shape[0], array_bcwh.shape[1], -1])
            index = np.argmax(array_bcwh, axis=2)
            y, x = np.unravel_index(index, (h, w))
            pts_loc_bxcx2 = np.stack([x, y], axis=2).astype(np.float32)
            pts_bxcx1 = np.max(array_bcwh, axis=2)[..., None]

            pts_loc_bxcx2 *= 4

            for i,pts_loc_cx2 in enumerate(pts_loc_bxcx2):
                cpn_para_dict=batch_dict[i]
                bbox_f=batch_bbox[i]
                top = cpn_para_dict['pad_tblr'][0]
                left = cpn_para_dict['pad_tblr'][2]
                scale = cpn_para_dict['scale']
                pts_loc_cx2[..., 0] = pts_loc_cx2[..., 0] - left
                pts_loc_cx2[..., 1] = pts_loc_cx2[..., 1] - top
                pts_loc_cx2 = pts_loc_cx2 / scale

                pts_loc_cx2[..., 0] += bbox_f[0]
                pts_loc_cx2[..., 1] += bbox_f[1]

                pts_loc_cx2 = np.array(pts_loc_cx2, np.int64)
                for keypoint in pts_loc_cx2:
                    cv2.circle(image_copy, tuple(keypoint), radius=6, color=(0, 0, 255), thickness=-1)

        for i,(bbox,label) in enumerate(zip(bboxes,lables)):
            bbox_f = np.array(bbox[:4], np.int32)
            if label==0:
                image_copy = cv2.rectangle(image_copy, (bbox_f[0], bbox_f[1]), (bbox_f[2], bbox_f[3]), (255, 0, 0),
                                           5)
            else:
                image_copy = cv2.rectangle(image_copy, (bbox_f[0], bbox_f[1]), (bbox_f[2], bbox_f[3]), (0, 0, 255),
                                           5)

        cv2.namedWindow('img',0 )
        cv2.imshow('img', image_copy)
        cv2.waitKey(0)
