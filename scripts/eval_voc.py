import os,sys
import numpy

def parse_ground_truth(filename):
    """ Parse a PASCAL VOC xml file """
    objects = []
    filename = filename.replace('test_image', 'labels')
    filename = filename.replace('.jpg', '.txt')
    with open(filename, 'r') as f:
        lines = f.readlines()
    ground_truth = [x.strip() for x in lines]
    for obj in ground_truth:
        obj = obj.split(' ')
        obj_struct = {}
        obj_struct['class'] = int(obj[0])
        obj_struct['bbox'] = [float(obj[1]) - float(obj[3]) / 2, float(obj[2]) - float(obj[4]) / 2,
                              float(obj[1]) + float(obj[3]) / 2, float(obj[2]) + float(obj[4]) / 2]
        objects.append(obj_struct)

    return objects

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in numpy.arange(0., 1.1, 0.1):
            if numpy.sum(rec >= t) == 0:
                p = 0
            else:
                p = numpy.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = numpy.concatenate(([0.], rec, [1.]))
        mpre = numpy.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = numpy.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = numpy.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = numpy.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def voc_eval(detpath, annopath, imagesetfile, classname, classes_all, ovthresh=0.5, use_07_metric=False):
    """
    Top level function that does the PASCAL VOC evaluation.
    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    # load ground_truth
    ground_truth = {}
    for i, imagename in enumerate(imagenames):
        ground_truth[imagename] = parse_ground_truth(imagename)
        if i % 100 == 0:
            print('Reading annotation for {:d}/{:d}'.format(i + 1, len(imagenames)))

    # extract gt objects for this class
    class_ground_truth = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in ground_truth[imagename] if obj['class'] == classname]
        bbox = numpy.array([x['bbox'] for x in R])
        det = [False] * len(R)
        npos += len(bbox)
        class_ground_truth[imagename] = {'bbox': bbox, 'det': det}

    # read dets
    detfile = detpath.format(classes_all[classname])
    with open(detfile, 'r') as f:
        lines = f.readlines()
    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = numpy.array([float(x[1]) for x in splitlines])
    detect_bboxes = numpy.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = numpy.argsort(-confidence)
    sorted_scores = numpy.sort(-confidence)
    detect_bboxes = detect_bboxes[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    tp = numpy.zeros(len(image_ids))
    fp = numpy.zeros(len(image_ids))
    for d in range(len(image_ids)):
        detect_bbox = detect_bboxes[d, :].astype(float)
        ovmax = -numpy.inf
        _ground_truth = class_ground_truth[image_ids[d]]
        _ground_truth_bbox = _ground_truth['bbox'].astype(float)

        #print _ground_truth
        if _ground_truth_bbox.size > 0:
            # intersection
            ixmin = numpy.maximum(_ground_truth_bbox[:, 0], detect_bbox[0])
            iymin = numpy.maximum(_ground_truth_bbox[:, 1], detect_bbox[1])
            ixmax = numpy.minimum(_ground_truth_bbox[:, 2], detect_bbox[2])
            iymax = numpy.minimum(_ground_truth_bbox[:, 3], detect_bbox[3])
            iw = numpy.maximum(ixmax - ixmin, 0.)
            ih = numpy.maximum(iymax - iymin, 0.)
            inters = iw * ih

            # union
            uni = ((detect_bbox[2] - detect_bbox[0]) * (detect_bbox[3] - detect_bbox[1]) +
                   (_ground_truth_bbox[:, 2] - _ground_truth_bbox[:, 0]) * (_ground_truth_bbox[:, 3] - _ground_truth_bbox[:, 1]) - inters)

            overlaps = inters / uni
            ovmax = numpy.max(overlaps)
            jmax = numpy.argmax(overlaps)

        if ovmax > ovthresh:
            if not _ground_truth['det'][jmax]:
                tp[d] = 1.
                _ground_truth['det'][jmax] = 1
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = numpy.cumsum(fp)
    tp = numpy.cumsum(tp)
    print 'precide: ', tp[-1], '/', fp[-1], tp[-1] / (tp[-1] + fp[-1])
    print 'positive num: ', npos, 'detect bbox numx: ', len(image_ids), 'recall', tp[-1] / npos
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    prec = tp / numpy.maximum(tp + fp, numpy.finfo(numpy.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap
    

if __name__ == '__main__':
    classes = ['face', ]
    res_prefix = 'results/' + 'comp4_det_test_'
    filename = res_prefix + '{:s}.txt'
    annopath = '/var/darknet/FDDB_reorder/labels.txt'
    imagesetfile = '/var/darknet/FDDB_reorder/test.txt'
    for cls in range(0, len(classes)):
        rec, prec, ap = voc_eval(filename, annopath, imagesetfile, cls, classes, ovthresh=0.5)
        # print 'recall: ', rec
        # print 'precise: ', prec
        print 'ap: ', ap
