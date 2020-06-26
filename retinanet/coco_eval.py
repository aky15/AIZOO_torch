from .cocoeval import COCOeval
import json
import torch


def evaluate_coco(dataset, model, threshold=0.05):
    
    model.eval()
    
    with torch.no_grad():

        # start collecting results
        results = []
        image_ids = []

        for index in range(len(dataset)):
            data = dataset[index]
            scale = data['scale']
            # run network
            if torch.cuda.is_available():
                scores, labels, boxes = model(data['img'].permute(2, 0, 1).cuda().float().unsqueeze(dim=0))
            else:
                scores, labels, boxes = model(data['img'].permute(2, 0, 1).float().unsqueeze(dim=0))
            scores = scores.cpu()
            labels = labels.cpu()
            boxes  = boxes.cpu()
            # correct boxes for image scale
            boxes /= scale

            if boxes.shape[0] > 0:
                # change to (x, y, w, h) (MS COCO standard)
                boxes[:, 2] -= boxes[:, 0]
                boxes[:, 3] -= boxes[:, 1]

                # compute predicted labels and scores
                #for box, score, label in zip(boxes[0], scores[0], labels[0]):
                for box_id in range(boxes.shape[0]):
                    score = float(scores[box_id])
                    label = int(labels[box_id])
                    box = boxes[box_id, :]

                    # scores are sorted, so we can break
                    if score < threshold:
                        break

                    # append detection for each positively labeled class
                    image_result = {
                        'image_id'    : dataset.imgs[index],                      
                        'category_id' : label,
                        'score'       : float(score),
                        'bbox'        : box.tolist(),
                    }

                    # append detection to results
                    results.append(image_result)

            # append image to list of processed images
            image_ids.append(dataset.imgs[index])

            # print progress
            print('{}/{}'.format(index, len(dataset)), end='\r')

        if not len(results):
            return

        # write output
        json.dump(results, open('bbox_results.json', 'w'), indent=4)
        # load results in COCO evaluation tool
        aizoo_true = dataset.aizoo_test
        aizoo_pred = aizoo_true.loadRes('bbox_results.json')

        # run COCO evaluation
        #exit()
        coco_eval = COCOeval(aizoo_true, aizoo_pred, 'bbox')
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        for iouthrs in [0.5,0.7,0.9]:
            idx = int((iouthrs-0.5)/0.05)
            print("face  mAP @ {}: {}".format(iouthrs,coco_eval.eval['precision'][idx,0:101:10,0,0,2].mean()))
            print("face mask mAP @ {}: {}".format(iouthrs,coco_eval.eval['precision'][idx,0:101:10,1,0,2].mean()))          
            #print("face precision @ {}: {}".format(iouthrs,coco_eval.eval['precision'][idx,:,0,0,2]))
            #print("face mask precision @ {}: {}".format(iouthrs,coco_eval.eval['precision'][idx,:,1,0,2]))
        print("face mean mAP: {}".format(coco_eval.eval['precision'][:,0:101:10,0,0,2].mean()))
        print("face mask mean mAP: {}".format(coco_eval.eval['precision'][:,0:101:10,1,0,2].mean()))


        model.train()

        return
