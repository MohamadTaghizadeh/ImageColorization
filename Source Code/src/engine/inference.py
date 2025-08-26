import cv2
import torch 
from torchvision import transforms


def process_images(context_norm, body_norm, image_context_path=None, image_context=None, image_body=None, bbox=None):

  if image_context is None and image_context_path is None:
    raise ValueError('both image_context and image_context_path cannot be none. Please specify one of the two.')
  if image_body is None and bbox is None: 
    raise ValueError('both body image and bounding box cannot be none. Please specify one of the two')

  if image_context_path is not None:
    image_context =  cv2.cvtColor(cv2.imread(image_context_path), cv2.COLOR_BGR2RGB)
  
  if bbox is not None:
    image_body = image_context[bbox[1]:bbox[3],bbox[0]:bbox[2]].copy()
  
  image_context = cv2.resize(image_context, (224,224))
  image_body = cv2.resize(image_body, (128,128))
  
  test_transform = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()])
  context_norm = transforms.Normalize(context_norm[0], context_norm[1])  
  body_norm = transforms.Normalize(body_norm[0], body_norm[1])

  image_context = context_norm(test_transform(image_context)).unsqueeze(0)
  image_body = body_norm(test_transform(image_body)).unsqueeze(0)

  return image_context, image_body  


def infer(context_norm, body_norm, ind2cat, device, thresholds, models, image_context_path=None, image_context=None, image_body=None, bbox=None, to_print=True):

  image_context, image_body = process_images(context_norm, body_norm, image_context_path=image_context_path, image_context=image_context, image_body=image_body, bbox=bbox)

  model_context, model_body, emotic_model = models
  
  with torch.no_grad():
    image_context = image_context.to(device)
    image_body = image_body.to(device)
    
    pred_context = model_context(image_context)
    pred_body = model_body(image_body)
    pred_cat, pred_cont = emotic_model(pred_context, pred_body)
    pred_cat = pred_cat.squeeze(0)
    pred_cont = pred_cont.squeeze(0).to("cpu").data.numpy()

    bool_cat_pred = torch.gt(pred_cat, thresholds)
  
  cat_emotions = list()
  for i in range(len(bool_cat_pred)):
    if bool_cat_pred[i] == True:
      cat_emotions.append(ind2cat[i])

  if to_print == True:
    print ('\n Image predictions')
    print ('Continuous Dimnesions Predictions') 
    for i in range(len(pred_cont)):
      print ('Continuous %10s %.5f' %(ind2vad[i], 10*pred_cont[i]))
    print ('Categorical Emotion Predictions')
    for emotion in cat_emotions:
      print ('Categorical %16s' %(emotion))
  
  return cat_emotions, 10*pred_cont

