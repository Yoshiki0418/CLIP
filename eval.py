import numpy as np
import torch
import hydra
from omegaconf import DictConfig
import albumentations as A
from tqdm import tqdm
from transformers import AutoTokenizer

from src.dataset_info import imagenet_templates, imagenet_classes
from src.metrics import calc_acc


from src.model import CLIP_Module
from datasets import load_dataset

def preproc_image(image, transforms, device):
    image = transforms(image=np.array(image))['image']
    image = torch.tensor(image).permute(2, 0, 1).float().to(device)
    image = image.unsqueeze(0)
    return image

def zeroshot_classifier(labels, templates, model, tokenizer, device, max_length=100):
    zeroshot_weights = []
    for label in tqdm(labels):
        texts = [template.format(label) for template in templates]
        texts = tokenizer(
            text=texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt').to(device)
        text_embeddings = model.encode_text(texts['input_ids'], texts['attention_mask'])
        text_embeddings = text_embeddings.mean(dim=0)
        text_embeddings /= text_embeddings.norm()
        text_embeddings = text_embeddings.cpu().detach().numpy()
        zeroshot_weights.append(text_embeddings)
    zeroshot_weights = np.stack(zeroshot_weights, axis=1)
    return zeroshot_weights

@hydra.main(version_base=None,config_path='configs',config_name='config')
def run(args: DictConfig):

    # -----dataLoader-----
    imagenette = load_dataset(
        'frgfm/imagenette',
        '320px',
        split='validation',
        revision='4d512db'
    )

    target_size = 224
    transforms = A.Compose(
        [
            A.Resize(target_size, target_size, always_apply=True),
            A.Normalize(max_pixel_value=255.0, always_apply=True),
        ]
    )
    tokenizer = AutoTokenizer.from_pretrained(args.datasets.tokenizer_alias)

    # -----model-----
    model = CLIP_Module(**args.model).to(args.device)
    model.load_state_dict(torch.load('/content/drive/MyDrive/Deep_Learning/CLIP/outputs/2024-12-15/06-24-11/model_best.pt',map_location=torch.device('cpu'))) 

    # Prompt ensembling
    labels = imagenette.info.features['label'].names
    templates = imagenet_templates
    zeroshot_weights = zeroshot_classifier(labels, templates, model, tokenizer, args.device)


    # -----推論-----
    preds = {}
    for i in tqdm(range(len(imagenette))):
        if imagenette[i]['image'].mode != 'RGB':
            continue
        image = preproc_image(imagenette[i]['image'], transforms=transforms, device=args.device)
        image_embeddings = model.encode_image(image)
        image_embeddings = image_embeddings.cpu().detach().numpy()
        score = np.dot(image_embeddings, zeroshot_weights)
        pred = np.argmax(score)
        preds[i] = pred

    true_preds = []
    true_preds_per_class = {i: [] for i in range(len(labels))}
    for i, label in enumerate(imagenette['label']):
        if i not in preds:
            continue
        if label == preds[i]:
            true_preds.append(1)
            true_preds_per_class[label].append(1)
        else:
            true_preds.append(0)
            true_preds_per_class[label].append(0)
    print(f'Accuracy: {calc_acc(true_preds)}')
    for label, preds in true_preds_per_class.items():
        print(f'{labels[label]},{calc_acc(preds)}')

if __name__ == "__main__":
    run()